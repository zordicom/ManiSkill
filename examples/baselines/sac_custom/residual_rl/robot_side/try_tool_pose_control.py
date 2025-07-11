#!/usr/bin/env python3
"""
Copyright 2025 Zordi, Inc. All rights reserved.

Enhanced tool pose control with object detection and pose estimation.
This script detects the pick target object and moves the arm to 15 cm above it.
"""

import os
import threading

import cv2
import numpy as np
import rclpy
from calibration_utils import CalibrationData, load_calibration
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image
from utils import adjust_k_for_crop_and_resize, center_crop_and_resize_image
from ws_det_client import ObjectDetectionClient
from ws_pose_multi_client import MultiObjectFoundationPoseClient

# Configuration constants
DETECTION_SERVER_HOST = os.getenv("DETECTION_SERVER_HOST", "yk-dev-4090")
DETECTION_SERVER_PORT = int(os.getenv("DETECTION_SERVER_PORT", "10015"))
POSE_SERVER_HOST = os.getenv("POSE_SERVER_HOST", "yk-dev-4090")
POSE_SERVER_PORT = int(os.getenv("POSE_SERVER_PORT", "10014"))
DEFAULT_CALIBRATION_FILE = "camera_calibration.json"

# Object detection and pose estimation configuration
OBJECT_NAME_MAPPING = {
    "pick_target": {
        "detection_name": "small white box",  # Name used for object detection
        "pose_name": "b5box",  # Name used for pose estimation model
    },
}

# Custom rotations to apply to estimated poses (in object coordinate system)
# Values are (x, y, z) euler angles in degrees
CUSTOM_ROTATIONS = {
    "pick_target": (0, 90, 0),  # Rotation for box
}

# Image size for object detection and pose estimation
DETECTION_POSE_IMAGE_SIZE = 480

# Pose tracking configuration
POSE_CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence to accept pose predictions
QUEUE_SIZE = 10

# Safety boundaries for end effector pose (in meters)
SAFETY_BOUNDS = {
    "x": (0.1, 0.5),
    "y": (-0.2, 0.2),
    "z": (0.05, 0.6),
}

# Height offset above pick target (15 cm)
TARGET_HEIGHT_OFFSET = 0.15


class PickTargetPoseController(Node):
    """ROS node that detects pick target and controls arm pose accordingly."""

    def __init__(self):
        super().__init__("pick_target_pose_controller")

        # WebSocket clients for object detection and pose estimation
        self.detection_client: ObjectDetectionClient | None = None
        self.pose_client: MultiObjectFoundationPoseClient | None = None

        # Camera calibration data
        self.calibration_data: CalibrationData | None = None
        self._load_calibration()

        # Initialize WebSocket clients
        self._init_websocket_clients()

        # Camera topics
        self.camera_topics = {
            "static_top_rgb": "/camera/static_rs405_top/color/image_rect_raw",
            "static_top_depth": "/camera/static_rs405_top/depth/image_rect_raw",
        }

        # Initialize image storage and camera intrinsics
        self.images: dict[str, np.ndarray | None] = {}
        self.camera_k_matrices: dict[str, list[float]] = {}
        self.bridge = CvBridge()

        # Create camera subscribers
        self.subscribers = []
        for name, topic in self.camera_topics.items():
            self.images[name] = None
            self.subscribers.append(
                self.create_subscription(
                    Image, topic, self._create_callback_for_camera(name), QUEUE_SIZE
                )
            )
            self.get_logger().info(f"Subscribed to camera topic: {topic} as {name}")

        # Camera info subscriber for RGB camera only
        self.camera_info_topic = "/camera/static_rs405_top/color/camera_info"
        self.camera_info_subscriber = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            QUEUE_SIZE,
        )
        self.get_logger().info(
            f"Subscribed to camera info topic: {self.camera_info_topic}"
        )

        # End-effector pose publisher
        self.publisher_ = self.create_publisher(
            PoseStamped, "/a1_ee_target_follower_r", 10
        )

        # Timer for continuous pose updates (1 Hz)
        self.timer = self.create_timer(1.0, self.update_target_pose)

        # Object tracking state
        self.object_tracking_ready = False
        self.pose_tracking_initialized = False
        self.current_pick_target_pose: list[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

        # Thread management for pose tracking
        self.pose_tracking_lock = threading.Lock()
        self.pose_tracking_thread: threading.Thread | None = None
        self.pose_tracking_shutdown = threading.Event()
        self.pose_tracking_trigger = threading.Event()
        self.pose_tracking_in_progress = threading.Event()

        # WebSocket context managers for persistent connections
        self.detection_client_context = None
        self.pose_client_context = None

        # Initialization state
        self.initialization_attempted = False

        self.get_logger().info("PickTargetPoseController initialized")
        self.get_logger().info(
            f"Detection server: {DETECTION_SERVER_HOST}:{DETECTION_SERVER_PORT}"
        )
        self.get_logger().info(f"Pose server: {POSE_SERVER_HOST}:{POSE_SERVER_PORT}")

    def _load_calibration(self) -> None:
        """Load camera calibration data from JSON file."""
        try:
            self.calibration_data = load_calibration(DEFAULT_CALIBRATION_FILE)
            self.get_logger().info(f"✅ Loaded calibration: {self.calibration_data}")
        except Exception as e:
            self.get_logger().error(
                f"❌ Failed to load calibration from {DEFAULT_CALIBRATION_FILE}: {e}"
            )
            self.get_logger().warning("Object poses will remain in camera coordinates")
            self.calibration_data = None

    def _init_websocket_clients(self) -> None:
        """Initialize WebSocket clients for object detection and pose estimation."""
        try:
            # Initialize detection client
            self.detection_client = ObjectDetectionClient(
                DETECTION_SERVER_HOST, DETECTION_SERVER_PORT
            )
            self.get_logger().info(
                f"Detection client initialized ({DETECTION_SERVER_HOST}:{DETECTION_SERVER_PORT})"
            )

            # Initialize pose estimation client
            self.pose_client = MultiObjectFoundationPoseClient(
                POSE_SERVER_HOST, POSE_SERVER_PORT
            )
            self.get_logger().info(
                f"Pose client initialized ({POSE_SERVER_HOST}:{POSE_SERVER_PORT})"
            )

        except Exception as e:
            self.get_logger().error(f"Failed to initialize WebSocket clients: {e}")
            self.detection_client = None
            self.pose_client = None

    def _create_callback_for_camera(self, camera_name: str):
        """Create a callback function for a specific camera."""

        def callback(msg: Image) -> None:
            self.image_callback(msg, camera_name)

        return callback

    def image_callback(self, msg: Image, camera_name: str) -> None:
        """Process camera images."""
        try:
            # Convert ROS Image message to numpy array
            if "depth" in camera_name:
                # For depth images, use passthrough to preserve original data
                cv_image = self.bridge.imgmsg_to_cv2(
                    msg, desired_encoding="passthrough"
                )
                cv_image = cv_image.astype(np.uint16)
            else:
                # RGB images: convert from ROS BGR format to RGB for storage
                cv_image_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                cv_image = cv2.cvtColor(cv_image_bgr, cv2.COLOR_BGR2RGB)

            # Center crop to 480x480 for object detection and pose estimation
            cv_image = center_crop_and_resize_image(
                cv_image,
                image_size=(DETECTION_POSE_IMAGE_SIZE, DETECTION_POSE_IMAGE_SIZE),
            )

            self.images[camera_name] = cv_image

        except Exception as e:
            self.get_logger().error(f"Error processing camera image {camera_name}: {e}")

    def camera_info_callback(self, msg: CameraInfo) -> None:
        """Process camera info messages and extract K matrix."""
        try:
            # Extract K matrix and adjust for center crop
            if len(msg.k) == 9:
                k_matrix = np.array(msg.k).reshape(3, 3)
                orig_size = (int(msg.width), int(msg.height))

                # K matrix for pose estimation (480x480 center crop from 640x480)
                pose_estimation_size = (
                    DETECTION_POSE_IMAGE_SIZE,
                    DETECTION_POSE_IMAGE_SIZE,
                )

                adjusted_k = adjust_k_for_crop_and_resize(
                    k_matrix, orig_size, pose_estimation_size, pad_ratio=0.0
                )

                # Store the adjusted K matrix for pose estimation
                self.camera_k_matrices["static_top_rgb"] = adjusted_k.flatten().tolist()
            else:
                self.get_logger().warning(
                    f"Invalid K matrix size: expected 9, got {len(msg.k)}"
                )
        except Exception as e:
            self.get_logger().error(f"Error processing camera info: {e}")

    def _enter_websocket_contexts(self) -> bool:
        """Enter WebSocket context managers for persistent connections."""
        try:
            # Enter detection client context
            if self.detection_client is not None:
                self.detection_client_context = self.detection_client.__enter__()
                self.get_logger().info(
                    "✅ Detection client context entered successfully"
                )

            # Enter pose client context
            if self.pose_client is not None:
                self.pose_client_context = self.pose_client.__enter__()
                self.get_logger().info("✅ Pose client context entered successfully")

            return True

        except Exception as e:
            self.get_logger().error(f"Failed to enter WebSocket contexts: {e}")
            self._exit_websocket_contexts()
            return False

    def _exit_websocket_contexts(self) -> None:
        """Exit WebSocket context managers."""
        # Exit pose client context
        if self.pose_client_context is not None:
            try:
                self.pose_client.__exit__(None, None, None)
                if rclpy.ok():
                    self.get_logger().info("Pose client context exited successfully")
            except Exception as e:
                if rclpy.ok():
                    self.get_logger().warning(f"Error exiting pose client context: {e}")
            finally:
                self.pose_client_context = None

        # Exit detection client context
        if self.detection_client_context is not None:
            try:
                self.detection_client.__exit__(None, None, None)
                if rclpy.ok():
                    self.get_logger().info(
                        "Detection client context exited successfully"
                    )
            except Exception as e:
                if rclpy.ok():
                    self.get_logger().warning(
                        f"Error exiting detection client context: {e}"
                    )
            finally:
                self.detection_client_context = None

    def initialize_object_tracking(self) -> bool:
        """Initialize object detection and pose tracking."""
        if self.detection_client is None or self.pose_client is None:
            self.get_logger().error("WebSocket clients not initialized")
            return False

        # Check if camera data is available
        static_rgb = self.images.get("static_top_rgb")
        static_depth = self.images.get("static_top_depth")
        static_k = self.camera_k_matrices.get("static_top_rgb")

        if static_rgb is None or static_depth is None or static_k is None:
            self.get_logger().error("Camera data not available for initialization")
            return False

        self.get_logger().info("✅ All required camera data available!")

        # Enter WebSocket contexts for persistent connections
        if not self._enter_websocket_contexts():
            return False

        camera_intrinsics = np.array(static_k).reshape(3, 3)

        # Perform object detection for pick target
        target_key = "pick_target"
        names = OBJECT_NAME_MAPPING[target_key]
        detection_name = names["detection_name"]
        pose_name = names["pose_name"]

        self.get_logger().info(f"Detecting {target_key}: '{detection_name}'...")

        try:
            # Perform object detection
            success, message, boxes, scores, labels, mask = (
                self.detection_client.detect_objects(static_rgb, detection_name)
            )

            if not success or len(boxes) == 0:
                self.get_logger().warning(
                    f"Object detection failed for {detection_name}: {message}"
                )
                return False

            self.get_logger().info(
                f"Detected {len(boxes)} instances of '{detection_name}' "
                f"with max confidence {max(scores):.3f}"
            )

            # Use the highest confidence detection
            best_idx = np.argmax(scores)
            best_mask = (
                mask
                if mask is not None
                else np.ones_like(static_rgb[:, :, 0], dtype=np.uint8)
            )

            # Ensure mask is in correct format for pose estimation
            if best_mask.dtype != np.uint8:
                if best_mask.max() <= 1.0:
                    best_mask = (best_mask * 255).astype(np.uint8)
                else:
                    best_mask = best_mask.astype(np.uint8)

            if best_mask.max() == 1:
                best_mask *= 255

            # Prepare object data for pose estimation
            objects_data = [
                {
                    "object_name": pose_name,
                    "rgb_image": static_rgb,
                    "depth_image": static_depth,
                    "mask_image": best_mask,
                    "camera_intrinsics": camera_intrinsics,
                }
            ]

            # Initialize pose estimation
            self.get_logger().info("Initializing pose estimation...")
            init_response = self.pose_client.initialize_objects(objects_data)

            if not init_response.get("success", False):
                self.get_logger().error(
                    f"Pose initialization failed: {init_response.get('message', 'Unknown error')}"
                )
                return False

            self.get_logger().info(
                f"✅ Pose initialization successful: {init_response.get('message', '')}"
            )

            # Extract initial pose
            initial_poses_7d = init_response.get("poses", {})
            if pose_name in initial_poses_7d:
                pose_7d = initial_poses_7d[pose_name]

                # Apply custom rotation
                modified_pose_7d = self.apply_custom_rotation(pose_7d, target_key)

                # Apply additional rotation regularization for stability
                regularized_pose_7d = self.apply_rotation_regularization(
                    modified_pose_7d
                )

                # Transform from camera coordinates to world coordinates if calibration is available
                if self.calibration_data is not None:
                    world_pose_7d = (
                        self.calibration_data.transform_pose_camera_to_world(
                            regularized_pose_7d
                        )
                    )
                    self.current_pick_target_pose = world_pose_7d
                    self.get_logger().info(
                        f"Transformed {target_key} pose to world coordinates: "
                        f"camera={regularized_pose_7d[:3]} -> world={world_pose_7d[:3]}"
                    )
                else:
                    # No calibration available, use camera coordinates directly
                    self.current_pick_target_pose = regularized_pose_7d
                    self.get_logger().warning(
                        f"No calibration available - {target_key} pose remains in camera coordinates"
                    )

                self.pose_tracking_initialized = True

                # Log initial pose details
                self.log_pose_details(self.current_pick_target_pose, target_key)

                # Start pose tracking thread
                self.start_pose_tracking_thread()

                return True
            else:
                self.get_logger().warning(f"No initial pose returned for {target_key}")
                return False

        except Exception as e:
            self.get_logger().error(
                f"Error during object detection/pose estimation: {e}"
            )
            return False

    def apply_custom_rotation(
        self, pose_7d: list[float], target_key: str
    ) -> list[float]:
        """Apply custom rotation to a 7D pose in the object's coordinate system."""
        if target_key not in CUSTOM_ROTATIONS:
            return pose_7d

        custom_rotation_deg = CUSTOM_ROTATIONS[target_key]

        # Check if any rotation is specified
        if all(abs(rot) < 1e-6 for rot in custom_rotation_deg):
            return pose_7d

        try:
            # Convert 7D pose to transformation components
            translation = np.array(pose_7d[:3])
            quaternion = np.array(pose_7d[3:7])  # [qx, qy, qz, qw]

            # Get current rotation matrix
            current_rotation = Rotation.from_quat(quaternion).as_matrix()

            # Create custom rotation matrix from euler angles (in degrees)
            custom_rotation = Rotation.from_euler(
                "xyz", custom_rotation_deg, degrees=True
            ).as_matrix()

            # Apply custom rotation in object coordinate system
            new_rotation_matrix = current_rotation @ custom_rotation

            # Convert back to quaternion
            new_quaternion = Rotation.from_matrix(new_rotation_matrix).as_quat()

            # Return modified 7D pose
            return [
                translation[0],
                translation[1],
                translation[2],  # Position unchanged
                new_quaternion[0],
                new_quaternion[1],
                new_quaternion[2],
                new_quaternion[3],  # New quaternion
            ]

        except Exception as e:
            self.get_logger().warning(
                f"Failed to apply custom rotation to {target_key}: {e}"
            )
            return pose_7d

    def apply_rotation_regularization(self, pose_7d: list[float]) -> list[float]:
        """Apply additional rotation regularization for stability."""
        try:
            translation = np.array(pose_7d[:3])
            quaternion = np.array(pose_7d[3:7])  # [qx, qy, qz, qw]

            # Get current rotation matrix
            rotation_matrix = Rotation.from_quat(quaternion).as_matrix()

            # Extract axis vectors
            x_axis = rotation_matrix[:, 0]  # Object's X-axis in world frame
            z_axis = rotation_matrix[:, 2]  # Object's Z-axis in world frame

            # Regularize X-axis to point near->far (positive X direction)
            world_x = np.array([1, 0, 0])
            if np.dot(x_axis, world_x) < 0:
                # Flip X and Y axes to maintain right-handed coordinate system
                rotation_matrix[:, 0] = -rotation_matrix[:, 0]  # Flip X
                rotation_matrix[:, 1] = -rotation_matrix[:, 1]  # Flip Y
                self.get_logger().info("Applied X-axis regularization (near->far)")

            # Regularize Z-axis to point high->low (negative Z direction for downward)
            world_down = np.array([0, 0, -1])
            if np.dot(z_axis, world_down) < 0:
                # Flip Z axis and one other axis to maintain right-handed system
                rotation_matrix[:, 2] = -rotation_matrix[:, 2]  # Flip Z
                rotation_matrix[:, 1] = -rotation_matrix[
                    :, 1
                ]  # Flip Y to maintain handedness
                self.get_logger().info("Applied Z-axis regularization (high->low)")

            # Convert back to quaternion
            regularized_quaternion = Rotation.from_matrix(rotation_matrix).as_quat()

            return [
                translation[0],
                translation[1],
                translation[2],
                regularized_quaternion[0],
                regularized_quaternion[1],
                regularized_quaternion[2],
                regularized_quaternion[3],
            ]

        except Exception as e:
            self.get_logger().warning(f"Failed to apply rotation regularization: {e}")
            return pose_7d

    def log_pose_details(self, pose_7d: list[float], target_key: str) -> None:
        """Log detailed pose information."""
        translation = np.array(pose_7d[:3])
        quaternion = np.array(pose_7d[3:7])
        rotation = Rotation.from_quat(quaternion)
        euler_angles = rotation.as_euler("xyz", degrees=True)

        self.get_logger().info(f"Pose details for {target_key}:")
        self.get_logger().info(
            f"  Translation: [{translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f}]"
        )
        self.get_logger().info(
            f"  Rotation (XYZ): [{euler_angles[0]:.1f}°, {euler_angles[1]:.1f}°, {euler_angles[2]:.1f}°]"
        )

        # Log axis orientations
        rotation_matrix = rotation.as_matrix()
        x_axis = rotation_matrix[:, 0]
        y_axis = rotation_matrix[:, 1]
        z_axis = rotation_matrix[:, 2]

        self.get_logger().info(
            f"  X-axis direction: [{x_axis[0]:.3f}, {x_axis[1]:.3f}, {x_axis[2]:.3f}]"
        )
        self.get_logger().info(
            f"  Y-axis direction: [{y_axis[0]:.3f}, {y_axis[1]:.3f}, {y_axis[2]:.3f}]"
        )
        self.get_logger().info(
            f"  Z-axis direction: [{z_axis[0]:.3f}, {z_axis[1]:.3f}, {z_axis[2]:.3f}]"
        )

    def start_pose_tracking_thread(self) -> None:
        """Start pose tracking worker thread."""
        if (
            self.pose_tracking_thread is None
            or not self.pose_tracking_thread.is_alive()
        ):
            self.pose_tracking_shutdown.clear()
            self.pose_tracking_thread = threading.Thread(
                target=self._pose_tracking_worker_thread,
                daemon=True,
                name="pose_tracking_worker",
            )
            self.pose_tracking_thread.start()
            self.get_logger().info("Pose tracking worker thread started")

    def _pose_tracking_worker_thread(self) -> None:
        """Worker thread that handles pose tracking predictions."""
        while not self.pose_tracking_shutdown.is_set():
            try:
                # Wait for trigger or shutdown
                if self.pose_tracking_trigger.wait(timeout=0.1):
                    self.pose_tracking_trigger.clear()

                    # Set in progress flag
                    self.pose_tracking_in_progress.set()

                    try:
                        self._perform_pose_tracking()
                    finally:
                        # Always clear the in progress flag
                        self.pose_tracking_in_progress.clear()

            except Exception as e:
                if not self.pose_tracking_shutdown.is_set():
                    self.get_logger().error(
                        f"Error in pose tracking worker thread: {e}"
                    )
                self.pose_tracking_in_progress.clear()

    def _perform_pose_tracking(self) -> None:
        """Perform the actual pose tracking prediction."""
        if not self.pose_tracking_initialized:
            return

        # Get latest images
        static_rgb = self.images.get("static_top_rgb")
        static_depth = self.images.get("static_top_depth")
        static_k = self.camera_k_matrices.get("static_top_rgb")

        # Check if we have valid images
        if static_rgb is None or static_depth is None or static_k is None:
            return

        camera_intrinsics = np.array(static_k).reshape(3, 3)

        try:
            # Use persistent connection for pose tracking
            track_response = self.pose_client.predict_poses(
                static_rgb, static_depth, None, camera_intrinsics
            )

            # Extract tracking results
            tracked_poses_7d = track_response.get("poses", {})
            tracked_confidences = track_response.get("confidences", {})

            # Update pose tracking state
            pose_name = OBJECT_NAME_MAPPING["pick_target"]["pose_name"]
            if pose_name in tracked_poses_7d:
                pose_7d = tracked_poses_7d[pose_name]
                confidence = tracked_confidences.get(pose_name, 0.0)

                if confidence >= POSE_CONFIDENCE_THRESHOLD:
                    # Apply custom rotation and regularization
                    modified_pose_7d = self.apply_custom_rotation(
                        pose_7d, "pick_target"
                    )
                    regularized_pose_7d = self.apply_rotation_regularization(
                        modified_pose_7d
                    )

                    # Transform to world coordinates if calibration is available
                    with self.pose_tracking_lock:
                        if self.calibration_data is not None:
                            world_pose_7d = (
                                self.calibration_data.transform_pose_camera_to_world(
                                    regularized_pose_7d
                                )
                            )
                            self.current_pick_target_pose = world_pose_7d
                        else:
                            self.current_pick_target_pose = regularized_pose_7d

        except Exception as e:
            if "ConnectionClosed" not in str(type(e).__name__):
                self.get_logger().warning(f"Failed to track object pose: {e}")

    def trigger_pose_tracking(self) -> None:
        """Trigger pose tracking update."""
        if (
            self.pose_tracking_initialized
            and not self.pose_tracking_in_progress.is_set()
        ):
            self.pose_tracking_trigger.set()

    def compute_target_ee_pose(self, pick_target_pose_7d: list[float]) -> PoseStamped:
        """Compute target end-effector pose 15 cm above the pick target."""
        # Extract pick target position and orientation
        target_pos = np.array(pick_target_pose_7d[:3])
        target_quat = np.array(pick_target_pose_7d[3:7])  # [qx, qy, qz, qw]

        # Compute target position 15 cm above the pick target
        ee_target_pos = target_pos.copy()
        ee_target_pos[2] += TARGET_HEIGHT_OFFSET  # Add 15 cm in Z direction

        # Apply safety boundaries
        ee_target_pos[0] = np.clip(
            ee_target_pos[0], SAFETY_BOUNDS["x"][0], SAFETY_BOUNDS["x"][1]
        )
        ee_target_pos[1] = np.clip(
            ee_target_pos[1], SAFETY_BOUNDS["y"][0], SAFETY_BOUNDS["y"][1]
        )
        ee_target_pos[2] = np.clip(
            ee_target_pos[2], SAFETY_BOUNDS["z"][0], SAFETY_BOUNDS["z"][1]
        )

        # Use the same orientation as the pick target (for now)
        # In the future, you might want to compute a specific gripper orientation
        ee_target_quat = target_quat

        # Create PoseStamped message
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = "world"
        pose_msg.header.stamp = self.get_clock().now().to_msg()

        pose_msg.pose.position.x = ee_target_pos[0]
        pose_msg.pose.position.y = ee_target_pos[1]
        pose_msg.pose.position.z = ee_target_pos[2]

        pose_msg.pose.orientation.x = ee_target_quat[0]
        pose_msg.pose.orientation.y = ee_target_quat[1]
        pose_msg.pose.orientation.z = ee_target_quat[2]
        pose_msg.pose.orientation.w = ee_target_quat[3]

        return pose_msg

    def update_target_pose(self) -> None:
        """Timer callback to update target pose."""
        # Initialize object tracking if not done yet
        if not self.initialization_attempted:
            # Check if all required camera data is available
            static_rgb = self.images.get("static_top_rgb")
            static_depth = self.images.get("static_top_depth")
            static_k = self.camera_k_matrices.get("static_top_rgb")

            if (
                static_rgb is not None
                and static_depth is not None
                and static_k is not None
            ):
                self.initialization_attempted = True
                self.get_logger().info(
                    "📸 Camera data ready! Starting object tracking initialization..."
                )

                success = self.initialize_object_tracking()
                if success:
                    self.object_tracking_ready = True
                    self.get_logger().info("✅ Object tracking ready")
                else:
                    self.get_logger().error("❌ Object tracking initialization failed")
                    return
            else:
                self.get_logger().info("Waiting for camera data...")
                return

        # Skip if object tracking is not ready
        if not self.object_tracking_ready:
            return

        # Trigger pose tracking update
        self.trigger_pose_tracking()

        # Get current pick target pose
        with self.pose_tracking_lock:
            current_pose = self.current_pick_target_pose.copy()

        # Check if pose is valid (not all zeros)
        if any(abs(x) > 1e-6 for x in current_pose[:3]):
            # Compute target end-effector pose
            target_ee_pose = self.compute_target_ee_pose(current_pose)

            # Publish target pose
            self.publisher_.publish(target_ee_pose)

            # Log target pose info
            pos = target_ee_pose.pose.position
            ori = target_ee_pose.pose.orientation

            self.get_logger().info(
                f"Published target EE pose: "
                f"pos=[{pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}], "
                f"ori=[{ori.x:.3f}, {ori.y:.3f}, {ori.z:.3f}, {ori.w:.3f}]"
            )
        else:
            self.get_logger().warn("Pick target pose not available yet")

    def shutdown(self) -> None:
        """Clean up resources when shutting down."""
        # Stop pose tracking thread
        self.pose_tracking_shutdown.set()
        self.pose_tracking_trigger.set()  # Unblock the thread
        if self.pose_tracking_thread and self.pose_tracking_thread.is_alive():
            self.pose_tracking_thread.join(timeout=1.0)

        # Exit WebSocket contexts
        self._exit_websocket_contexts()

        if rclpy.ok():
            self.get_logger().info("PickTargetPoseController shutdown complete")


def main(args=None):
    rclpy.init(args=args)
    node = PickTargetPoseController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
