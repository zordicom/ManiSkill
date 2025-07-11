#!/usr/bin/env python3
"""
Copyright 2025 Zordi, Inc. All rights reserved.
"""

import argparse
import json
import os
import shutil
import threading
import time
from datetime import datetime
from enum import Enum, unique
from pathlib import Path
from queue import Queue
from typing import Callable, Dict, Optional, Tuple

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from galaxea_orchestrator.common.calibration_utils import (
    CalibrationData,
    load_calibration,
)
from galaxea_orchestrator.utils import (
    adjust_k_for_crop_and_resize,
    center_crop_and_resize_image,
    colorize_depth_image,
    custom_json_dumps,
)
from galaxea_orchestrator.ws_clients.ws_det_client import ObjectDetectionClient
from galaxea_orchestrator.ws_clients.ws_pose_multi_client import (
    MultiObjectFoundationPoseClient,
)
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image, JointState
from signal_arm_msgs.msg import GripperPositionControl
from std_msgs.msg import Header, String

ROBOT_ID = os.environ["ZORDI_ROBOT_ID"]

DEFAULT_EPISODE_DIR = os.getenv(
    "ZORDI_BIMANUAL_EPISODES_DIR",
    str(Path("~/galaxea_human_demos/box_pnp/").expanduser()),
)

DEFAULT_VELOCITY = 1.0
DEFAULT_EFFORT = 1.0
SAFE_VELOCITY = 0.35
SAFE_EFFORT = 0.5
GRIPPER_POSITION_SCALE = 55
ARM_CONN_TIMEOUT_SEC = 1.5
DEAD_ZONE_THRESH_DEG = 0.0

# Right arm offset from origin (in meters)
RIGHT_ARM_OFFSET = [-0.025, -0.365, 0.005]

# J6 to tool tip offset (in meters, relative to J6 frame)
J6_TO_TOOL_TIP_OFFSET = [0.0, 0.0, 0.078]  # [x, y, z] offset from J6 to tool tip

QUEUE_SIZE = 10
RECORDING_HZ = 50

IMAGE_SIZE = 480
JPEG_QUALITY = 90
RESIZE_IMAGE = True

# Depth compression settings
# Intel RealSense depth compression using colorization technique
# Based on: https://dev.intelrealsense.com/docs/depth-image-compression-by-colorization-for-intel-realsense-depth-cameras
USE_DEPTH_COLORIZATION = True
DEPTH_MIN_RANGE = 0.05  # Minimum depth in meters (adjust based on your scene)
DEPTH_MAX_RANGE = 2.0  # Maximum depth in meters (adjust based on your scene)

# Visualization settings
ENABLE_POSE_VISUALIZATION = True  # Set to False for maximum performance

# Camera viewer settings
ENABLE_CAMERA_VIEWER = True  # Enable/disable camera stream viewer
CAMERA_VIEWER_FPS = 10.0  # FPS for camera viewer display
CAMERA_VIEWER_WIDTH = 1440  # Width of the camera viewer window
CAMERA_VIEWER_HEIGHT = 480  # Height of the camera viewer window

# Align the J6 before turning on the robot. If it doesn't work, use these offsets.
LEADER_L_J6_OFF_DEG = 0.0
LEADER_R_J6_OFF_DEG = 0.0

# Object names for detection and pose tracking
# Detection uses descriptive names, pose tracking uses model names
OBJECT_NAME_MAPPING = {
    "pick_target": {
        "detection_name": "small white box",  # Name used for object detection
        "pose_name": "b5box",  # Name used for pose estimation model
    },
    "place_target": {
        "detection_name": "gray basket",  # Name used for object detection
        "pose_name": "basket",  # Name used for pose estimation model
    },
}

# WebSocket server configuration
DETECTION_SERVER_HOST = os.getenv("DETECTION_SERVER_HOST", "yk-dev-4090")
DETECTION_SERVER_PORT = int(os.getenv("DETECTION_SERVER_PORT", "10015"))
POSE_SERVER_HOST = os.getenv("POSE_SERVER_HOST", "yk-dev-4090")
POSE_SERVER_PORT = int(os.getenv("POSE_SERVER_PORT", "10014"))

# Pose tracking configuration
POSE_TRACKING_HZ = 50
POSE_CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence to accept pose predictions

# Custom rotations to apply to estimated poses (in object coordinate system)
# Values are (x, y, z) euler angles in degrees
CUSTOM_ROTATIONS = {
    "pick_target": (0, 90, 0),  # No rotation by default
    "place_target": (0, 0, 0),  # No rotation by default
}

# Custom position offsets to apply to estimated poses (in object coordinate system)
# Values are (x, y, z) offsets in meters
CUSTOM_POSITION_OFFSETS = {
    "pick_target": (0, 0, 0.065),  # Z-axis offset for pick target
    "place_target": (0, 0, 0),  # No offset by default
}

# Camera calibration constants
DEFAULT_CALIBRATION_FILE = "camera_calibration.json"


def get_calibration_file_path(custom_path: Optional[str] = None) -> str:
    """Get the path to the camera calibration file.

    Args:
        custom_path: Custom calibration file path if provided

    Returns:
        Path to the calibration file
    """
    # If custom path is provided, use it as-is
    if custom_path and custom_path != DEFAULT_CALIBRATION_FILE:
        return custom_path

    # Try to find the calibration file in the ROS2 package share directory
    try:
        package_share_dir = get_package_share_directory("galaxea_orchestrator")
        robot_id = os.environ.get("ZORDI_ROBOT_ID")

        if robot_id:
            # Construct path to robot-specific calibration file
            calibration_path = (
                Path(package_share_dir)
                / "configs"
                / robot_id
                / "camera_calibration.json"
            )

            if calibration_path.exists():
                return str(calibration_path)
            else:
                print(
                    f"Warning: Robot-specific calibration file not found at {calibration_path}"
                )
        else:
            print("Warning: ZORDI_ROBOT_ID environment variable not set")

    except Exception as e:
        print(
            f"Warning: Failed to locate calibration file in package share directory: {e}"
        )

    # Fall back to default behavior (look in current directory)
    return DEFAULT_CALIBRATION_FILE


@unique
class ArmType(str, Enum):
    """Enumeration of robot arm types."""

    LEFT = "left"
    RIGHT = "right"
    LEADER = "leader"
    FOLLOWER = "follower"
    LEADER_LEFT = "leader_left"
    LEADER_RIGHT = "leader_right"
    FOLLOWER_LEFT = "follower_left"
    FOLLOWER_RIGHT = "follower_right"


@unique
class EStopType(str, Enum):
    """Enumeration of e-stop types."""

    SOFT = "soft"
    HARD = "hard"


@unique
class PositionType(str, Enum):
    """Enumeration of arm position types."""

    HOME = "home"
    ZERO = "zero"


@unique
class TeleopState(str, Enum):
    """Enumeration of possible teleoperation states."""

    E_STOPPED = "e_stopped"
    IDLE = "idle"
    HUMAN_OPERATOR = "human_operator"
    RECORDING = "recording"
    AWAITING_FEEDBACK = "awaiting_feedback"
    PROGRAM = "program"


POSITION_PRESETS_IN_DEGS = {
    PositionType.HOME: {
        # ArmType.LEFT: [-42.0, 45.0, -48.0, 100.0, -35.0, -15.0, 70],  # bimanual
        ArmType.LEFT: [-41.4, 0.11, -51.7, 88.9, -65.4, -2.1, 90],  # static
        ArmType.RIGHT: [42.0, 45.0, -48.0, 80.0, -35.0, 15.0, 90],
    },
    PositionType.ZERO: {
        ArmType.LEFT: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 45.0],
        ArmType.RIGHT: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 45.0],
    },
}


class TeleopBridge(Node):
    """Teleoperation bridge for bimanual control."""

    def __init__(
        self,
        episode_dir: str = DEFAULT_EPISODE_DIR,
        verbose: bool = False,
        calibration_file: str = DEFAULT_CALIBRATION_FILE,
    ):
        super().__init__("leader_to_follower_bridge")

        self.episode_dirpath = Path(episode_dir)
        self.episode_dirpath.mkdir(parents=True, exist_ok=True)
        self.current_episode_dirpath = None
        self.verbose = verbose
        self.calibration_file = calibration_file

        # Camera calibration data
        self.calibration_data: CalibrationData | None = None
        self._load_calibration()

        # Camera topics for recording
        self.camera_topics: dict[str, str] = {
            "static_top_rgb": "/camera/static_rs405_top/color/image_rect_raw",
            "static_top_depth": "/camera/static_rs405_top/depth/image_rect_raw",
            "eoat_left_top_rgb": "/camera/eoat_rs405_left_top/color/image_rect_raw",
            "eoat_right_top_rgb": "/camera/eoat_rs405_right_top/color/image_rect_raw",
        }

        # Camera info topics for RGB cameras only (derived from image topics)
        self.camera_info_topics: dict[str, str] = {}
        self.camera_k_matrices: dict[str, list[float]] = {}
        self.camera_original_k_matrices: dict[str, list[float]] = {}
        self.camera_image_dimensions: dict[str, tuple[int, int]] = {}
        for name, topic in self.camera_topics.items():
            if "rgb" in name:  # Only RGB cameras, not depth
                # Replace image_rect_raw with camera_info
                info_topic = topic.replace("image_rect_raw", "camera_info")
                self.camera_info_topics[name] = info_topic

        # Initialize camera image storage
        self.images: dict[str, np.ndarray | None] = {}
        self.camera_subscribers = []
        self.camera_info_subscribers = []

        # Initialize camera viewer components
        self.camera_viewer_enabled = ENABLE_CAMERA_VIEWER
        self.camera_viewer_window_name = "teleop_camera_viewer"
        self.camera_viewer_images: dict[str, np.ndarray | None] = {}
        self.camera_viewer_layout: dict[tuple[int, int], str] = {}
        self.camera_viewer_grid_size: tuple[int, int] = (3, 1)  # 3 columns, 1 row
        self.camera_viewer_cell_width = (
            CAMERA_VIEWER_WIDTH // self.camera_viewer_grid_size[0]
        )
        self.camera_viewer_cell_height = (
            CAMERA_VIEWER_HEIGHT // self.camera_viewer_grid_size[1]
        )

        # Initialize camera viewer if enabled
        if self.camera_viewer_enabled:
            self._init_camera_viewer()

        # WebSocket clients for object detection and pose estimation
        self.detection_client: Optional[ObjectDetectionClient] = None
        self.pose_client: Optional[MultiObjectFoundationPoseClient] = None

        # WebSocket context managers for episode-level connections
        self.pose_client_context = None
        self.detection_client_context = None

        # Pose tracking state - DUAL COORDINATE SYSTEM:
        #
        # We maintain poses in two coordinate systems to handle different use cases:
        # 1. World coordinates (target_poses): Used for behavior cloning/RL models
        #    - Poses are transformed from camera to world coordinates using calibration
        #    - Recorded in observations for training data
        # 2. Camera coordinates (target_poses_camera): Used for visualization
        #    - Poses remain in camera coordinate system
        #    - Used for 3D visualization that projects using camera intrinsics

        # World coordinates for behavior cloning/RL models
        self.target_poses: dict[str, list[float]] = {
            "pick_target_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "place_target_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        }
        # Camera coordinates for visualization
        self.target_poses_camera: dict[str, list[float]] = {
            "pick_target_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "place_target_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        }
        self.pose_tracking_initialized: dict[str, bool] = {
            "pick_target": False,
            "place_target": False,
        }
        self.pose_tracking_lock = threading.Lock()  # Thread safety for pose updates

        # Pose tracking thread management
        self.pose_tracking_thread: threading.Thread | None = None
        self.pose_tracking_in_progress = (
            threading.Event()
        )  # Flag for prediction in progress
        self.pose_tracking_trigger = threading.Event()  # Trigger for new prediction
        self.pose_tracking_shutdown = threading.Event()  # Shutdown signal
        self.pose_tracking_skip_counter = 0  # Counter for skipped frames

        # Object tracking initialization thread management
        self.initialization_thread: threading.Thread | None = None
        self.initialization_complete = (
            threading.Event()
        )  # Flag for initialization completion
        self.initialization_success = (
            threading.Event()
        )  # Flag for initialization success
        self.initialization_shutdown = threading.Event()  # Shutdown signal
        self.initialization_in_progress = (
            False  # Flag to prevent multiple initializations
        )

        # Initialize recording state
        self.recording_enabled: bool = False
        self.recording_pending: bool = (
            False  # Flag for recording waiting for initialization
        )
        self.frame_index: int = 0
        self.observations: list[dict] = []
        self.episode_start_time: float | None = None
        self.last_frame_time: float | None = None

        # Observation monitoring for early warning system
        self.last_successful_observation_time: float | None = None
        self.last_observation_check_time: float | None = None
        self.observation_warning_interval: float = 5.0  # Warn every 5 seconds
        self.missing_data_counters: dict[str, int] = {
            "joint_states": 0,
            "ee_poses": 0,
            "images": 0,
            "queue_full": 0,
        }

        # Threading for file I/O operations
        self.recording_queue: Queue = Queue(maxsize=100)  # Prevent memory overflow
        self.recording_thread: threading.Thread | None = None
        self.recording_thread_shutdown: threading.Event = threading.Event()

        # End-effector pose storage
        self.last_ee_pose_set: dict[ArmType, any] = {
            ArmType.FOLLOWER_LEFT: None,
            ArmType.FOLLOWER_RIGHT: None,
        }

        # CV bridge for image conversion
        self.bridge = CvBridge()

        # Joint names and parameters
        self.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        self.frame_id = "world"

        self.last_msg_set: Dict[ArmType, JointState] = {
            ArmType.FOLLOWER_LEFT: JointState(),
            ArmType.FOLLOWER_RIGHT: JointState(),
            ArmType.LEADER_LEFT: JointState(),
            ArmType.LEADER_RIGHT: JointState(),
        }

        # Convert degree values to radians for all position presets
        self.positions = {}
        for position_type, arm_positions in POSITION_PRESETS_IN_DEGS.items():
            self.positions[position_type] = {}
            for arm_type, position in arm_positions.items():
                position_with_offset = position.copy()

                # Apply the J6 offset (6th joint, index 5)
                if arm_type == ArmType.LEFT:
                    position_with_offset[5] += LEADER_L_J6_OFF_DEG
                elif arm_type == ArmType.RIGHT:
                    position_with_offset[5] += LEADER_R_J6_OFF_DEG

                # Convert to radians
                self.positions[position_type][arm_type] = np.deg2rad(
                    position_with_offset
                ).tolist()

        self.teleop_state = TeleopState.IDLE
        self.is_left_arm_held = False

        self.e_stop_states = {
            ArmType.LEADER: {
                EStopType.SOFT: False,
                EStopType.HARD: False,
            },
            ArmType.FOLLOWER: {
                EStopType.SOFT: False,
                EStopType.HARD: False,
            },
        }

        self.init_arm_topics()
        self.init_camera_topics()
        self.init_io_topics()
        self.init_teleop_state_topic()

        # Recording timer
        self.recording_timer = self.create_timer(1.0 / RECORDING_HZ, self.record_step)

        # Arm connection checker
        self.arm_conn_checker = self.create_timer(1.0, self.check_arm_connection)

        # Pose tracking timer
        self.pose_tracking_timer = self.create_timer(
            1.0 / POSE_TRACKING_HZ, self.track_poses
        )

        # Initialization checker timer (checks if initialization is complete)
        self.initialization_checker = self.create_timer(
            0.1, self.check_initialization_status
        )

        # Camera viewer timer
        if self.camera_viewer_enabled:
            self.camera_viewer_timer = self.create_timer(
                1.0 / CAMERA_VIEWER_FPS, self.update_camera_viewer_display
            )

        # Observation monitoring timer (check every 2 seconds)
        self.observation_monitor_timer = self.create_timer(
            2.0, self.monitor_observation_recording
        )

        # Initialize WebSocket clients
        self._init_websocket_clients()

        # End-effector pose subscribers for follower arms
        self.subscription_ee_pose_follower_l = self.create_subscription(
            PoseStamped,
            "/ee_pose_follower_l",
            self.create_ee_pose_callback(ArmType.FOLLOWER_LEFT),
            QUEUE_SIZE,
        )
        self.subscription_ee_pose_follower_r = self.create_subscription(
            PoseStamped,
            "/ee_pose_follower_r",
            self.create_ee_pose_callback(ArmType.FOLLOWER_RIGHT),
            QUEUE_SIZE,
        )

        self.get_logger().info("Bimanual Teleop Bridge running...")

    def _recording_worker_thread(self) -> None:
        """Worker thread that handles file I/O operations for recording."""
        while not self.recording_thread_shutdown.is_set():
            try:
                # Wait for recording data with timeout to check shutdown periodically
                recording_data = self.recording_queue.get(timeout=0.1)
                if recording_data is None:  # Sentinel value for shutdown
                    break

                self._save_recording_data(recording_data)
                self.recording_queue.task_done()

            except Exception as e:
                # Handle queue timeout (expected) and other exceptions
                if "Empty" in str(type(e).__name__):
                    continue  # Queue timeout, check shutdown flag and continue
                elif not self.recording_thread_shutdown.is_set():
                    self.get_logger().error(f"Error in recording worker thread: {e}")

    def _save_recording_data(self, recording_data: dict) -> None:
        """Save recording data to disk (runs in worker thread)."""
        frame_index = recording_data["frame_index"]
        images = recording_data["images"]
        observation = recording_data["observation"]
        raw_data_dir = recording_data["raw_data_dir"]
        camera_intrinsics = recording_data["camera_intrinsics"]

        # Store camera images
        image_paths: dict[str, str] = {}
        for cam_name, img in images.items():
            if img is None:
                continue
            cam_dir = raw_data_dir / cam_name
            cam_dir.mkdir(exist_ok=True)

            # Handle depth images with optional colorization compression
            if "depth" in cam_name:
                if USE_DEPTH_COLORIZATION:
                    # Use Intel's colorization technique for better compression
                    colorized_depth = colorize_depth_image(
                        img, DEPTH_MIN_RANGE, DEPTH_MAX_RANGE
                    )
                    fname = f"{frame_index:06d}.jpg"
                    fpath = cam_dir / fname
                    # Save colorized depth as JPEG
                    cv2.imwrite(
                        str(fpath),
                        colorized_depth,
                        [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
                    )

                    # Also save depth range metadata for reconstruction
                    if frame_index == 0:  # Save metadata once per recording
                        metadata = {
                            "min_depth": DEPTH_MIN_RANGE,
                            "max_depth": DEPTH_MAX_RANGE,
                            "colorized": True,
                        }
                        metadata_path = cam_dir / "depth_metadata.json"
                        with open(metadata_path, "w", encoding="utf-8") as f:
                            json.dump(metadata, f, indent=2)
                else:
                    # Traditional PNG compression
                    fname = f"{frame_index:06d}.png"
                    fpath = cam_dir / fname
                    # Save as 16-bit PNG with compression for depth data
                    cv2.imwrite(str(fpath), img, [int(cv2.IMWRITE_PNG_COMPRESSION), 6])
            else:
                fname = f"{frame_index:06d}.jpg"
                fpath = cam_dir / fname
                # Convert RGB to BGR for cv2.imwrite, then save as JPEG
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(
                    str(fpath), img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
                )

            image_paths[cam_name] = f"{cam_name}/{fname}"

        # Update observation with image paths
        observation.update(image_paths)

        # Generate and save visualization for debugging
        if ENABLE_POSE_VISUALIZATION:
            # Create visualizations directory
            vis_dir = raw_data_dir / "visualizations"
            vis_dir.mkdir(exist_ok=True)

            # Get static top RGB image for visualization
            static_rgb = images.get("static_top_rgb")
            if static_rgb is not None:
                # Generate visualization if we have poses
                if camera_intrinsics is not None:
                    # Convert pose data to matrices and create confidences
                    # Use camera coordinates for visualization
                    # (from target_poses_camera)
                    poses = {}
                    confidences = {}

                    # Extract tracked object poses from camera coordinates
                    # (we need to get these from the current state, not the observation)
                    with self.pose_tracking_lock:
                        for target_key in ["pick_target", "place_target"]:
                            pose_key = f"{target_key}_pose"
                            if pose_key in self.target_poses_camera:
                                pose_7d = self.target_poses_camera[pose_key]

                                # Convert 7D pose to 4x4 matrix
                                if len(pose_7d) == 7 and any(
                                    abs(x) > 1e-6 for x in pose_7d[:3]
                                ):
                                    # Only include non-zero poses
                                    translation = np.array(pose_7d[:3])
                                    quaternion = np.array(
                                        pose_7d[3:7]
                                    )  # [qx, qy, qz, qw]

                                    rotation_matrix = Rotation.from_quat(
                                        quaternion
                                    ).as_matrix()

                                    pose_matrix = np.eye(4)
                                    pose_matrix[:3, :3] = rotation_matrix
                                    pose_matrix[:3, 3] = translation

                                    poses[target_key] = pose_matrix
                                    # Use a default confidence since we don't track it
                                    # in observations
                                    confidences[target_key] = 0.8

                    # Create visualization if we have poses
                    if poses:
                        vis_image_rgb = self.create_integrated_visualization(
                            static_rgb, poses, confidences, camera_intrinsics
                        )
                        # Convert RGB visualization to BGR for saving with cv2.imwrite
                        vis_image_bgr = cv2.cvtColor(vis_image_rgb, cv2.COLOR_RGB2BGR)

                        # Save visualization with 80% JPEG quality
                        vis_fname = f"{frame_index:06d}.jpg"
                        vis_fpath = vis_dir / vis_fname
                        cv2.imwrite(
                            str(vis_fpath),
                            vis_image_bgr,  # BGR format for cv2.imwrite
                            [int(cv2.IMWRITE_JPEG_QUALITY), 80],
                        )

        # Add to observations list (this is thread-safe since we're only appending)
        self.observations.append(observation)

    def init_arm_topics(self) -> None:
        """Initialize the ROS topics."""
        # Left arm setup
        self.subscription_leader_l = self.create_subscription(
            JointState,
            "/joint_states_leader_l",
            self.create_joint_state_callback(ArmType.LEADER_LEFT),
            QUEUE_SIZE,
        )
        self.subscription_follower_l = self.create_subscription(
            JointState,
            "/joint_states_follower_l",
            self.create_joint_state_callback(ArmType.FOLLOWER_LEFT),
            QUEUE_SIZE,
        )
        self.publisher_follower_l = self.create_publisher(
            JointState, "/arm_joint_target_position_follower_l", QUEUE_SIZE
        )
        self.publisher_leader_l = self.create_publisher(
            JointState, "/arm_joint_target_position_leader_l", QUEUE_SIZE
        )
        self.publisher_follower_g1_gripper_l = self.create_publisher(
            GripperPositionControl, "/gripper_position_control_follower_l", QUEUE_SIZE
        )

        # Right arm setup
        self.subscription_leader_r = self.create_subscription(
            JointState,
            "/joint_states_leader_r",
            self.create_joint_state_callback(ArmType.LEADER_RIGHT),
            QUEUE_SIZE,
        )
        self.subscription_follower_r = self.create_subscription(
            JointState,
            "/joint_states_follower_r",
            self.create_joint_state_callback(ArmType.FOLLOWER_RIGHT),
            QUEUE_SIZE,
        )
        self.publisher_follower_r = self.create_publisher(
            JointState, "/arm_joint_target_position_follower_r", QUEUE_SIZE
        )
        self.publisher_leader_r = self.create_publisher(
            JointState, "/arm_joint_target_position_leader_r", QUEUE_SIZE
        )
        self.publisher_follower_g1_gripper_r = self.create_publisher(
            GripperPositionControl, "/gripper_position_control_follower_r", QUEUE_SIZE
        )

    def init_camera_topics(self) -> None:
        """Initialize camera subscribers for recording."""
        # Create camera subscribers
        for name, topic in self.camera_topics.items():
            self.images[name] = None
            self.camera_subscribers.append(
                self.create_subscription(
                    Image, topic, self._create_camera_callback(name), QUEUE_SIZE
                )
            )
            self.get_logger().info(f"Subscribed to camera topic: {topic} as {name}")

        # Create camera info subscribers for RGB cameras
        for name, topic in self.camera_info_topics.items():
            self.camera_info_subscribers.append(
                self.create_subscription(
                    CameraInfo,
                    topic,
                    self._create_camera_info_callback(name),
                    QUEUE_SIZE,
                )
            )
            self.get_logger().info(
                f"Subscribed to camera info topic: {topic} as {name}"
            )

    def _create_camera_callback(self, camera_name: str):
        """Create a callback function for a specific camera."""

        def callback(msg: Image) -> None:
            self.image_callback(msg, camera_name)

        return callback

    def _create_camera_info_callback(self, camera_name: str):
        """Create a callback function for a specific camera info."""

        def callback(msg: CameraInfo) -> None:
            self.camera_info_callback(msg, camera_name)

        return callback

    def image_callback(self, msg: Image, camera_name: str) -> None:
        """Process camera images.

        Note: ROS images come in BGR format, but we convert to RGB immediately
        and store RGB images in self.images for consistency with server APIs.
        BGR conversion only happens when saving with cv2.imwrite().
        """
        try:
            # Convert ROS Image message to numpy array
            if "depth" in camera_name:
                # For depth images, use passthrough to avoid cv_bridge format conversion
                # issues. This preserves the original 16UC1 data without conversion.
                cv_image = self.bridge.imgmsg_to_cv2(
                    msg, desired_encoding="passthrough"
                )
                cv_image = cv_image.astype(np.uint16)

                self.get_logger().debug(
                    f"Depth image {camera_name}: {msg.encoding} -> passthrough, "
                    f"shape: {cv_image.shape}, dtype: {cv_image.dtype}"
                )
            else:
                # RGB images: convert from ROS BGR format to RGB for storage
                cv_image_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                cv_image = cv2.cvtColor(cv_image_bgr, cv2.COLOR_BGR2RGB)

                self.get_logger().debug(
                    f"RGB image {camera_name}: {msg.encoding} -> bgr8 -> rgb, "
                    f"shape: {cv_image.shape}, dtype: {cv_image.dtype}"
                )

            # Resize to standard size if enabled
            if RESIZE_IMAGE:
                cv_image = center_crop_and_resize_image(
                    cv_image, image_size=(IMAGE_SIZE, IMAGE_SIZE)
                )

            self.images[camera_name] = cv_image
        except Exception as e:
            self.get_logger().error(
                f"Error processing camera image {camera_name} "
                f"(encoding: {msg.encoding}): {e}"
            )

    def camera_info_callback(self, msg: CameraInfo, camera_name: str) -> None:
        """Process camera info messages and extract K matrix and image dimensions."""
        try:
            # Extract K matrix (3x3 intrinsic camera matrix stored as 9-element array in
            # row-major order)
            if len(msg.k) == 9:  # noqa: PLR2004
                # Store original K matrix and image dimensions
                self.camera_original_k_matrices[camera_name] = list(msg.k)
                self.camera_image_dimensions[camera_name] = (
                    int(msg.width),
                    int(msg.height),
                )

                # Compute adjusted K matrix if RESIZE_IMAGE is enabled
                if RESIZE_IMAGE:
                    # Convert K matrix from list to numpy array for processing
                    k_matrix = np.array(msg.k).reshape(3, 3)
                    orig_size = (int(msg.width), int(msg.height))
                    final_size = (IMAGE_SIZE, IMAGE_SIZE)

                    # Apply K matrix adjustment for center crop and resize
                    adjusted_k = adjust_k_for_crop_and_resize(
                        k_matrix, orig_size, final_size, pad_ratio=0.0
                    )

                    # Store the adjusted K matrix (flattened back to list)
                    self.camera_k_matrices[camera_name] = adjusted_k.flatten().tolist()

                    self.get_logger().debug(
                        f"Updated adjusted K matrix for {camera_name} "
                        f"({orig_size} -> {final_size}): "
                        f"fx={adjusted_k[0, 0]:.2f}, fy={adjusted_k[1, 1]:.2f}, "
                        f"cx={adjusted_k[0, 2]:.2f}, cy={adjusted_k[1, 2]:.2f}"
                    )
                else:
                    # Use original K matrix if no image processing
                    self.camera_k_matrices[camera_name] = list(msg.k)

                    self.get_logger().debug(
                        f"Updated original K matrix for {camera_name}: "
                        f"fx={msg.k[0]:.2f}, fy={msg.k[4]:.2f}, "
                        f"cx={msg.k[2]:.2f}, cy={msg.k[5]:.2f}"
                    )
            else:
                self.get_logger().warning(
                    f"Invalid K matrix size for {camera_name}: expected 9, "
                    f"got {len(msg.k)}"
                )
        except Exception as e:
            self.get_logger().error(f"Error processing camera info {camera_name}: {e}")

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

    def monitor_observation_recording(self) -> None:
        """Monitor observation recording health and warn if no observations are being collected."""
        if not self.recording_enabled:
            return

        current_time = time.time()

        # Initialize timing if this is the first check
        if self.last_observation_check_time is None:
            self.last_observation_check_time = current_time
            return

        # Check if we should issue a warning (every 5 seconds)
        time_since_last_check = current_time - self.last_observation_check_time
        if time_since_last_check < self.observation_warning_interval:
            return

        # Check if any observations have been recorded recently
        if self.last_successful_observation_time is None:
            # No observations recorded yet since recording started
            time_since_recording_started = current_time - (
                self.last_observation_check_time - time_since_last_check
            )
            if time_since_recording_started >= self.observation_warning_interval:
                self.get_logger().warning(
                    f"🚨 RECORDING ISSUE: No observations recorded in {time_since_recording_started:.1f}s since recording started!"
                )
                self._diagnose_recording_issues()
        else:
            # Check if observations have stopped being recorded
            time_since_last_obs = current_time - self.last_successful_observation_time
            if time_since_last_obs >= self.observation_warning_interval:
                self.get_logger().warning(
                    f"🚨 RECORDING ISSUE: No observations recorded in {time_since_last_obs:.1f}s! "
                    f"(Last observation at frame {self.frame_index - 1})"
                )
                self._diagnose_recording_issues()

        self.last_observation_check_time = current_time

    def _diagnose_recording_issues(self) -> None:
        """Diagnose and report specific issues preventing observation recording."""
        issues = []

        # Check recording state
        if not self.recording_enabled:
            issues.append("❌ Recording not enabled")

        # Check joint states
        left_joint_state = self.last_msg_set[ArmType.FOLLOWER_LEFT]
        right_joint_state = self.last_msg_set[ArmType.FOLLOWER_RIGHT]

        if not left_joint_state.position or len(left_joint_state.position) < 7:
            issues.append("❌ Left follower joint states missing or incomplete")
        if not right_joint_state.position or len(right_joint_state.position) < 7:
            issues.append("❌ Right follower joint states missing or incomplete")

        # Check end-effector poses
        left_ee_pose = self.last_ee_pose_set[ArmType.FOLLOWER_LEFT]
        right_ee_pose = self.last_ee_pose_set[ArmType.FOLLOWER_RIGHT]

        if left_ee_pose is None:
            issues.append("❌ Left follower end-effector pose missing")
        if right_ee_pose is None:
            issues.append("❌ Right follower end-effector pose missing")

        # Check images
        missing_images = [name for name, img in self.images.items() if img is None]
        if missing_images:
            issues.append(f"❌ Missing camera images: {missing_images}")

        # Check recording queue status
        if hasattr(self, "recording_queue"):
            queue_size = self.recording_queue.qsize()
            if queue_size >= 90:  # Near the 100 limit
                issues.append(f"⚠️ Recording queue nearly full: {queue_size}/100")

        # Check if recording thread is alive
        if self.recording_thread is None or not self.recording_thread.is_alive():
            issues.append("❌ Recording worker thread not running")

        # Report findings
        if issues:
            self.get_logger().warning("📋 RECORDING DIAGNOSIS:")
            for issue in issues:
                self.get_logger().warning(f"   {issue}")

            # Provide specific guidance
            if any("joint states" in issue for issue in issues):
                self.get_logger().warning(
                    "💡 Check if follower arms are connected and publishing joint states"
                )
            if any("end-effector pose" in issue for issue in issues):
                self.get_logger().warning(
                    "💡 Check if end-effector pose publishers are active"
                )
            if any("camera images" in issue for issue in issues):
                self.get_logger().warning(
                    "💡 Check if camera nodes are running and publishing images"
                )
        else:
            self.get_logger().warning(
                "🤔 No obvious issues detected - recording may be working but very slow"
            )

        # Show current data counter status
        if any(count > 0 for count in self.missing_data_counters.values()):
            self.get_logger().warning(
                f"📊 Missing data counters: {dict(self.missing_data_counters)}"
            )

    def _enter_websocket_contexts(self) -> bool:
        """Enter WebSocket context managers for episode-level connections.

        Returns:
            True if both connections successful, False otherwise
        """
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
            # Clean up any partial connections
            self._exit_websocket_contexts()
            return False

    def _exit_websocket_contexts(self) -> None:
        """Exit WebSocket context managers."""
        # Wait briefly for any in-flight requests to complete
        if self.pose_tracking_in_progress.is_set():
            self.get_logger().info("Waiting for in-flight pose tracking to complete...")
            self.pose_tracking_in_progress.wait(timeout=0.5)

        # Exit pose client context
        if self.pose_client_context is not None:
            try:
                self.pose_client.__exit__(None, None, None)
                self.get_logger().info("Pose client context exited successfully")
            except Exception as e:
                self.get_logger().warning(f"Error exiting pose client context: {e}")
            finally:
                self.pose_client_context = None

        # Exit detection client context
        if self.detection_client_context is not None:
            try:
                self.detection_client.__exit__(None, None, None)
                self.get_logger().info("Detection client context exited successfully")
            except Exception as e:
                self.get_logger().warning(
                    f"Error exiting detection client context: {e}"
                )
            finally:
                self.detection_client_context = None

    def _init_camera_viewer(self) -> None:
        """Initialize camera viewer components."""
        try:
            # Define camera layout for viewer (1x3 grid)
            self.camera_viewer_layout = {
                (0, 0): "static_top_rgb",  # Left: Static top RGB
                (0, 1): "eoat_left_top_rgb",  # Center: Left EOAT RGB
                (0, 2): "eoat_right_top_rgb",  # Right: Right EOAT RGB
            }

            # Initialize camera viewer image storage
            for camera_name in self.camera_viewer_layout.values():
                self.camera_viewer_images[camera_name] = None

            # Set up OpenCV window
            cv2.namedWindow(self.camera_viewer_window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(
                self.camera_viewer_window_name,
                CAMERA_VIEWER_WIDTH,
                CAMERA_VIEWER_HEIGHT,
            )

            self.get_logger().info("Camera viewer initialized successfully")

        except Exception as e:
            self.get_logger().error(f"Failed to initialize camera viewer: {e}")
            self.camera_viewer_enabled = False

    def _resize_image_with_padding(self, image: np.ndarray | None) -> np.ndarray:
        """Resize image to fit within a cell while maintaining aspect ratio, with black padding.

        Args:
            image: Input image in OpenCV format

        Returns:
            Resized image with padding
        """
        if image is None:
            # Return a black image if no image is available
            return np.zeros(
                (self.camera_viewer_cell_height, self.camera_viewer_cell_width, 3),
                dtype=np.uint8,
            )

        h, w = image.shape[:2]
        target_w, target_h = (
            self.camera_viewer_cell_width,
            self.camera_viewer_cell_height,
        )

        # Calculate aspect ratios
        aspect_ratio = w / h
        target_aspect_ratio = target_w / target_h

        # Determine new dimensions while maintaining aspect ratio
        if aspect_ratio > target_aspect_ratio:
            # Image is wider than the cell aspect ratio
            new_w = target_w
            new_h = int(new_w / aspect_ratio)
        else:
            # Image is taller than the cell aspect ratio
            new_h = target_h
            new_w = int(new_h * aspect_ratio)

        # Resize image to the new dimensions
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Create a black background
        result = np.zeros((target_h, target_w, 3), dtype=np.uint8)

        # Calculate position to place the resized image centered in the cell
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2

        # Embed the resized image in the black background
        result[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

        return result

    def _get_display_image(self, camera_name: str) -> np.ndarray | None:
        """Get the appropriate image for display (either original or with pose visualization).

        Args:
            camera_name: Name of the camera

        Returns:
            Image to display, or None if not available
        """
        # Get the original image
        original_image = self.images.get(camera_name)
        if original_image is None:
            return None

        # For static_top_rgb, show pose visualization when tracking is active
        if camera_name == "static_top_rgb":
            # Check if pose tracking is active
            objects_being_tracked = [
                target_key
                for target_key in ["pick_target", "place_target"]
                if self.pose_tracking_initialized.get(target_key, False)
            ]

            if objects_being_tracked and self.recording_enabled:
                # Generate pose visualization
                static_k = self.camera_k_matrices.get("static_top_rgb")
                if static_k is not None:
                    try:
                        camera_intrinsics = np.array(static_k).reshape(3, 3)

                        # Get current poses in camera coordinates for visualization (thread-safe)
                        with self.pose_tracking_lock:
                            poses = {}
                            confidences = {}

                            for target_key in objects_being_tracked:
                                pose_key = f"{target_key}_pose"
                                if pose_key in self.target_poses_camera:
                                    pose_7d = self.target_poses_camera[pose_key]

                                    # Convert 7D pose to 4x4 matrix
                                    if len(pose_7d) == 7 and any(
                                        abs(x) > 1e-6 for x in pose_7d[:3]
                                    ):
                                        translation = np.array(pose_7d[:3])
                                        quaternion = np.array(
                                            pose_7d[3:7]
                                        )  # [qx, qy, qz, qw]

                                        rotation_matrix = Rotation.from_quat(
                                            quaternion
                                        ).as_matrix()

                                        pose_matrix = np.eye(4)
                                        pose_matrix[:3, :3] = rotation_matrix
                                        pose_matrix[:3, 3] = translation

                                        poses[target_key] = pose_matrix
                                        confidences[target_key] = (
                                            0.8  # Default confidence
                                        )

                        # Create visualization if we have poses
                        if poses:
                            vis_image = self.create_integrated_visualization(
                                original_image, poses, confidences, camera_intrinsics
                            )
                            # Convert RGB to BGR for display
                            return cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)

                    except Exception as e:
                        self.get_logger().debug(
                            f"Error creating pose visualization: {e}"
                        )

        # For all RGB images, convert from RGB to BGR for OpenCV display
        return cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)

    def update_camera_viewer_display(self) -> None:
        """Update the camera viewer display with the latest images."""
        if not self.camera_viewer_enabled:
            return

        try:
            # Get current images for display
            display_images = {}
            for camera_name in self.camera_viewer_layout.values():
                display_image = self._get_display_image(camera_name)
                display_images[camera_name] = self._resize_image_with_padding(
                    display_image
                )

            # Create a grid
            rows = []
            for row_idx in range(self.camera_viewer_grid_size[1]):
                cols = []
                for col_idx in range(self.camera_viewer_grid_size[0]):
                    camera_name = self.camera_viewer_layout.get((row_idx, col_idx))
                    if camera_name and camera_name in display_images:
                        cols.append(display_images[camera_name])
                    else:
                        # Empty cell
                        cols.append(
                            np.zeros(
                                (
                                    self.camera_viewer_cell_height,
                                    self.camera_viewer_cell_width,
                                    3,
                                ),
                                dtype=np.uint8,
                            )
                        )
                rows.append(np.hstack(cols))

            grid = np.vstack(rows)

            # Add labels to each cell
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            color = (255, 255, 255)
            thickness = 2

            for (row, col), camera_name in self.camera_viewer_layout.items():
                label_x = col * self.camera_viewer_cell_width + 10
                label_y = row * self.camera_viewer_cell_height + 30

                # Create display label
                label_text = camera_name.replace("_", " ").title()

                # Add pose tracking status for static camera
                if camera_name == "static_top_rgb":
                    objects_being_tracked = [
                        target_key
                        for target_key in ["pick_target", "place_target"]
                        if self.pose_tracking_initialized.get(target_key, False)
                    ]
                    if objects_being_tracked and self.recording_enabled:
                        label_text += " (Tracking Active)"

                cv2.putText(
                    grid,
                    label_text,
                    (label_x, label_y),
                    font,
                    font_scale,
                    color,
                    thickness,
                )

            # Show the combined image
            cv2.imshow(self.camera_viewer_window_name, grid)
            cv2.waitKey(1)  # Required for OpenCV to update the window

        except Exception as e:
            self.get_logger().error(f"Error updating camera viewer display: {e}")

    def track_poses(self) -> None:
        """Trigger pose tracking if not already in progress (runs at 50Hz)."""
        if not self.recording_enabled:
            return

        # Check if any objects are being tracked
        objects_being_tracked = [
            target_key
            for target_key in ["pick_target", "place_target"]
            if self.pose_tracking_initialized.get(target_key, False)
        ]

        if not objects_being_tracked or self.pose_client is None:
            return

        # If a prediction is already in progress, skip this frame
        if self.pose_tracking_in_progress.is_set():
            self.pose_tracking_skip_counter += 1
            return

        # Reset skip counter when we can process
        self.pose_tracking_skip_counter = min(self.pose_tracking_skip_counter, 0)

        # Trigger pose tracking thread
        self.pose_tracking_trigger.set()

    def check_initialization_status(self) -> None:
        """Check if object tracking initialization is complete and start recording if needed."""
        if not self.recording_pending:
            return

        # Check if initialization is complete
        if self.initialization_complete.is_set():
            if self.initialization_success.is_set():
                # Initialization successful, start recording
                self.get_logger().info(
                    "✅ Object tracking initialization successful! Starting recording..."
                )
                self.recording_enabled = True
                self.recording_pending = False
                self.set_teleop_state(TeleopState.RECORDING)
            else:
                # Initialization failed, but we still wait for feedback if episode directory exists
                self.get_logger().error(
                    "❌ Object tracking initialization failed! Episode will await feedback."
                )
                self.recording_pending = False

                # Always transition to awaiting feedback state instead of deleting directory
                # The user can decide whether to mark it as good/bad through feedback
                if self.current_episode_dirpath is not None:
                    self.get_logger().info(
                        "⚠️ Initialization failed but episode directory exists - awaiting user feedback"
                    )
                    self.set_teleop_state(TeleopState.AWAITING_FEEDBACK)
                else:
                    # Shouldn't happen since start_episode creates directory, but handle gracefully
                    self.get_logger().warning(
                        "Initialization failed and no episode directory found - returning to idle"
                    )
                    self.set_teleop_state(TeleopState.IDLE)

    def _initialization_worker_thread(self) -> None:
        """Worker thread that handles object tracking initialization."""
        try:
            self.get_logger().info(
                "🔄 Starting object tracking initialization in background..."
            )

            # Perform the initialization
            success = self._initialize_object_tracking()

            if success:
                self.initialization_success.set()
                self.get_logger().info(
                    "✅ Object tracking initialization completed successfully"
                )
            else:
                self.get_logger().error("❌ Object tracking initialization failed")

        except Exception as e:
            self.get_logger().error(
                f"❌ Object tracking initialization failed with exception: {e}"
            )

        finally:
            # Always signal completion
            self.initialization_complete.set()
            self.initialization_in_progress = False

    def _pose_tracking_worker_thread(self) -> None:
        """Worker thread that handles pose tracking predictions."""
        while not self.pose_tracking_shutdown.is_set():
            try:
                # Wait for trigger or shutdown with minimal timeout for fast response
                if self.pose_tracking_trigger.wait(
                    timeout=0.001
                ):  # 1ms timeout for minimal delay
                    self.pose_tracking_trigger.clear()

                    # Set in progress flag
                    self.pose_tracking_in_progress.set()

                    try:
                        self._perform_pose_tracking()
                    finally:
                        # Always clear the in progress flag
                        self.pose_tracking_in_progress.clear()

            except Exception as e:  # noqa: PERF203
                if not self.pose_tracking_shutdown.is_set():
                    self.get_logger().error(
                        f"Error in pose tracking worker thread: {e}"
                    )
                # Make sure to clear the in progress flag on error
                self.pose_tracking_in_progress.clear()

    def _perform_pose_tracking(self) -> None:
        """Perform the actual pose tracking prediction (runs on pose tracking thread)."""
        # Sample latest images directly from self.images (atomic read operations)
        static_rgb = self.images.get("static_top_rgb")
        static_depth = self.images.get("static_top_depth")
        static_k = self.camera_k_matrices.get("static_top_rgb")

        # Check if we have valid images
        if static_rgb is None or static_depth is None or static_k is None:
            return

        camera_intrinsics = np.array(static_k).reshape(3, 3)

        # Check if any objects are being tracked
        objects_being_tracked = [
            target_key
            for target_key in ["pick_target", "place_target"]
            if self.pose_tracking_initialized.get(target_key, False)
        ]

        if not objects_being_tracked or self.pose_client is None:
            return

        # Verify context is still active
        if self.pose_client_context is None:
            self.get_logger().warning(
                "Pose client context not active, skipping pose tracking"
            )
            return

        try:
            # Use persistent connection (no context manager)
            track_response = self.pose_client.predict_poses(
                static_rgb, static_depth, None, camera_intrinsics
            )

            # Extract tracking results
            tracked_poses_7d = track_response.get("poses", {})
            tracked_confidences = track_response.get("confidences", {})

            # Update pose tracking state for each object (thread-safe)
            with self.pose_tracking_lock:
                for target_key in objects_being_tracked:
                    pose_name = OBJECT_NAME_MAPPING[target_key]["pose_name"]

                    if pose_name in tracked_poses_7d:
                        pose_7d = tracked_poses_7d[pose_name]
                        confidence = tracked_confidences.get(pose_name, 0.0)

                        if confidence >= POSE_CONFIDENCE_THRESHOLD:
                            # Apply custom rotation to the pose
                            modified_pose_7d = self.apply_custom_rotation(
                                pose_7d, target_key
                            )

                            # Store camera coordinates for visualization
                            self.target_poses_camera[f"{target_key}_pose"] = (
                                modified_pose_7d
                            )

                            # Transform from camera coordinates to world coordinates
                            # if calibration is available
                            if self.calibration_data is not None:
                                world_pose_7d = self.calibration_data.transform_pose_camera_to_world(
                                    modified_pose_7d
                                )
                                # Store world coordinates for behavior cloning/RL models
                                self.target_poses[f"{target_key}_pose"] = world_pose_7d
                            else:
                                # No calibration available, use camera coordinates for both
                                self.target_poses[f"{target_key}_pose"] = (
                                    modified_pose_7d
                                )

        except Exception as e:
            # Handle connection closed gracefully
            if (
                "ConnectionClosed" in str(type(e).__name__)
                or "connection" in str(e).lower()
            ):
                self.get_logger().debug(
                    "WebSocket connection closed during pose tracking"
                )
                return
            else:
                self.get_logger().warning(f"Failed to track object poses: {e}")
            # Make sure to clear the in progress flag on error
            self.pose_tracking_in_progress.clear()

    def init_io_topics(self) -> None:
        """Initialize the io topics."""
        # Keyboard operator message from the robot control UI
        self.subscription_keyboard_operator_message = self.create_subscription(
            String,
            "/keyboard_operator_message",
            self.keyboard_operator_message_callback,
            QUEUE_SIZE,
        )

    def init_teleop_state_topic(self) -> None:
        """Initialize the teleop state topic."""
        self.publisher_teleop_state = self.create_publisher(
            String, "/teleop_bridge/teleop_state", QUEUE_SIZE
        )

    @staticmethod
    def require_estop_released(func: Callable) -> Callable:
        """Decorator to ensure the e-stop is released before executing the function."""

        def wrapper(self, *args, **kwargs):
            if self.teleop_state == TeleopState.E_STOPPED:
                self.get_logger().warning("E-stop is active. Release E-stop first.")
                return
            return func(self, *args, **kwargs)

        return wrapper

    def check_arm_connection(self) -> None:
        """Check if the arms are connected."""
        max_dt = -1.0
        current_time = self.get_clock().now().to_msg().sec
        for last_msg in self.last_msg_set.values():
            if last_msg is None:
                continue
            dt = current_time - last_msg.header.stamp.sec
            max_dt = max(dt, max_dt)

        if max_dt < 0.0 or max_dt > ARM_CONN_TIMEOUT_SEC:
            self.get_logger().warning(f"Max arm connection time: {max_dt} seconds")
            self.get_logger().warning("Check the arm connection.")

            self.handle_e_stop(ArmType.LEADER, EStopType.SOFT, True)

    def get_header(self) -> Header:
        """Get a header with the current time and frame id."""
        return Header(stamp=self.get_clock().now().to_msg(), frame_id=self.frame_id)

    def validate_joint_state(self, msg: Optional[JointState]) -> bool:
        """Validate a joint state message."""
        if msg is None:
            self.get_logger().error("[validate_joint_state] msg is None")
            return False
        elif len(msg.position) == 0:
            self.get_logger().error("[validate_joint_state] msg.position is empty")
            return False
        return True

    def create_joint_state_callback(
        self, arm_type: ArmType
    ) -> Callable[[JointState], None]:
        """Create a joint state callback for a given arm."""

        def joint_state_callback(joint_state_msg: JointState) -> None:
            """Callback for leader left arm joint states."""
            self.validate_joint_state(joint_state_msg)

            if (
                arm_type in {ArmType.LEADER_LEFT, ArmType.LEADER_RIGHT}
                and self.teleop_state == TeleopState.HUMAN_OPERATOR
            ):
                self.sync_joint_state(arm_type, joint_state_msg)

            self.last_msg_set[arm_type] = joint_state_msg

        return joint_state_callback

    def _apply_j6_to_tool_tip_offset(
        self, pose_msg: PoseStamped, arm_type: ArmType
    ) -> PoseStamped:
        """Apply J6 to tool tip offset and transform from robot to world coordinates."""
        # Only apply offsets to right arm for now
        if arm_type != ArmType.FOLLOWER_RIGHT:
            return pose_msg

        # Extract J6 position and orientation
        j6_pos = np.array([
            pose_msg.pose.position.x,
            pose_msg.pose.position.y,
            pose_msg.pose.position.z,
        ])
        j6_quat = np.array([
            pose_msg.pose.orientation.x,
            pose_msg.pose.orientation.y,
            pose_msg.pose.orientation.z,
            pose_msg.pose.orientation.w,
        ])

        # Convert quaternion to rotation matrix and apply J6 to tool tip offset
        j6_rotation = Rotation.from_quat(j6_quat)

        # Apply rotation to the J6 to tool tip offset vector
        # (transform from J6 frame to world frame)
        j6_to_tool_tip_offset_world = j6_rotation.apply(J6_TO_TOOL_TIP_OFFSET)

        # Calculate tool tip position (still in robot coordinates)
        tool_tip_pos_robot = j6_pos + j6_to_tool_tip_offset_world

        # Transform from robot coordinates to world coordinates
        # by adding RIGHT_ARM_OFFSET
        tool_tip_pos_world = tool_tip_pos_robot + np.array(RIGHT_ARM_OFFSET)

        # Create new pose message with adjusted position
        adjusted_pose_msg = PoseStamped()
        adjusted_pose_msg.header = pose_msg.header
        adjusted_pose_msg.pose.position.x = tool_tip_pos_world[0]
        adjusted_pose_msg.pose.position.y = tool_tip_pos_world[1]
        adjusted_pose_msg.pose.position.z = tool_tip_pos_world[2]
        # Keep the same orientation (J6 orientation)
        adjusted_pose_msg.pose.orientation = pose_msg.pose.orientation

        return adjusted_pose_msg

    def _log_pose_details(
        self,
        tool_pose_right: list[float],
        pick_target_pose: list[float],
        frame_index: int,
    ) -> None:
        """Log detailed pose information for debugging."""
        try:
            # Extract positions and orientations
            ee_pos = np.array(tool_pose_right[:3])
            ee_quat = np.array(tool_pose_right[3:])  # [qx, qy, qz, qw]

            target_pos = np.array(pick_target_pose[:3])
            target_quat = np.array(pick_target_pose[3:])  # [qx, qy, qz, qw]

            # Calculate position delta
            delta_pos = target_pos - ee_pos

            # Convert quaternions to euler angles (roll, pitch, yaw)
            ee_rotation = Rotation.from_quat(ee_quat)
            target_rotation = Rotation.from_quat(target_quat)

            ee_euler = ee_rotation.as_euler("xyz", degrees=True)  # [roll, pitch, yaw]
            target_euler = target_rotation.as_euler(
                "xyz", degrees=True
            )  # [roll, pitch, yaw]

            # Calculate orientation delta
            delta_euler = target_euler - ee_euler

            # Normalize angles to [-180, 180] range
            delta_euler = ((delta_euler + 180) % 360) - 180

            # Log the detailed information
            self.get_logger().info(f"[Frame {frame_index}] Pose Details:")
            self.get_logger().info("  Right arm tool pose:")
            self.get_logger().info(
                f"    Position: [{ee_pos[0]:.4f}, {ee_pos[1]:.4f}, {ee_pos[2]:.4f}]"
            )
            self.get_logger().info(
                f"    Orientation (RPY): [{ee_euler[0]:.2f}°, {ee_euler[1]:.2f}°, {ee_euler[2]:.2f}°]"
            )
            self.get_logger().info("  Pick target pose:")
            self.get_logger().info(
                f"    Position: [{target_pos[0]:.4f}, {target_pos[1]:.4f}, {target_pos[2]:.4f}]"
            )
            self.get_logger().info(
                f"    Orientation (RPY): [{target_euler[0]:.2f}°, {target_euler[1]:.2f}°, {target_euler[2]:.2f}°]"
            )
            self.get_logger().info("  Delta (target - ee):")
            self.get_logger().info(
                f"    Position: [{delta_pos[0]:.4f}, {delta_pos[1]:.4f}, {delta_pos[2]:.4f}]"
            )
            self.get_logger().info(
                f"    Orientation (RPY): [{delta_euler[0]:.2f}°, {delta_euler[1]:.2f}°, {delta_euler[2]:.2f}°]"
            )
            self.get_logger().info(f"    Distance: {np.linalg.norm(delta_pos):.4f}")

        except Exception as e:
            self.get_logger().error(f"Error logging pose details: {e}")

    def create_ee_pose_callback(self, arm_type: ArmType):
        """Create an end-effector pose callback for a given arm."""

        def ee_pose_callback(pose_msg: PoseStamped) -> None:
            """Callback for end-effector pose updates."""
            self.last_ee_pose_set[arm_type] = self._apply_j6_to_tool_tip_offset(
                pose_msg, arm_type
            )

        return ee_pose_callback

    def get_publishers_for_arm(self, arm_type: ArmType) -> Tuple:
        """Get the appropriate publishers for the given arm type."""
        if arm_type in {ArmType.LEFT, ArmType.LEADER_LEFT}:
            return (
                self.publisher_leader_l,
                self.publisher_follower_l,
                self.publisher_follower_g1_gripper_l,
                LEADER_L_J6_OFF_DEG,
            )
        elif arm_type in {ArmType.RIGHT, ArmType.LEADER_RIGHT}:
            return (
                self.publisher_leader_r,
                self.publisher_follower_r,
                self.publisher_follower_g1_gripper_r,
                LEADER_R_J6_OFF_DEG,
            )
        else:
            self.get_logger().warning(f"Invalid arm type for publishers: {arm_type}")
            return None, None, None, 0.0

    @require_estop_released
    def sync_joint_state(
        self,
        arm_type: ArmType,
        joint_state_msg: Optional[JointState],
        velocity: float = DEFAULT_VELOCITY,
        effort: float = DEFAULT_EFFORT,
    ) -> None:
        """Common processing for joint states for both arms"""
        if joint_state_msg is None:
            self.get_logger().warning("[sync_joint_state] msg is None")
            return
        if len(joint_state_msg.position) == 0:
            self.get_logger().warning("[sync_joint_state] msg.position is empty")
            return

        # Send joint target positions to the follower
        new_joint_state = JointState()
        new_joint_state.header = self.get_header()
        new_joint_state.name = self.joint_names
        new_joint_state.velocity = [velocity] * len(self.joint_names)
        new_joint_state.effort = [effort] * len(self.joint_names)
        new_joint_state.position = joint_state_msg.position

        if arm_type in self.last_msg_set:
            last_pos = np.array(self.last_msg_set[arm_type].position)
            new_pos = np.array(joint_state_msg.position)
            dt_pos = new_pos - last_pos
            if np.sum(np.abs(dt_pos)) < np.deg2rad(DEAD_ZONE_THRESH_DEG):
                new_joint_state.position = self.last_msg_set[arm_type].position
                new_joint_state.velocity = [0.0] * len(self.joint_names)

        leader_pub, follower_pub, gripper_pub, j6_offset = self.get_publishers_for_arm(
            arm_type
        )

        if leader_pub is None:
            self.get_logger().warning(
                f"[sync_joint_state] invalid arm type: {arm_type}"
            )
            return

        # Publish to leader
        leader_pub.publish(new_joint_state)

        # Apply J6 offset and publish to follower
        follower_joint_state = JointState()
        follower_joint_state.header = new_joint_state.header
        follower_joint_state.name = new_joint_state.name
        follower_joint_state.velocity = new_joint_state.velocity
        follower_joint_state.effort = new_joint_state.effort
        follower_joint_state.position = list(new_joint_state.position)
        follower_joint_state.position[-2] -= np.deg2rad(j6_offset)
        follower_pub.publish(follower_joint_state)

        # Publish gripper position
        self.publish_g1_gripper_target_position(
            gripper_pub, new_joint_state.position[-1]
        )

    @require_estop_released
    def publish_g1_gripper_target_position(self, publisher, pos: float) -> None:
        """Publish the gripper target position."""
        new_gripper_state = GripperPositionControl()
        new_gripper_state.header = self.get_header()
        new_gripper_state.gripper_stroke = pos * GRIPPER_POSITION_SCALE
        publisher.publish(new_gripper_state)

    def set_teleop_state(self, state: TeleopState) -> None:
        """Set the teleop state."""
        self.teleop_state = state
        self.get_logger().info(f"Teleop state set to: {state}")

        msg = String()
        msg.data = state.value
        self.publisher_teleop_state.publish(msg)

    @require_estop_released
    def move_arms_to_position(self, position: PositionType) -> None:
        """Move both arms to the specified position.

        Args:
            position: The position to move to (PositionType.HOME or PositionType.ZERO)
        """
        if position not in PositionType.__members__.values():
            self.get_logger().error(f"Unknown position: {position}")
            return

        self.get_logger().info(f"Moving arms to {position} position.")
        self.set_teleop_state(TeleopState.PROGRAM)

        # Get position dict for the requested position type
        position_dict = self.positions.get(position)
        if not position_dict:
            self.get_logger().error(f"No position data for {position}")
            return

        for arm_type in [ArmType.LEFT, ArmType.RIGHT]:
            self.sync_joint_state(
                arm_type,
                JointState(position=position_dict[arm_type]),
                velocity=SAFE_VELOCITY,
                effort=SAFE_EFFORT,
            )

    @require_estop_released
    def hold_position(self) -> None:
        """Hold the position of the arms."""
        self.get_logger().info("Holding position.")

        if self.teleop_state != TeleopState.AWAITING_FEEDBACK:
            self.set_teleop_state(TeleopState.IDLE)

        self.sync_joint_state(
            ArmType.LEFT,
            self.last_msg_set[ArmType.LEADER_LEFT],
            velocity=0.0,
            effort=0.0,
        )
        self.sync_joint_state(
            ArmType.RIGHT,
            self.last_msg_set[ArmType.LEADER_RIGHT],
            velocity=0.0,
            effort=0.0,
        )

    @require_estop_released
    def toggle_hold_left_arm(self) -> None:
        """Toggle holding the left arm in place."""
        self.is_left_arm_held = not self.is_left_arm_held
        self.get_logger().info(
            "Left arm is held." if self.is_left_arm_held else "Left arm is free."
        )
        if self.is_left_arm_held:
            self.sync_joint_state(
                ArmType.LEFT,
                self.last_msg_set[ArmType.LEADER_LEFT],
                velocity=0.0,
                effort=0.0,
            )

    def is_recording(self) -> bool:
        """Check if recording is enabled or pending."""
        return self.recording_enabled or self.recording_pending

    def record_step(self) -> None:
        """Record one step of teleoperation data at 50Hz (non-blocking)."""
        if not self.recording_enabled:
            return

        # Get current joint states
        left_joint_state = self.last_msg_set[ArmType.FOLLOWER_LEFT]
        right_joint_state = self.last_msg_set[ArmType.FOLLOWER_RIGHT]
        left_ee_pose = self.last_ee_pose_set[ArmType.FOLLOWER_LEFT]
        right_ee_pose = self.last_ee_pose_set[ArmType.FOLLOWER_RIGHT]

        # Check if all required data is available with detailed logging
        if (
            not left_joint_state.position
            or not right_joint_state.position
            or len(left_joint_state.position) < 7  # noqa: PLR2004
            or len(right_joint_state.position) < 7  # noqa: PLR2004
        ):
            self.missing_data_counters["joint_states"] += 1
            if (
                self.missing_data_counters["joint_states"] == 1
            ):  # Log only the first occurrence
                self.get_logger().warning(
                    f"⚠️ Missing joint states: left={len(left_joint_state.position) if left_joint_state.position else 0}/7, "
                    f"right={len(right_joint_state.position) if right_joint_state.position else 0}/7"
                )
            return

        if left_ee_pose is None or right_ee_pose is None:
            self.missing_data_counters["ee_poses"] += 1
            if (
                self.missing_data_counters["ee_poses"] == 1
            ):  # Log only the first occurrence
                self.get_logger().warning(
                    f"⚠️ Missing end-effector poses: left={'✓' if left_ee_pose else '✗'}, "
                    f"right={'✓' if right_ee_pose else '✗'}"
                )
            return

        # Check if all images are available
        missing_images = [name for name, img in self.images.items() if img is None]
        if missing_images:
            self.missing_data_counters["images"] += 1
            if (
                self.missing_data_counters["images"] == 1
            ):  # Log only the first occurrence
                self.get_logger().warning(f"⚠️ Missing camera images: {missing_images}")
            return

        try:
            # Timing calculations
            current_time = time.time()
            if self.episode_start_time is None:
                self.episode_start_time = current_time
                self.last_frame_time = current_time

            elapsed_ms = (current_time - self.episode_start_time) * 1000.0
            delta_ms = (
                (current_time - self.last_frame_time) * 1000.0
                if self.last_frame_time
                else 20.0
            )
            self.last_frame_time = current_time

            # Extract joint states (7 values each: 6 joints + 1 gripper)
            joint_states_left = list(left_joint_state.position[:7])
            joint_states_right = list(right_joint_state.position[:7])

            # Extract tool poses (7 values each: 3 position + 4 quaternion)
            tool_pose_left = [
                left_ee_pose.pose.position.x,
                left_ee_pose.pose.position.y,
                left_ee_pose.pose.position.z,
                left_ee_pose.pose.orientation.x,
                left_ee_pose.pose.orientation.y,
                left_ee_pose.pose.orientation.z,
                left_ee_pose.pose.orientation.w,
            ]
            tool_pose_right = [
                right_ee_pose.pose.position.x,
                right_ee_pose.pose.position.y,
                right_ee_pose.pose.position.z,
                right_ee_pose.pose.orientation.x,
                right_ee_pose.pose.orientation.y,
                right_ee_pose.pose.orientation.z,
                right_ee_pose.pose.orientation.w,
            ]

            # Create simplified observation dict (without image paths yet)
            observation: dict = {
                "frame_index": self.frame_index,
                "delta_ms": delta_ms,
                "elapsed_ms": elapsed_ms,
                "joint_states_left": joint_states_left,
                "joint_states_right": joint_states_right,
                "tool_pose_left": tool_pose_left,
                "tool_pose_right": tool_pose_right,
            }

            # Add tracked object poses (thread-safe access)
            with self.pose_tracking_lock:
                observation["pick_target_pose"] = self.target_poses[
                    "pick_target_pose"
                ].copy()
                observation["place_target_pose"] = self.target_poses[
                    "place_target_pose"
                ].copy()

            # Verbose logging of poses and deltas
            if self.verbose:
                self._log_pose_details(
                    tool_pose_right, observation["pick_target_pose"], self.frame_index
                )

            # Copy images for thread safety
            # (shallow copy is sufficient for numpy arrays)
            images_copy = {
                name: img.copy() if img is not None else None
                for name, img in self.images.items()
            }

            # Get camera intrinsics for visualization
            static_k = self.camera_k_matrices.get("static_top_rgb")
            camera_intrinsics = np.array(static_k).reshape(3, 3) if static_k else None

            # Package data for worker thread
            recording_data = {
                "frame_index": self.frame_index,
                "images": images_copy,
                "observation": observation,
                "raw_data_dir": self.current_raw_data_dirpath,
                "camera_intrinsics": camera_intrinsics,
            }

            # Queue the data for processing in worker thread (non-blocking)
            try:
                self.recording_queue.put_nowait(recording_data)
                self.frame_index += 1

                # Track successful observation recording
                self.last_successful_observation_time = current_time

                # Log the first successful observation to confirm recording is working
                if self.frame_index == 1:
                    self.get_logger().info(
                        "✅ First observation successfully recorded!"
                    )
                elif self.frame_index % 250 == 0:  # Log every 5 seconds at 50Hz
                    self.get_logger().info(
                        f"📊 Recording progress: {self.frame_index} observations"
                    )

            except Exception as queue_error:
                self.missing_data_counters["queue_full"] += 1
                if (
                    self.missing_data_counters["queue_full"] == 1
                ):  # Log only the first occurrence
                    self.get_logger().warning(
                        f"⚠️ Recording queue full, dropping frames. Queue size: {self.recording_queue.qsize()}/100"
                    )
                self.get_logger().warning(
                    f"Recording queue full, dropping frame {self.frame_index}: "
                    f"{queue_error}"
                )

        except Exception as e:
            self.get_logger().error(f"Error recording step: {e}")

    @require_estop_released
    def start_episode(self) -> None:
        """Start a new episode."""
        self.get_logger().info("Starting a new episode recording.")

        self.is_left_arm_held = False

        if self.is_recording() or self.recording_pending:
            self.get_logger().warning(
                "🚨 Recording is already active or pending. Please stop it first."
            )
            return

        if self.current_episode_dirpath is not None:
            self.get_logger().warning(
                "🚨 Previous episode has not been annotated. "
                "Please annotate the episode first before starting a new one."
            )

        # Check if initialization is already in progress
        if self.initialization_in_progress:
            self.get_logger().warning(
                "🚨 Object tracking initialization is already in progress. Please wait."
            )
            return

        # Enter WebSocket contexts for the episode (persistent connections)
        self.get_logger().info("🔗 Connecting to WebSocket servers...")
        if not self._enter_websocket_contexts():
            self.get_logger().error(
                "❌ Failed to connect to WebSocket servers, cannot start episode"
            )
            return

        # Reset recording state (but don't enable recording yet)
        self.frame_index = 0
        self.observations = []
        self.episode_start_time = None
        self.last_frame_time = None

        # Reset observation monitoring state
        self.last_successful_observation_time = None
        self.last_observation_check_time = None
        self.missing_data_counters = {
            "joint_states": 0,
            "ee_poses": 0,
            "images": 0,
            "queue_full": 0,
        }

        # Create episode directory
        now = datetime.now()
        output_dir = self.episode_dirpath / now.strftime("%Y/%m/%d")
        output_dir.mkdir(parents=True, exist_ok=True)
        self.current_episode_dirpath = output_dir / now.strftime("%Y%m%d_%H%M%S")
        self.current_episode_dirpath.mkdir(parents=True, exist_ok=True)
        self.get_logger().info(
            f"🎥 Episode recording path: {self.current_episode_dirpath}"
        )
        self.current_episode_name = self.current_episode_dirpath.name

        self.current_raw_data_dirpath = self.current_episode_dirpath / "raw_data"
        self.current_raw_data_dirpath.mkdir(parents=True, exist_ok=True)

        # Start recording worker thread if not already running
        if self.recording_thread is None or not self.recording_thread.is_alive():
            self.recording_thread_shutdown.clear()
            self.recording_thread = threading.Thread(
                target=self._recording_worker_thread,
                daemon=True,
                name="recording_worker",
            )
            self.recording_thread.start()
            self.get_logger().info("Recording worker thread started")

        # Start pose tracking worker thread if not already running
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

        # Set recording as pending (waiting for initialization)
        self.recording_pending = True
        self.set_teleop_state(
            TeleopState.RECORDING
        )  # Show that we're starting recording

        # Start initialization on background thread (non-blocking)
        self.get_logger().info(
            "🔄 Starting object detection and pose tracking initialization in background..."
        )
        self.get_logger().info(
            "⏳ This may take 5-10 seconds. The robot will remain responsive during initialization."
        )

        # Reset initialization events
        self.initialization_complete.clear()
        self.initialization_success.clear()
        self.initialization_in_progress = True

        # Start initialization worker thread
        self.initialization_thread = threading.Thread(
            target=self._initialization_worker_thread,
            daemon=True,
            name="initialization_worker",
        )
        self.initialization_thread.start()
        self.get_logger().info("Initialization worker thread started")

    def stop_episode(self) -> None:
        """Stop the current episode."""
        self.get_logger().info("Stopping the current episode recording.")
        if not self.recording_enabled and not self.recording_pending:
            self.get_logger().warning("No recording is active or pending.")
            return

        # Stop recording and pending recording
        self.recording_enabled = False
        self.recording_pending = False

        # If initialization is in progress, stop it
        if self.initialization_in_progress:
            self.get_logger().info("Stopping initialization in progress...")
            self.initialization_shutdown.set()
            # Wait for initialization thread to complete
            if self.initialization_thread and self.initialization_thread.is_alive():
                self.initialization_thread.join(timeout=2.0)
                if self.initialization_thread.is_alive():
                    self.get_logger().warning(
                        "Initialization thread did not stop gracefully"
                    )
            self.initialization_shutdown.clear()
            self.initialization_in_progress = False

        # Reset pose tracking state
        self._reset_pose_tracking()

        # Exit WebSocket contexts (episode is over)
        self.get_logger().info("🔌 Disconnecting WebSocket clients...")
        self._exit_websocket_contexts()

        # Wait for all queued data to be processed
        self.get_logger().info("Waiting for recording queue to be processed...")
        if self.recording_thread and self.recording_thread.is_alive():
            # Wait for queue to be empty with timeout
            timeout_counter = 0
            while (
                not self.recording_queue.empty() and timeout_counter < 50  # noqa: PLR2004
            ):  # 5 second timeout
                time.sleep(0.1)
                timeout_counter += 1

            if not self.recording_queue.empty():
                self.get_logger().warning(
                    f"Recording queue not empty after timeout "
                    f"({self.recording_queue.qsize()} items remaining)"
                )

        # Always wait for feedback if an episode directory was created,
        # regardless of whether observations were recorded or initialization succeeded
        if self.current_episode_dirpath is not None:
            # Finalize episode data if we have observations
            if self.observations:
                self._finalize_episode()
                self.get_logger().info(
                    f"✅ Episode with {len(self.observations)} observations awaiting feedback"
                )
            else:
                self.get_logger().info(
                    "⚠️ Episode with no observations awaiting feedback (may indicate failure)"
                )

            self.set_teleop_state(TeleopState.AWAITING_FEEDBACK)
        else:
            # No episode directory was created, go directly to idle
            self.get_logger().info("No episode directory to process, returning to idle")
            self.set_teleop_state(TeleopState.IDLE)

    def _reset_pose_tracking(self) -> None:
        """Reset pose tracking state and clean up pose estimation session."""
        self.get_logger().info("Resetting pose tracking state...")

        # Clear any pending pose tracking requests
        self.pose_tracking_trigger.clear()

        # Wait for any in-progress prediction to complete (with timeout)
        if self.pose_tracking_in_progress.is_set():
            self.get_logger().info(
                "Waiting for in-progress pose prediction to complete..."
            )
            self.pose_tracking_in_progress.wait(timeout=1.0)

        # Reset tracking flags
        with self.pose_tracking_lock:
            self.pose_tracking_initialized = {
                "pick_target": False,
                "place_target": False,
            }
            self.target_poses = {
                "pick_target_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                "place_target_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            }
            self.target_poses_camera = {
                "pick_target_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                "place_target_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            }

        # Reset pose estimation session (only if context is active)
        try:
            if self.pose_client is not None and self.pose_client_context is not None:
                # Use persistent connection (no context manager)
                self.pose_client.reset_session()
        except Exception as e:
            # Handle connection closed gracefully
            if (
                "ConnectionClosed" in str(type(e).__name__)
                or "connection" in str(e).lower()
            ):
                self.get_logger().debug(
                    "WebSocket connection closed during session reset"
                )
            else:
                self.get_logger().warning(f"Failed to reset pose client session: {e}")

        self.get_logger().info("Pose tracking state reset completed")

    def _finalize_episode(self) -> None:
        """Finalize episode recording and save metadata."""
        if not self.current_raw_data_dirpath or not self.observations:
            self.get_logger().warning("No episode data to finalize.")
            return

        try:
            # Create simplified metadata
            metadata = {
                "episode_name": self.current_episode_name,
                "robot_id": ROBOT_ID,
                "k_mats": self.camera_k_matrices.copy(),
                "original_k_mats": self.camera_original_k_matrices.copy(),
                "camera_image_dimensions": self.camera_image_dimensions.copy(),
                "image_processing": {
                    "resize_enabled": RESIZE_IMAGE,
                    "target_image_size": (IMAGE_SIZE, IMAGE_SIZE)
                    if RESIZE_IMAGE
                    else None,
                    "k_matrices_adjusted": RESIZE_IMAGE,
                },
            }

            episode_data = {"metadata": metadata, "observations": self.observations}

            # Save observations JSON
            out_path = self.current_raw_data_dirpath / "observations.json"
            out_path.write_text(custom_json_dumps(episode_data, max_indent_level=3))

            self.get_logger().info(f"✅ Episode saved: {out_path}")
            self.get_logger().info(f"Total frames recorded: {len(self.observations)}")

        except Exception as e:
            self.get_logger().error(f"Error finalizing episode: {e}")

    def feedback_callback(self, msg: String) -> None:
        """Callback for feedback messages"""
        self.get_logger().info(f"Received feedback: {msg.data}")
        # (Hacky) Parse the feedback message: format "feedback_good" or "feedback_bad"
        grade = msg.data.split("_")[-1]

        if self.current_episode_dirpath is None:
            self.get_logger().warning(
                "Not currently expecting feedback. Please record an episode first."
            )
            return

        # Determine episode characteristics for better logging
        num_observations = len(self.observations) if self.observations else 0
        episode_type = "successful" if num_observations > 0 else "failed/incomplete"

        try:
            if grade == "good":
                # Save the episode annotation for good episodes
                metadata_path = (
                    self.current_episode_dirpath
                    / "feature_store/episode_annotation/episode_annotation.json"
                )
                metadata_path.parent.mkdir(parents=True, exist_ok=True)
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump({"grade": grade}, f, indent=4)
                self.get_logger().info(
                    f"💾 {episode_type.capitalize()} episode marked as GOOD and saved: {self.current_episode_dirpath} "
                    f"({num_observations} observations)"
                )

            elif grade == "bad":
                # Remove the entire episode directory for bad episodes
                episode_path = self.current_episode_dirpath
                shutil.rmtree(episode_path)
                self.get_logger().info(
                    f"🗑️ {episode_type.capitalize()} episode marked as BAD and deleted: {episode_path} "
                    f"({num_observations} observations)"
                )
            else:
                self.get_logger().warning(f"🚨 Unknown grade: {grade}")
                return

            self.current_episode_dirpath = None
        except Exception as e:
            self.get_logger().error(
                f"Failed to process feedback for grade '{grade}': {e}"
            )

        self.set_teleop_state(TeleopState.IDLE)

    def keyboard_operator_message_callback(self, msg: String) -> None:
        """Callback for input messages"""
        self.get_logger().info(f"Received keyboard input: {msg.data}")

        if msg.data == "home":
            self.move_arms_to_position(PositionType.HOME)
        elif msg.data == "zero":
            self.move_arms_to_position(PositionType.ZERO)
        elif msg.data == "hold_left":
            self.toggle_hold_left_arm()
        elif msg.data == "start_episode":
            self.start_episode()
        elif msg.data == "end_episode":
            self.stop_episode()
        elif msg.data == "e_stop":
            self.handle_e_stop(ArmType.LEADER, EStopType.SOFT, True)
        elif msg.data == "e_stop_reset":
            self.handle_e_stop(ArmType.LEADER, EStopType.SOFT, False)
        elif msg.data == "feedback_good":
            self.feedback_callback(msg)
        elif msg.data == "feedback_bad":
            self.feedback_callback(msg)
        elif msg.data == "operator_in_control":
            self.set_teleop_state(TeleopState.HUMAN_OPERATOR)
        elif msg.data == "hold_position":
            self.hold_position()
        elif msg.data == "toggle_camera_viewer":
            self.toggle_camera_viewer()
        elif msg.data == "quit":
            pass
        else:
            self.get_logger().warning(f"Received unknown input message: {msg.data}")

    def toggle_camera_viewer(self) -> None:
        """Toggle camera viewer on/off."""
        if self.camera_viewer_enabled:
            self.camera_viewer_enabled = False
            self._cleanup_camera_viewer()
            self.get_logger().info("Camera viewer disabled")
        else:
            self.camera_viewer_enabled = True
            self._init_camera_viewer()
            # Create camera viewer timer if it doesn't exist
            if not hasattr(self, "camera_viewer_timer"):
                self.camera_viewer_timer = self.create_timer(
                    1.0 / CAMERA_VIEWER_FPS, self.update_camera_viewer_display
                )
            self.get_logger().info("Camera viewer enabled")

    def handle_e_stop(
        self, arm_type: ArmType, e_stop_type: EStopType, state: bool
    ) -> None:
        """Handle e-stop states."""
        if state != self.e_stop_states[arm_type][e_stop_type]:
            self.get_logger().info(
                f"Received e-stop message: {state} ({arm_type} {e_stop_type})"
            )

        if state:
            self.e_stop_states[arm_type][e_stop_type] = True
            if self.teleop_state != TeleopState.E_STOPPED:
                self.get_logger().info("E-stop engaged. Holding position.")
                self.set_teleop_state(TeleopState.E_STOPPED)
                self.hold_position()
        else:
            self.e_stop_states[arm_type][e_stop_type] = False
            if (
                not self.check_all_e_stop_states()
                and self.teleop_state == TeleopState.E_STOPPED
            ):
                self.get_logger().info("E-stop released. Returning to idle.")
                self.set_teleop_state(TeleopState.IDLE)

    def check_all_e_stop_states(self) -> bool:
        """Check if all e-stop states are active."""
        for arm_type in {ArmType.LEADER, ArmType.FOLLOWER}:
            for e_stop_type in {EStopType.SOFT, EStopType.HARD}:
                if self.e_stop_states[arm_type][e_stop_type]:
                    return True
        return False

    def _initialize_object_tracking(self) -> bool:
        """Initialize object detection and pose tracking for the episode.

        Returns:
            True if initialization successful, False otherwise
        """
        if self.detection_client is None:
            self.get_logger().error("Detection client not initialized")
            return False

        if self.pose_client is None:
            self.get_logger().error("Pose client not initialized")
            return False

        # Verify contexts are active
        if self.detection_client_context is None or self.pose_client_context is None:
            self.get_logger().error("WebSocket contexts not active")
            return False

        # Wait for static camera images to be available
        max_wait_time = 5.0  # seconds
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            # Check for shutdown signal
            if self.initialization_shutdown.is_set():
                self.get_logger().info("Initialization interrupted by shutdown signal")
                return False

            static_rgb = self.images.get("static_top_rgb")
            static_depth = self.images.get("static_top_depth")
            static_k = self.camera_k_matrices.get("static_top_rgb")

            if (
                static_rgb is not None
                and static_depth is not None
                and static_k is not None
            ):
                break
            time.sleep(0.1)
        else:
            self.get_logger().error(
                "Timeout waiting for static camera images and intrinsics"
            )
            return False

        camera_intrinsics = np.array(static_k).reshape(3, 3)

        self.get_logger().info(
            "Performing object detection for pick and place targets..."
        )

        # Detect both objects and collect their data
        objects_data = []
        successful_detections = []

        for target_key, names in OBJECT_NAME_MAPPING.items():
            # Check for shutdown signal
            if self.initialization_shutdown.is_set():
                self.get_logger().info("Initialization interrupted by shutdown signal")
                return False

            detection_name = names["detection_name"]
            pose_name = names["pose_name"]

            self.get_logger().info(f"Detecting {target_key}: '{detection_name}'...")

            try:
                # Perform object detection (using persistent connection)
                success, message, boxes, scores, labels, mask = (
                    self.detection_client.detect_objects(static_rgb, detection_name)
                )

                if not success or len(boxes) == 0:
                    self.get_logger().warning(
                        f"Object detection failed for {detection_name}: {message}"
                    )
                    continue

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
                    self.get_logger().info(
                        f"Converting {target_key} mask from {best_mask.dtype} to uint8"
                    )
                    if best_mask.max() <= 1.0:
                        best_mask = (best_mask * 255).astype(np.uint8)
                    else:
                        best_mask = best_mask.astype(np.uint8)

                if best_mask.max() == 1:
                    best_mask *= 255

                # Add to objects data for multi-object initialization
                objects_data.append({
                    "object_name": pose_name,
                    "rgb_image": static_rgb,  # Already in RGB format for pose server
                    "depth_image": static_depth,
                    "mask_image": best_mask,
                    "camera_intrinsics": camera_intrinsics,
                })

                successful_detections.append(target_key)

                self.get_logger().info(
                    f"Prepared {target_key} for pose initialization: "
                    f"mask shape={best_mask.shape}, dtype={best_mask.dtype}, "
                    f"range=[{best_mask.min()}, {best_mask.max()}]"
                )

            except Exception as e:
                # Handle connection closed gracefully
                if (
                    "ConnectionClosed" in str(type(e).__name__)
                    or "connection" in str(e).lower()
                ):
                    self.get_logger().info(
                        "WebSocket connection closed during object detection"
                    )
                    return False
                else:
                    self.get_logger().error(
                        f"Error detecting {target_key} ({detection_name}): {e}"
                    )
                continue

        # Check for shutdown signal
        if self.initialization_shutdown.is_set():
            self.get_logger().info("Initialization interrupted by shutdown signal")
            return False

        # Check if we have at least one successful detection
        if not successful_detections:
            self.get_logger().error(
                "No successful detections found - cannot proceed with pose estimation"
            )
            return False

        self.get_logger().info(f"Successful detections: {successful_detections}")

        # Initialize multi-object pose estimation
        self.get_logger().info("Initializing multi-object pose estimation...")
        self.get_logger().info(
            "⏳ This may take 5-10 seconds as models load and process data..."
        )

        try:
            # Use persistent connection (no context manager)
            init_response = self.pose_client.initialize_objects(objects_data)

            if not init_response.get("success", False):
                self.get_logger().error(
                    f"Multi-object pose initialization failed: "
                    f"{init_response.get('message', 'Unknown error')}"
                )
                return False

            self.get_logger().info(
                f"✅ Multi-object pose initialization successful: "
                f"{init_response.get('message', '')}"
            )

            # Extract initial poses and update tracking state
            initial_poses_7d = init_response.get("poses", {})

            with self.pose_tracking_lock:
                for target_key in successful_detections:
                    pose_name = OBJECT_NAME_MAPPING[target_key]["pose_name"]

                    if pose_name in initial_poses_7d:
                        pose_7d = initial_poses_7d[pose_name]
                        # Apply custom rotation to the initial pose
                        modified_pose_7d = self.apply_custom_rotation(
                            pose_7d, target_key
                        )

                        # Store camera coordinates for visualization
                        self.target_poses_camera[f"{target_key}_pose"] = (
                            modified_pose_7d
                        )

                        # Transform from camera coordinates to world coordinates
                        # if calibration is available
                        if self.calibration_data is not None:
                            world_pose_7d = (
                                self.calibration_data.transform_pose_camera_to_world(
                                    modified_pose_7d
                                )
                            )
                            # Store world coordinates for behavior cloning/RL models
                            self.target_poses[f"{target_key}_pose"] = world_pose_7d
                            self.get_logger().info(
                                f"Transformed {target_key} pose to world coordinates: "
                                f"camera={modified_pose_7d[:3]} -> "
                                f"world={world_pose_7d[:3]}"
                            )
                        else:
                            # No calibration available, use camera coordinates for both
                            self.target_poses[f"{target_key}_pose"] = modified_pose_7d
                            self.get_logger().warning(
                                f"No calibration available - {target_key} pose remains "
                                "in camera coordinates"
                            )

                        self.pose_tracking_initialized[target_key] = True

                        # Log initial pose (after custom rotation, in camera coordinates)
                        pose_matrix = self.pose_client.pose_7d_to_matrix(
                            modified_pose_7d
                        )
                        translation = pose_matrix[:3, 3]
                        rotation_matrix = pose_matrix[:3, :3]
                        rotation = Rotation.from_matrix(rotation_matrix)
                        euler_angles = rotation.as_euler("xyz", degrees=True)

                        self.get_logger().info(
                            f"Initial pose for {target_key} ({pose_name}) in camera coordinates:"
                        )
                        self.get_logger().info(
                            f"  Translation: [{translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f}]"
                        )
                        self.get_logger().info(
                            f"  Rotation (XYZ): [{euler_angles[0]:.1f}°, {euler_angles[1]:.1f}°, {euler_angles[2]:.1f}°]"
                        )

                        # Log custom rotation if applied
                        custom_rotation_deg = CUSTOM_ROTATIONS.get(
                            target_key, (0, 0, 0)
                        )
                        if any(abs(rot) > 1e-6 for rot in custom_rotation_deg):
                            self.get_logger().info(
                                f"  Custom rotation applied: [{custom_rotation_deg[0]:.1f}°, {custom_rotation_deg[1]:.1f}°, {custom_rotation_deg[2]:.1f}°]"
                            )
                    else:
                        self.get_logger().warning(
                            f"No initial pose returned for {target_key} ({pose_name})"
                        )

        except Exception as e:
            # Handle connection closed gracefully
            if (
                "ConnectionClosed" in str(type(e).__name__)
                or "connection" in str(e).lower()
            ):
                self.get_logger().info(
                    "WebSocket connection closed during pose initialization"
                )
                return False
            else:
                self.get_logger().error(
                    f"Error initializing multi-object pose tracking: {e}"
                )
            return False

        self.get_logger().info("Object tracking initialization completed successfully")
        return True

    def apply_custom_rotation(
        self, pose_7d: list[float], target_key: str
    ) -> list[float]:
        """Apply custom rotation to a 7D pose in the object's coordinate system.

        Args:
            pose_7d: 7D pose [x, y, z, qx, qy, qz, qw]
            target_key: Target object key ("pick_target" or "place_target")

        Returns:
            Modified 7D pose with custom rotation applied
        """
        # Apply custom position offset
        if target_key in CUSTOM_POSITION_OFFSETS:
            custom_offset = CUSTOM_POSITION_OFFSETS[target_key]
            pose_7d[0] += custom_offset[0]  # x offset
            pose_7d[1] += custom_offset[1]  # y offset
            pose_7d[2] += custom_offset[2]  # z offset

        if target_key not in CUSTOM_ROTATIONS:
            return pose_7d

        custom_rotation_deg = CUSTOM_ROTATIONS[target_key]

        # Check if any rotation is specified
        if all(abs(rot) < 1e-6 for rot in custom_rotation_deg):
            return pose_7d

        try:
            # Convert 7D pose to 4x4 transformation matrix
            translation = np.array(pose_7d[:3])
            quaternion = np.array(pose_7d[3:7])  # [qx, qy, qz, qw]

            # Get current rotation matrix
            current_rotation = Rotation.from_quat(quaternion).as_matrix()

            # Create custom rotation matrix from euler angles (in degrees)
            custom_rotation = Rotation.from_euler(
                "xyz", custom_rotation_deg, degrees=True
            ).as_matrix()

            # Apply custom rotation in object coordinate system
            # This means we multiply the current rotation by the custom rotation
            new_rotation_matrix = current_rotation @ custom_rotation

            # Convert back to quaternion
            new_quaternion = Rotation.from_matrix(new_rotation_matrix).as_quat()

            # Return modified 7D pose (position unchanged, only rotation modified)
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

    def _project_3d_to_2d(
        self, points_3d: np.ndarray, pose_matrix: np.ndarray, K: np.ndarray
    ) -> np.ndarray:
        """Project 3D points to 2D image coordinates.

        Args:
            points_3d: 3D points in object coordinate system (N, 3)
            pose_matrix: 4x4 pose transformation matrix
            K: Camera intrinsics matrix (3, 3)

        Returns:
            2D points in image coordinates (N, 2)
        """
        # Transform points to camera coordinate system
        points_3d_homo = np.concatenate(
            [points_3d, np.ones((points_3d.shape[0], 1))], axis=1
        )
        points_cam = (pose_matrix @ points_3d_homo.T).T[:, :3]

        # Project to image plane
        points_2d_homo = (K @ points_cam.T).T

        # Handle divide by zero (points behind camera or at camera plane)
        z_coords = points_2d_homo[:, 2:3]
        z_coords = np.where(
            np.abs(z_coords) < 1e-8, 1e-8, z_coords
        )  # Avoid division by zero

        points_2d = points_2d_homo[:, :2] / z_coords

        # Filter out points behind the camera (negative z in camera coordinates)
        valid_mask = points_cam[:, 2] > 0
        points_2d[~valid_mask] = [
            -1000,
            -1000,
        ]  # Move invalid points out of image bounds

        return points_2d.astype(int)

    def _visualize_pose_3d(
        self,
        image: np.ndarray,
        pose_matrix: np.ndarray,
        K: np.ndarray,
        scale: float = 0.1,
    ) -> np.ndarray:
        """Visualize 3D pose by drawing coordinate axes.

        Args:
            image: RGB image to draw on
            pose_matrix: 4x4 pose transformation matrix
            K: Camera intrinsics matrix (3, 3)
            scale: Scale factor for visualization

        Returns:
            Image with pose visualization
        """
        vis_image = image.copy()

        # Define 3D coordinate axes points (in object coordinate system)
        origin = np.array([[0, 0, 0]])
        x_axis = np.array([[scale, 0, 0]])
        y_axis = np.array([[0, scale, 0]])
        z_axis = np.array([[0, 0, scale]])

        try:
            # Project all points to 2D
            origin_2d = self._project_3d_to_2d(origin, pose_matrix, K)[0]
            x_axis_2d = self._project_3d_to_2d(x_axis, pose_matrix, K)[0]
            y_axis_2d = self._project_3d_to_2d(y_axis, pose_matrix, K)[0]
            z_axis_2d = self._project_3d_to_2d(z_axis, pose_matrix, K)[0]

            # Check if points are within image bounds
            h, w = image.shape[:2]

            def is_point_valid(pt):
                return 0 <= pt[0] < w and 0 <= pt[1] < h

            # Draw coordinate axes
            if is_point_valid(origin_2d):
                # X-axis (red)
                if is_point_valid(x_axis_2d):
                    cv2.arrowedLine(
                        vis_image, tuple(origin_2d), tuple(x_axis_2d), (0, 0, 255), 3
                    )
                    cv2.putText(
                        vis_image,
                        "X",
                        tuple(x_axis_2d + 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )

                # Y-axis (green)
                if is_point_valid(y_axis_2d):
                    cv2.arrowedLine(
                        vis_image, tuple(origin_2d), tuple(y_axis_2d), (0, 255, 0), 3
                    )
                    cv2.putText(
                        vis_image,
                        "Y",
                        tuple(y_axis_2d + 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

                # Z-axis (blue)
                if is_point_valid(z_axis_2d):
                    cv2.arrowedLine(
                        vis_image, tuple(origin_2d), tuple(z_axis_2d), (255, 0, 0), 3
                    )
                    cv2.putText(
                        vis_image,
                        "Z",
                        tuple(z_axis_2d + 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 0, 0),
                        2,
                    )

        except Exception as e:
            self.get_logger().debug(f"Failed to draw 3D pose visualization: {e}")

        return vis_image

    def create_integrated_visualization(
        self,
        rgb_image: np.ndarray,
        poses: dict[str, np.ndarray],
        confidences: dict[str, float],
        K: np.ndarray,
    ) -> np.ndarray:
        """Create integrated visualization showing pose results for tracked objects.

        Args:
            rgb_image: Original RGB image (not BGR)
            poses: Dict of 4x4 pose transformation matrices in CAMERA coordinates {object_type: pose_matrix}
            confidences: Dict of pose estimation confidences {object_type: confidence}
            K: Camera intrinsics matrix (3, 3)

        Returns:
            Visualization image in RGB format

        Note:
            This function expects poses in camera coordinates for proper 3D visualization projection.
            Use target_poses_camera (not target_poses) when calling this function.
        """
        vis_image = rgb_image.copy()

        # Color mapping for different object types
        colors = {
            "pick_target": (255, 0, 0),  # Red for pick target
            "place_target": (0, 255, 0),  # Green for place target
        }

        # Draw pose information and 3D visualization
        y_offset = 30
        for obj_type, pose_matrix in poses.items():
            confidence = confidences.get(obj_type, 0.0)
            color = colors.get(obj_type, (255, 255, 255))  # Default to white

            # Extract pose information
            translation = pose_matrix[:3, 3]
            rotation_matrix = pose_matrix[:3, :3]
            rotation = Rotation.from_matrix(rotation_matrix)
            euler_angles = rotation.as_euler("xyz", degrees=True)

            # Draw pose text info
            pose_info = [
                f"{obj_type.replace('_', ' ').title()} Pose Confidence: {confidence:.3f}",
                f"{obj_type.replace('_', ' ').title()} Translation: [{translation[0]:.2f}, {translation[1]:.2f}, {translation[2]:.2f}]",
                f"{obj_type.replace('_', ' ').title()} Rotation (XYZ): [{euler_angles[0]:.1f}°, {euler_angles[1]:.1f}°, {euler_angles[2]:.1f}°]",
            ]

            # Draw pose info on image
            for i, info in enumerate(pose_info):
                cv2.putText(
                    vis_image,
                    info,
                    (10, y_offset + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

            y_offset += len(pose_info) * 25 + 10  # Space between objects

            # Draw 3D pose visualization if pose is reasonable
            if confidence > 0.1 and np.linalg.norm(translation) < 5.0:
                vis_image = self._visualize_pose_3d(
                    vis_image, pose_matrix, K, scale=0.08
                )

        return vis_image

    def _cleanup_camera_viewer(self) -> None:
        """Clean up camera viewer resources."""
        if self.camera_viewer_enabled:
            try:
                cv2.destroyWindow(self.camera_viewer_window_name)
                cv2.destroyAllWindows()
                self.get_logger().info("Camera viewer cleanup completed")
            except Exception as e:
                self.get_logger().warning(f"Error during camera viewer cleanup: {e}")

    def _load_calibration(self) -> None:
        """Load camera calibration data from JSON file."""
        try:
            self.calibration_data = load_calibration(self.calibration_file)
            self.get_logger().info(
                f"✅ Loaded calibration from {self.calibration_file}"
            )
            self.get_logger().info(f"📊 Calibration data: {self.calibration_data}")
        except Exception as e:
            self.get_logger().error(
                f"❌ Failed to load calibration from {self.calibration_file}: {e}"
            )
            self.get_logger().warning("Object poses will remain in camera coordinates")
            self.calibration_data = None


def main(args=None):
    """Main entry point for the teleop bridge.

    Automatically detects the camera calibration file from the configs directory
    based on the ZORDI_ROBOT_ID environment variable. Falls back to looking for
    'camera_calibration.json' in the current directory if not found.
    """
    parser = argparse.ArgumentParser(description="Teleop Bridge Direct RL")
    parser.add_argument(
        "--episode-dir",
        type=str,
        default=DEFAULT_EPISODE_DIR,
        help=f"Episode directory (default: {DEFAULT_EPISODE_DIR})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging of pose details during recording",
    )
    parser.add_argument(
        "--calibration-file",
        type=str,
        default=DEFAULT_CALIBRATION_FILE,
        help="Camera calibration JSON file (default: auto-detected from configs/<ROBOT_ID>/camera_calibration.json)",
    )

    parsed_args = parser.parse_args(args)

    # Get the actual calibration file path (auto-detected or user-specified)
    calibration_file_path = get_calibration_file_path(parsed_args.calibration_file)

    rclpy.init(args=args)
    teleop_bridge = TeleopBridge(
        episode_dir=parsed_args.episode_dir,
        verbose=parsed_args.verbose,
        calibration_file=calibration_file_path,
    )
    rclpy.spin(teleop_bridge)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
