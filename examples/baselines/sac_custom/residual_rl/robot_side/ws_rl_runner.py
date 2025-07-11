#!/usr/bin/env python3
"""
Copyright 2025 Zordi, Inc. All rights reserved.

Zordi RL Runner - A ROS interface for robot control with RL server communication.
This acts as a robot controller that receives observations and executes actions.
"""

import argparse
import base64
import datetime
import json
import os
import sys
import threading
import time
from collections import deque
from enum import Enum, unique
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rclpy

# Camera calibration utilities
from calibration_utils import CalibrationData, load_calibration
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from PIL import Image as PILImage
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image, JointState
from signal_arm_msgs.msg import GripperPositionControl
from std_msgs.msg import Header
from utils import (
    adjust_k_for_crop_and_resize,
    center_crop_and_resize_image,
    colorize_depth_image,
)

# Object detection and pose estimation clients
from ws_det_client import ObjectDetectionClient  # type: ignore
from ws_pose_multi_client import MultiObjectFoundationPoseClient  # type: ignore
from ws_rl_client import ZordiRLClient  # type: ignore

# Progress estimation client depends on DirectClient
from zordi_policy_rpc.direct.client import DirectClient

# Default configuration
DEFAULT_SERVER_HOST = "localhost"
DEFAULT_SERVER_PORT = 10012
DEFAULT_PROGRESS_HOST = "yk-dev-4090"
DEFAULT_PROGRESS_PORT = 10016

DETECTION_SERVER_HOST = os.getenv("DETECTION_SERVER_HOST", "yk-dev-4090")
DETECTION_SERVER_PORT = int(os.getenv("DETECTION_SERVER_PORT", "10015"))
POSE_SERVER_HOST = os.getenv("POSE_SERVER_HOST", "yk-dev-4090")
POSE_SERVER_PORT = int(os.getenv("POSE_SERVER_PORT", "10014"))

QUEUE_SIZE = 10
UPDATE_HZ = 10

STATE_DIM = 28
ACTION_DIM = 28
DEFAULT_VELOCITY = 0.35
DEFAULT_EFFORT = 0.5
GRIPPER_POSITION_SCALE = 46

# Image sizes for different purposes
DETECTION_POSE_IMAGE_SIZE = 480
RL_POLICY_IMAGE_SIZE = 224

JPEG_QUALITY = 90  # Compression quality for transmitted images

# -----------------------------------------------------------------------------
# Rollout recording defaults
# -----------------------------------------------------------------------------
DEFAULT_RECORD_DIR = "~/galaxea_rollouts/box_pnp/"

# -----------------------------------------------------------------------------
# Camera calibration constants
# -----------------------------------------------------------------------------
DEFAULT_CALIBRATION_FILE = "camera_calibration.json"

# -----------------------------------------------------------------------------
# Right arm offset from origin (in meters)
# -----------------------------------------------------------------------------
RIGHT_ARM_OFFSET = [-0.025, -0.365, 0.005]

# J6 to tool tip offset (in meters, relative to J6 frame)
# -----------------------------------------------------------------------------
J6_TO_TOOL_TIP_OFFSET = [0.0, 0.0, 0.075]

# Object detection and pose estimation configuration
# -----------------------------------------------------------------------------
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
    "pick_target": (0, 0, 0.062),  # Z-axis offset for pick target
    "place_target": (0, 0, 0),  # No offset by default
}

PROGRESS_BONUS_SCALE = 0.002

TIME_PENALTY_PER_STEP = -0.15
DENSE_REWARD_MULTIPLIER = 0.05

GRADE_LEVEL_MAPPING = {
    0: (-100.0, False, True),  # reward, done, terminated
    1: (20.0, False, True),
    2: (40.0, False, True),
    3: (60.0, False, True),
    4: (80.0, False, True),
    5: (150.0, True, False),
    6: (250.0, True, False),
}

GRADE_LEVEL: dict[int, str] = {
    0: "Failed",
    1: "Stage 1 -- The right arm approached to the target object",
    2: "Stage 2 -- Picked up the target object with the right arm",
    3: "Stage 3 -- Moved the target object to the placement location",
    4: "Stage 4 -- Dropped the target object in the placement location",
    5: "Stage 5 -- Finished the entire task with some mistakes",
    6: "Stage 6 -- Finished the entire task smoothly without any mistakes",
}

# Phase detection constants
OBJECT_PICKED_HEIGHT_THRESHOLD = 0.08  # Object z > 0.08 means it's picked up
PHASE_LOCK_FRAME_THRESHOLD = 10  # Frames to allow place -> pick transition


def _calculate_pick_phase_rewards(
    ee_pos: np.ndarray,
    tgt_pos: np.ndarray,
    ee_x: np.ndarray,
    ee_z: np.ndarray,
    tgt_x: np.ndarray,
) -> dict[str, float]:
    """Calculate dense reward components for the pick phase.

    Args:
        ee_pos: End-effector position [x, y, z]
        tgt_pos: Target object position [x, y, z]
        ee_x: End-effector x-axis direction vector
        ee_z: End-effector z-axis direction vector
        tgt_x: Target object x-axis direction vector

    Returns:
        Dictionary containing individual reward components
    """
    # Position reward: closer to target is better
    d_pos = np.linalg.norm(ee_pos - tgt_pos)
    pos_reward = -1.0 * d_pos

    # Angle reward: aligned orientation is better
    ang = np.arccos(np.clip(np.dot(ee_x, tgt_x), -1, 1))
    ang_reward = -0.3 * ang

    # Z-alignment reward: gripper pointing downward is better
    downward = np.array([0, 0, -1])
    z_alignment = np.dot(ee_z, downward)
    z_reward = 0.05 * z_alignment

    return {
        "pos_reward": float(pos_reward),
        "ang_reward": float(ang_reward),
        "z_reward": float(z_reward),
    }


def _calculate_step_reward(
    latest_state: dict,
    extra_obs: dict,
    episode_name: str,
    task_phase: "TaskPhase",
) -> tuple[float, dict[str, float]]:
    """Calculate reward and components for a single timestep.

    Args:
        latest_state: Latest robot state from observations
        extra_obs: Extra observation data
        episode_name: Episode directory name for logging
        task_phase: The current task phase.

    Returns:
        Tuple of (total_reward, reward_components)
    """
    # Extract required data
    ee_pos = np.array(latest_state.get("right_arm_tool_pose", [])[:3])
    tgt_pos = np.array(latest_state.get("pick_target_pose", [])[:3])
    ee_x = np.array(extra_obs.get("ee_x_axis", []))
    ee_z = np.array(extra_obs.get("ee_z_axis", []))
    tgt_x = np.array(extra_obs.get("target_x_axis", []))

    # Validate data shapes
    if (
        ee_pos.shape != (3,)
        or tgt_pos.shape != (3,)
        or ee_x.shape != (3,)
        or ee_z.shape != (3,)
        or tgt_x.shape != (3,)
    ):
        print(
            f"[WARN] Missing or malformed data for reward calculation in {episode_name}"
        )
        return TIME_PENALTY_PER_STEP, {"time_penalty": TIME_PENALTY_PER_STEP}

    # Base reward components
    reward_components = {
        "time_penalty": TIME_PENALTY_PER_STEP,
        "task_phase": task_phase.value,
    }

    # Calculate phase-specific dense rewards
    if task_phase == TaskPhase.PICK:
        pick_rewards = _calculate_pick_phase_rewards(ee_pos, tgt_pos, ee_x, ee_z, tgt_x)
        reward_components.update(pick_rewards)
        reward_components["reward_multiplier"] = DENSE_REWARD_MULTIPLIER

        dense_reward = (
            pick_rewards["pos_reward"]
            + pick_rewards["ang_reward"]
            + pick_rewards["z_reward"]
        ) * DENSE_REWARD_MULTIPLIER
    else:
        # TaskPhase.PLACE: no dense rewards
        dense_reward = 0.0

    total_reward = TIME_PENALTY_PER_STEP + dense_reward
    return total_reward, reward_components


def custom_json_dumps(obj: dict, max_indent_level: int = 3) -> str:
    """Custom JSON serializer that indents only up to specified level.

    Args:
        obj: Dictionary to serialize
        max_indent_level: Maximum depth to apply indentation (default: 3)

    Returns:
        JSON string with custom indentation
    """

    def _serialize_with_level(obj, level=0, indent_size=2):
        if level >= max_indent_level:
            # Beyond max level, serialize inline without indentation
            return json.dumps(obj, separators=(",", ":"))

        if isinstance(obj, dict):
            if not obj:
                return "{}"

            indent = " " * (level * indent_size)
            next_indent = " " * ((level + 1) * indent_size)

            items = []
            for key, value in obj.items():
                key_str = json.dumps(key)
                if level + 1 >= max_indent_level:
                    # Next level should be inline
                    value_str = json.dumps(value, separators=(",", ":"))
                else:
                    value_str = _serialize_with_level(value, level + 1, indent_size)
                items.append(f"{next_indent}{key_str}: {value_str}")

            return "{\n" + ",\n".join(items) + f"\n{indent}}}"

        elif isinstance(obj, list):
            if not obj:
                return "[]"

            indent = " " * (level * indent_size)
            next_indent = " " * ((level + 1) * indent_size)

            items = []
            for item in obj:
                if level + 1 >= max_indent_level:
                    # Next level should be inline
                    item_str = json.dumps(item, separators=(",", ":"))
                else:
                    item_str = _serialize_with_level(item, level + 1, indent_size)
                items.append(f"{next_indent}{item_str}")

            return "[\n" + ",\n".join(items) + f"\n{indent}]"

        else:
            # Primitive value
            return json.dumps(obj)

    return _serialize_with_level(obj)


# -----------------------------------------------------------------------------
# Light-weight progress estimation client (copied from policy_runner_example)
# -----------------------------------------------------------------------------


class ProgressEstimationClient(DirectClient):
    """Client for progress estimation server with JPEG compression."""

    def __init__(self, host: str, port: int):
        super().__init__(host, port)

        self.metadata: Any | None = None

    def connect(self) -> None:
        """Connect to the progress estimation server."""
        super().connect()
        self.metadata = self.get_metadata()

    @staticmethod
    def _compress_image(image: np.ndarray) -> str:
        pil_image = PILImage.fromarray(image.astype(np.uint8))
        buf = BytesIO()
        pil_image.save(buf, format="JPEG", quality=90, optimize=True)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def get_progress(self, images: dict[str, np.ndarray]) -> dict:
        """Get progress from the progress estimation server."""
        comp: dict[str, str] = {k: self._compress_image(v) for k, v in images.items()}
        return self.request({"type": "get_progress", "images": comp})


@unique
class ArmType(str, Enum):
    """Enumeration of robot arm types."""

    LEFT = "left"
    RIGHT = "right"


@unique
class TaskPhase(str, Enum):
    """Enumeration of task phases for phase-aware action selection."""

    PICK = "pick"
    PLACE = "place"


def _detect_task_phase(tgt_pos: np.ndarray) -> TaskPhase:
    """Detect current task phase based on object height.

    Args:
        tgt_pos: Target object position [x, y, z]

    Returns:
        TaskPhase.PICK if object is on desk (z <= 0.07), TaskPhase.PLACE if picked up (z > 0.07)
    """
    return (
        TaskPhase.PLACE
        if tgt_pos[2] > OBJECT_PICKED_HEIGHT_THRESHOLD
        else TaskPhase.PICK
    )


class ZordiRLRunner(Node):
    """
    ROS interface for robot control with RL server communication.

    This node handles ROS topics for robot sensors and actuators, provides
    policy inference via WebSocket server communication.

    Args:
        node_name: Name of the ROS node
        server_host: RL server host
        server_port: RL server port
        *,
        record: bool = False,
        record_dir: str = DEFAULT_RECORD_DIR,
        max_frames: int = 250,
        progress_host: str = DEFAULT_PROGRESS_HOST,
        progress_port: int = DEFAULT_PROGRESS_PORT,
        inference_mode: bool = False,
        use_progress: bool = False,
        enable_object_tracking: bool = False,
        use_phase_based_action_selection: bool = True,
        detection_host: str = DETECTION_SERVER_HOST,
        detection_port: int = DETECTION_SERVER_PORT,
        pose_host: str = POSE_SERVER_HOST,
        pose_port: int = POSE_SERVER_PORT,
        calibration_file: str = DEFAULT_CALIBRATION_FILE,
    """

    def __init__(
        self,
        node_name: str = "zordi_rl_runner",
        server_host: str = DEFAULT_SERVER_HOST,
        server_port: int = DEFAULT_SERVER_PORT,
        *,
        record: bool = False,
        record_dir: str = DEFAULT_RECORD_DIR,
        max_frames: int = 250,
        progress_host: str = DEFAULT_PROGRESS_HOST,
        progress_port: int = DEFAULT_PROGRESS_PORT,
        inference_mode: bool = False,
        use_progress: bool = False,
        enable_object_tracking: bool = False,
        use_phase_based_action_selection: bool = True,
        detection_host: str = DETECTION_SERVER_HOST,
        detection_port: int = DETECTION_SERVER_PORT,
        pose_host: str = POSE_SERVER_HOST,
        pose_port: int = POSE_SERVER_PORT,
        calibration_file: str = DEFAULT_CALIBRATION_FILE,
    ) -> None:
        super().__init__(node_name)

        # Server connection parameters
        self.server_host = server_host
        self.server_port = server_port
        self.inference_mode = inference_mode
        self.use_phase_based_action_selection = use_phase_based_action_selection

        # Object tracking parameters
        self.enable_object_tracking = enable_object_tracking
        self.detection_host = detection_host
        self.detection_port = detection_port
        self.pose_host = pose_host
        self.pose_port = pose_port
        self.calibration_file = calibration_file

        # Camera calibration data
        self.calibration_data: CalibrationData | None = None
        if self.enable_object_tracking:
            self._load_calibration()

        # RL client for all server communication
        self.rl_client = ZordiRLClient(
            self.server_host,
            self.server_port,
            client_id=f"ros_client_{self.get_name()}",
        )

        # Field definitions - will be populated after connecting to server
        self.state_fields: dict[str, list[int]] = {}
        self.action_fields: dict[str, list[int]] = {}

        # Camera topics - matching expected names from server configuration
        self.camera_topics: dict[str, str] = {
            "static_top_rgb": "/camera/static_rs405_top/color/image_rect_raw",
            "static_top_depth": "/camera/static_rs405_top/depth/image_rect_raw",
            "eoat_left_top_rgb": "/camera/eoat_rs405_left_top/color/image_rect_raw",
            "eoat_right_top_rgb": "/camera/eoat_rs405_right_top/color/image_rect_raw",
        }

        # Initialize image storage
        self.images: dict[str, np.ndarray | None] = {}
        self.subscribers = []

        # Create camera subscribers
        for name, topic in self.camera_topics.items():
            self.images[name] = None
            self.subscribers.append(
                self.create_subscription(
                    Image, topic, self._create_callback_for_camera(name), QUEUE_SIZE
                )
            )
            self.get_logger().info(f"Subscribed to camera topic: {topic} as {name}")

        # Camera info topics for RGB cameras only (derived from image topics)
        self.camera_info_topics: dict[str, str] = {}
        self.camera_info_subscribers = []
        for name, topic in self.camera_topics.items():
            if "rgb" in name:  # Only RGB cameras, not depth
                # Replace image_rect_raw with camera_info
                info_topic = topic.replace("image_rect_raw", "camera_info")
                self.camera_info_topics[name] = info_topic

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

        # Initialize camera intrinsics storage
        self.camera_k_matrices: dict[
            str, list[float]
        ] = {}  # For pose estimation (480x480)
        self.camera_original_k_matrices: dict[str, list[float]] = {}
        self.camera_image_dimensions: dict[str, tuple[int, int]] = {}

        # WebSocket clients for object detection and pose estimation
        self.detection_client: ObjectDetectionClient | None = None
        self.pose_client: MultiObjectFoundationPoseClient | None = None

        # Add a new flag to track object tracking initialization status
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
        self.object_tracking_initialization_attempted = (
            False  # New flag to track if we've tried initialization
        )

        # Pose tracking thread management (similar to teleop_bridge)
        self.pose_tracking_lock = threading.Lock()  # For thread-safe pose updates
        self.pose_tracking_thread: threading.Thread | None = None
        self.pose_tracking_in_progress = threading.Event()  # Prediction in progress
        self.pose_tracking_trigger = threading.Event()  # Trigger for new prediction
        self.pose_tracking_shutdown = threading.Event()  # Shutdown signal

        # WebSocket context managers for persistent connections
        self.pose_client_context = None
        self.detection_client_context = None

        # Pose tracking state (simplified, no threading)
        self.target_poses: dict[str, list[float]] = {
            "pick_target_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "place_target_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        }
        self.pose_tracking_initialized: dict[str, bool] = {
            "pick_target": False,
            "place_target": False,
        }
        self.object_tracking_ready: bool = (
            False  # Flag indicating if object tracking is fully ready
        )

        # Joint state storage for right arm only
        self.last_msg_set: JointState = JointState()
        self.last_ee_pose_set: PoseStamped | None = None

        # Joint state subscribers and publishers for right arm only
        self.joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ]  # 7 joints for right arm
        self.frame_id = "world"

        # Right arm only
        self.subscription_follower_r = self.create_subscription(
            JointState,
            "/joint_states_follower_r",
            self.joint_state_callback,
            QUEUE_SIZE,
        )
        self.publisher_follower_r = self.create_publisher(
            JointState, "/arm_joint_target_position_follower_r", QUEUE_SIZE
        )
        self.publisher_follower_g1_gripper_r = self.create_publisher(
            GripperPositionControl, "/gripper_position_control_follower_r", QUEUE_SIZE
        )

        # End-effector pose subscriber for right arm
        self.subscription_ee_pose_r = self.create_subscription(
            PoseStamped,
            "/ee_pose_follower_r",
            self.ee_pose_callback,
            QUEUE_SIZE,
        )

        # For converting ROS images to numpy arrays
        self.bridge = CvBridge()

        # Create policy timer
        self.policy_timer = self.create_timer(1 / UPDATE_HZ, self.generate_actions)

        # Set up logging
        self.get_logger().info("ZordiRLRunner initialized")
        self.get_logger().info(f"Server: {self.server_host}:{self.server_port}")

        # Initialize WebSocket clients for object detection and pose estimation
        self._init_websocket_clients()

        # ------------------------------------------------------------ rollout recording
        self.record_enabled: bool = record
        self.base_record_dir = Path(os.path.expanduser(record_dir)).resolve()
        self.max_frames: int = max_frames
        self.model_id: str | None = None  # Filled after handshake
        self.n_obs_steps: int = 0
        self.horizon: int = 0
        self.n_action_steps: int = 0
        self.state_history: deque[dict[str, Any]] | None = None  # Changed to dict
        self.rollout_dir: Path | None = None
        self.frame_index: int = 0
        # Observation / reward logging
        self.observations: list[dict] = []
        # Rollout termination flag
        self.max_frames_reached: bool = False

        # Phase tracking state
        self.current_phase = TaskPhase.PICK
        self.place_phase_start_frame: int | None = None

        # Progress estimation support
        self.progress_client: ProgressEstimationClient | None = None
        self.latest_progress: float | None = None

        # Episode timing (for elapsed_ms in rollout)
        self.episode_start_time: float | None = None

        if use_progress and progress_host and progress_port:
            try:
                self.progress_client = ProgressEstimationClient(
                    progress_host, progress_port
                )
                self.progress_client.connect()
                self.get_logger().info(
                    f"Connected to progress server at {progress_host}:{progress_port}"
                )
            except Exception as e:
                self.get_logger().warning(
                    f"Failed to connect to progress server at "
                    f"{progress_host}:{progress_port}: {e}"
                )
                self.get_logger().info(
                    "Continuing without progress estimation "
                    "(reward computation will use base penalty only)"
                )
                self.progress_client = None

    def _init_websocket_clients(self) -> None:
        """Initialize WebSocket clients for object detection and pose estimation."""
        if not self.enable_object_tracking:
            self.get_logger().info(
                "Object tracking disabled - skipping WebSocket client initialization"
            )
            self.detection_client = None
            self.pose_client = None
            return

        try:
            # Initialize detection client
            self.detection_client = ObjectDetectionClient(
                self.detection_host, self.detection_port
            )
            self.get_logger().info(
                f"Detection client initialized ({self.detection_host}:{self.detection_port})"
            )

            # Initialize pose estimation client
            self.pose_client = MultiObjectFoundationPoseClient(
                self.pose_host, self.pose_port
            )
            self.get_logger().info(
                f"Pose client initialized ({self.pose_host}:{self.pose_port})"
            )

        except Exception as e:
            self.get_logger().error(f"Failed to initialize WebSocket clients: {e}")
            self.detection_client = None
            self.pose_client = None

    def _load_calibration(self) -> None:
        """Load camera calibration data from JSON file."""
        try:
            self.calibration_data = load_calibration(self.calibration_file)
            self.get_logger().info(f"✅ Loaded calibration: {self.calibration_data}")
        except Exception as e:
            self.get_logger().error(
                f"❌ Failed to load calibration from {self.calibration_file}: {e}"
            )
            self.get_logger().warning("Object poses will remain in camera coordinates")
            self.calibration_data = None

    def _enter_websocket_contexts(self) -> bool:
        """Enter WebSocket context managers for persistent connections.

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

    def _create_callback_for_camera(self, camera_name: str):
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
            else:
                # RGB images: convert from ROS BGR format to RGB for storage
                cv_image_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                cv_image = cv2.cvtColor(cv_image_bgr, cv2.COLOR_BGR2RGB)

            # Center crop to 480x480 for object detection and pose estimation
            # Original images are 640x480, center crop to 480x480 (no resize)
            # Note: Only pose estimation uses camera K matrices,
            # object detection doesn't need them

            cv_image = center_crop_and_resize_image(
                cv_image,
                image_size=(DETECTION_POSE_IMAGE_SIZE, DETECTION_POSE_IMAGE_SIZE),
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

                # Compute adjusted K matrix for pose estimation (480x480 center crop)
                # Convert K matrix from list to numpy array for processing
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
                # (flattened back to list)
                self.camera_k_matrices[camera_name] = adjusted_k.flatten().tolist()

            else:
                self.get_logger().warning(
                    f"Invalid K matrix size for {camera_name}: expected 9, "
                    f"got {len(msg.k)}"
                )
        except Exception as e:
            self.get_logger().error(f"Error processing camera info {camera_name}: {e}")

    def joint_state_callback(self, joint_state_msg: JointState) -> None:
        """Callback for joint states."""
        self.last_msg_set = joint_state_msg

    def _apply_j6_to_tool_tip_offset(self, pose_msg: PoseStamped) -> PoseStamped:
        """Apply J6 to tool tip offset and transform from robot to world coordinates."""
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

    def ee_pose_callback(self, pose_msg: PoseStamped) -> None:
        """Callback for end-effector pose updates."""
        self.last_ee_pose_set = self._apply_j6_to_tool_tip_offset(pose_msg)

    def get_header(self) -> Header:
        """Get a header with the current time and frame id."""
        return Header(stamp=self.get_clock().now().to_msg(), frame_id=self.frame_id)

    def get_observation(self) -> dict[str, Any] | None:
        """
        Get current observation with structured state dictionaries.

        Returns:
            Observation dictionary with structured state_sequence if complete,
            None otherwise
        """
        # Ensure we have a history buffer (set after server handshake)
        if self.state_history is None:
            return None

        # Check if all images are available
        if any(img is None for img in self.images.values()):
            self.get_logger().info("Waiting for all images to be available...")
            return None

        right_joint_state = self.last_msg_set
        right_ee_pose = self.last_ee_pose_set

        # Check if joint states and end-effector poses are available
        if not right_joint_state.position or right_ee_pose is None:
            self.get_logger().info("Waiting for joint states and end-effector poses...")
            return None

        # Build observation dictionary
        obs = {}

        for name, img in self.images.items():
            if img is None:
                self.get_logger().info(f"Image {name} is None")
                return None
            obs[name] = img

        # Build structured robot state dictionary (fields that correspond to actions)
        robot_state_dict = {}

        # Right arm joints (7 dimensions)
        robot_state_dict["right_arm_joints"] = list(right_joint_state.position)

        # Right tool pose (7 dimensions)
        ee_pos = np.array([
            right_ee_pose.pose.position.x,
            right_ee_pose.pose.position.y,
            right_ee_pose.pose.position.z,
        ])
        ee_quat_raw = [
            right_ee_pose.pose.orientation.x,
            right_ee_pose.pose.orientation.y,
            right_ee_pose.pose.orientation.z,
            right_ee_pose.pose.orientation.w,
        ]

        # Apply quaternion normalization
        # if ee_quat_raw[0] > 0:
        if ee_quat_raw[-1] < 0:
            ee_quat_raw = [-x for x in ee_quat_raw]

        ee_quat = np.array(ee_quat_raw)
        robot_state_dict["right_arm_tool_pose"] = np.concatenate([
            ee_pos,
            ee_quat,
        ]).tolist()

        # Get target poses from pose tracking
        with self.pose_tracking_lock:
            pick_target_pose_7d = self.target_poses["pick_target_pose"].copy()
            place_target_pose_7d = self.target_poses["place_target_pose"].copy()
        robot_state_dict["pick_target_pose"] = pick_target_pose_7d
        robot_state_dict["place_target_pose"] = place_target_pose_7d

        # --------------------------------------------------------------------
        # --- Build extra observation features for reward computation ---
        # --------------------------------------------------------------------
        # Extract positions and quaternions
        tgt_pos = np.array(pick_target_pose_7d[:3])
        tgt_quat = np.array(pick_target_pose_7d[3:])  # Stored as x,y,z,w

        # Convert quaternions to rotation matrices
        ee_rot_matrix = Rotation.from_quat(ee_quat).as_matrix()
        tgt_rot_matrix = Rotation.from_quat(tgt_quat).as_matrix()

        # Extract axis vectors
        ee_x = ee_rot_matrix[:, 0].copy()
        ee_y = ee_rot_matrix[:, 1].copy()
        ee_z = ee_rot_matrix[:, 2].copy()
        tgt_x_raw = tgt_rot_matrix[:, 0].copy()
        tgt_y = tgt_rot_matrix[:, 1].copy()

        # World-align target x-axis (same as in sim env)
        world_x = np.array([1, 0, 0])
        tgt_x = tgt_x_raw if np.dot(tgt_x_raw, world_x) >= 0 else -tgt_x_raw

        # Normalize axes (for safety)
        ee_x /= np.linalg.norm(ee_x) + 1e-8
        ee_y /= np.linalg.norm(ee_y) + 1e-8
        ee_z /= np.linalg.norm(ee_z) + 1e-8
        tgt_x /= np.linalg.norm(tgt_x) + 1e-8
        tgt_y /= np.linalg.norm(tgt_y) + 1e-8

        # Calculate deltas
        delta_xyz = tgt_pos - ee_pos
        delta_angle_x = np.arccos(np.clip(np.dot(ee_x, tgt_x), -1, 1))
        downward = np.array([0, 0, -1])
        delta_angle_z = np.arccos(np.clip(np.dot(ee_z, downward), -1, 1))

        # Build extra observation dictionary (only fields used for reward computation)
        extra_obs = {
            "ee_x_axis": ee_x.tolist(),
            "ee_z_axis": ee_z.tolist(),
            "target_x_axis": tgt_x.tolist(),
        }

        # Add only robot state to state history
        self.state_history.append(robot_state_dict)
        if len(self.state_history) < self.n_obs_steps:
            self.get_logger().info(
                f"Waiting for state history to be full "
                f"(current length: {len(self.state_history)})"
            )
            return None  # Wait until history buffer is full

        # Convert state history to list for sending
        obs["state_sequence"] = list(self.state_history)
        obs["extra_obs"] = extra_obs

        return obs

    def build_observation_dict(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Build observation dictionary for RL client."""
        obs_dict: dict[str, Any] = {}

        # ------------------------------------------------------------------ images
        # Images are stored at 480x480 (center crop from 640x480) for
        # object detection/pose estimation
        # Resize them to 224x224 for RL policy (only RL policy needs this smaller size)
        for cam_name, img in observation.items():
            if cam_name in self.images and img.ndim == 3:
                # Resize from 480x480 to 224x224 for RL policy
                rl_policy_img = center_crop_and_resize_image(
                    img, image_size=(RL_POLICY_IMAGE_SIZE, RL_POLICY_IMAGE_SIZE)
                )

                success, enc = cv2.imencode(
                    ".jpg",
                    rl_policy_img[:, :, ::-1],  # Convert RGB to BGR for encoding
                    [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
                )
                if not success:
                    continue
                obs_dict[cam_name] = enc.tobytes()

        # State sequence (list of dictionaries)
        state_seq = observation.get("state_sequence", [])
        if state_seq is not None:
            obs_dict["state_sequence"] = state_seq

        # Extra observations (for SAC model)
        extra_obs = observation.get("extra_obs", {})
        if extra_obs:
            obs_dict["extra_obs"] = extra_obs

        return obs_dict

    def publish_joint_states(
        self,
        action_dict: dict[str, Any],
        velocity: float = DEFAULT_VELOCITY,
        effort: float = DEFAULT_EFFORT,
    ) -> None:
        """
        Publish joint commands to right arm using structured action dictionary.

        Args:
            action_dict: Dictionary with action fields (right_arm_joints, etc.)
            velocity: Joint velocity
            effort: Joint effort
        """
        # Extract right arm joint actions
        right_arm_joints = action_dict.get("right_arm_joints", [])
        if not right_arm_joints:
            self.get_logger().error("No right_arm_joints found in action dictionary")
            return

        if len(right_arm_joints) < 7:
            self.get_logger().error(
                f"Insufficient right arm joint actions: {len(right_arm_joints)}, "
                f"expected 7"
            )
            return

        # Publish right arm joint states
        right_joint_msg = JointState()
        right_joint_msg.header = self.get_header()
        right_joint_msg.name = self.joint_names
        right_joint_msg.position = right_arm_joints[:7]  # First 7 joint positions
        right_joint_msg.velocity = [velocity] * len(self.joint_names)
        right_joint_msg.effort = [effort] * len(self.joint_names)
        self.publisher_follower_r.publish(right_joint_msg)

        # Publish gripper state (use the 7th joint for gripper)
        new_gripper_state = GripperPositionControl()
        new_gripper_state.header = self.get_header()
        new_gripper_state.gripper_stroke = (
            (abs(right_arm_joints[6]) - 0.05) * 0.9 * GRIPPER_POSITION_SCALE
        )
        self.publisher_follower_g1_gripper_r.publish(new_gripper_state)

    def connect(self) -> None:
        """Connect to RL server."""
        try:
            self.rl_client.connect()  # type: ignore[union-attr]
            self.get_logger().info("Successfully connected to RL server.")

            # Retrieve metadata for temporal parameters and field definitions
            meta = self.rl_client.get_metadata()  # type: ignore[call-arg]
            svc_meta = (
                meta.get("service_metadata", {}) if isinstance(meta, dict) else {}
            )
            self.n_obs_steps = int(svc_meta.get("n_obs_steps", 1))
            self.horizon = int(svc_meta.get("horizon", 1))
            self.n_action_steps = int(svc_meta.get("n_action_steps", 1))

            # Extract field definitions from server metadata
            self.state_fields = svc_meta.get("state_fields", {})
            self.action_fields = svc_meta.get("action_fields", {})

            self.get_logger().info(f"State fields: {self.state_fields}")
            self.get_logger().info(f"Action fields: {self.action_fields}")

            # Use dictionaries for state history instead of numpy arrays
            self.state_history = deque(maxlen=self.n_obs_steps)
            self.get_logger().info(
                f"Using n_obs_steps={self.n_obs_steps} from server metadata"
            )

            # Extract model_id for recording directory
            self.model_id = str(svc_meta.get("model_id", "unknown_model"))

            # Prepare rollout directory if recording is enabled
            if self.record_enabled:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                model_dir = self.base_record_dir / self.model_id
                self.rollout_dir = model_dir / timestamp
                self.rollout_dir.mkdir(parents=True, exist_ok=True)
                self.get_logger().info(
                    f"Recording enabled. Saving rollout to: {self.rollout_dir}"
                )
                self.get_logger().info(
                    f"Maximum frames limit: {self.max_frames} "
                    "(rollout will auto-terminate when reached)"
                )

            # Object tracking will be initialized later in generate_actions()
            # when camera data is available
            if self.enable_object_tracking:
                self.get_logger().info(
                    "🔄 Object tracking enabled - will initialize "
                    "when camera data is ready"
                )
            else:
                self.get_logger().info(
                    "Object tracking disabled - using placeholder poses"
                )

        except Exception as e:
            self.get_logger().error(f"Failed to connect to RL server: {e}")
            self.rl_client = None
            raise

    def close(self) -> None:
        """Close the connection to the RL server."""
        if self.rl_client:
            if rclpy.ok():
                self.get_logger().info("Closing connection to RL server.")

            # Stop object tracking first
            self.stop_object_tracking()

            self.rl_client.close()
            self.rl_client = None

            # Finalize rollout saving if needed
            self._finalize_rollout()

    def check_topic_connectivity(self) -> None:
        """Check and log the status of topic subscriptions."""
        self.get_logger().info("📡 Checking topic connectivity:")

        # Check camera topics
        for name, topic in self.camera_topics.items():
            self.get_logger().info(f"  {name}: {topic}")

        # Check camera info topics
        for name, topic in self.camera_info_topics.items():
            self.get_logger().info(f"  {name}_info: {topic}")

        # Show current image status
        self.get_logger().info("📸 Current image status:")
        for name in self.images:
            status = "✓" if self.images[name] is not None else "✗"
            shape = (
                f" (shape: {self.images[name].shape})"
                if self.images[name] is not None
                else ""
            )
            self.get_logger().info(f"  {name}: {status}{shape}")

        # Show current K matrix status
        self.get_logger().info("📐 Current K matrix status:")
        for name in self.camera_k_matrices:
            status = "✓" if name in self.camera_k_matrices else "✗"
            self.get_logger().info(f"  {name}: {status}")

    def _initialize_object_tracking(self) -> bool:
        """Initialize object detection and pose tracking.

        Returns:
            True if initialization successful, False otherwise
        """
        # Check topic connectivity first
        self.check_topic_connectivity()

        # Quick ROS topics check
        try:
            topic_names_and_types = self.get_topic_names_and_types()
            camera_topics_found = []
            for topic_name, topic_types in topic_names_and_types:
                if any(
                    camera_topic in topic_name
                    for camera_topic in self.camera_topics.values()
                ):
                    camera_topics_found.append(topic_name)

            self.get_logger().info(
                f"🔍 Found {len(camera_topics_found)} relevant camera topics in ROS network"
            )
            for topic in camera_topics_found:
                self.get_logger().info(f"  Found: {topic}")
        except Exception as e:
            self.get_logger().warning(f"Could not check ROS topics: {e}")

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

        # Get static camera images and intrinsics (should already be available)
        static_rgb = self.images.get("static_top_rgb")
        static_depth = self.images.get("static_top_depth")
        static_k = self.camera_k_matrices.get("static_top_rgb")

        # Double-check that camera data is available
        # (should not fail if called from generate_actions)
        if static_rgb is None or static_depth is None or static_k is None:
            self.get_logger().error(
                f"Camera data not available: "
                f"RGB={'✓' if static_rgb is not None else '✗'}, "
                f"Depth={'✓' if static_depth is not None else '✗'}, "
                f"K matrix={'✓' if static_k is not None else '✗'}"
            )
            return False

        self.get_logger().info("✅ All required camera data confirmed available!")

        camera_intrinsics = np.array(static_k).reshape(3, 3)

        self.get_logger().info(
            "Performing object detection for pick and place targets..."
        )

        # Detect both objects and collect their data
        objects_data = []
        successful_detections = []

        for target_key, names in OBJECT_NAME_MAPPING.items():
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

            for target_key in successful_detections:
                pose_name = OBJECT_NAME_MAPPING[target_key]["pose_name"]

                if pose_name in initial_poses_7d:
                    pose_7d = initial_poses_7d[pose_name]
                    # Apply custom rotation to the initial pose
                    modified_pose_7d = self.apply_custom_rotation(pose_7d, target_key)

                    # Transform from camera coordinates to world coordinates
                    # if calibration is available
                    if self.calibration_data is not None:
                        world_pose_7d = (
                            self.calibration_data.transform_pose_camera_to_world(
                                modified_pose_7d
                            )
                        )
                        self.target_poses[f"{target_key}_pose"] = world_pose_7d
                        self.get_logger().info(
                            f"Transformed {target_key} pose to world coordinates: "
                            f"camera={modified_pose_7d[:3]} -> "
                            f"world={world_pose_7d[:3]}"
                        )
                    else:
                        # No calibration available, use camera coordinates directly
                        self.target_poses[f"{target_key}_pose"] = modified_pose_7d
                        self.get_logger().warning(
                            f"No calibration available - {target_key} pose remains "
                            "in camera coordinates"
                        )

                    self.pose_tracking_initialized[target_key] = True

                    # Log initial pose (after custom rotation)
                    pose_matrix = self.pose_client.pose_7d_to_matrix(modified_pose_7d)
                    translation = pose_matrix[:3, 3]
                    rotation_matrix = pose_matrix[:3, :3]
                    rotation = Rotation.from_matrix(rotation_matrix)
                    euler_angles = rotation.as_euler("xyz", degrees=True)

                    self.get_logger().info(
                        f"Initial pose for {target_key} ({pose_name}):"
                    )
                    self.get_logger().info(
                        f"  Translation: "
                        f"[{translation[0]:.3f}, {translation[1]:.3f}, "
                        f"{translation[2]:.3f}]"
                    )
                    self.get_logger().info(
                        f"  Rotation (XYZ): "
                        f"[{euler_angles[0]:.1f}°, {euler_angles[1]:.1f}°, "
                        f"{euler_angles[2]:.1f}°]"
                    )

                    # Log custom rotation if applied
                    custom_rotation_deg = CUSTOM_ROTATIONS.get(target_key, (0, 0, 0))
                    if any(abs(rot) > 1e-6 for rot in custom_rotation_deg):
                        self.get_logger().info(
                            f"  Custom rotation applied: "
                            f"[{custom_rotation_deg[0]:.1f}°, "
                            f"{custom_rotation_deg[1]:.1f}°, "
                            f"{custom_rotation_deg[2]:.1f}°]"
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

    def _pose_tracking_worker_thread(self) -> None:
        """Worker thread that handles pose tracking predictions."""
        while not self.pose_tracking_shutdown.is_set():
            try:
                # Wait for trigger or shutdown with minimal timeout for fast response
                if self.pose_tracking_trigger.wait(timeout=0.001):  # 1ms timeout
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
                # Make sure to clear the in progress flag on error
                self.pose_tracking_in_progress.clear()

    def _perform_pose_tracking(self) -> None:
        """Perform the actual pose tracking prediction (synchronous)."""
        # Check if object tracking is ready
        if not self.object_tracking_ready:
            return

        # Sample latest images directly from self.images
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
            # Use persistent connection for pose tracking
            track_response = self.pose_client.predict_poses(
                static_rgb, static_depth, None, camera_intrinsics
            )

            # Extract tracking results
            tracked_poses_7d = track_response.get("poses", {})
            tracked_confidences = track_response.get("confidences", {})

            # Update pose tracking state for each object
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

                            # Transform from camera coordinates to world coordinates if calibration is available
                            if self.calibration_data is not None:
                                world_pose_7d = self.calibration_data.transform_pose_camera_to_world(
                                    modified_pose_7d
                                )
                                self.target_poses[f"{target_key}_pose"] = world_pose_7d
                            else:
                                # No calibration available, use camera coordinates directly
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

    def track_poses(self) -> None:
        """Trigger pose tracking directly (runs at 50Hz)."""
        if not self.object_tracking_ready:
            return

        # Check if any objects are being tracked
        objects_being_tracked = [
            target_key
            for target_key in ["pick_target", "place_target"]
            if self.pose_tracking_initialized.get(target_key, False)
        ]
        if not objects_being_tracked:
            return

        # If a prediction is already in progress, skip this frame
        if self.pose_tracking_in_progress.is_set():
            return

        # Trigger pose tracking thread
        self.pose_tracking_trigger.set()

    def start_object_tracking(self) -> None:
        """Start object detection and pose tracking (synchronous)."""
        if not self.enable_object_tracking:
            self.get_logger().info("Object tracking disabled - skipping start")
            return

        if self.detection_client is None or self.pose_client is None:
            self.get_logger().error("WebSocket clients not initialized")
            return

        # Enter WebSocket contexts for persistent connections
        self.get_logger().info("🔗 Connecting to WebSocket servers...")
        if not self._enter_websocket_contexts():
            self.get_logger().error("❌ Failed to connect to WebSocket servers")
            return

        # Initialize object tracking synchronously
        self.get_logger().info("🔄 Initializing object detection and pose tracking...")
        success = self._initialize_object_tracking()

        if success:
            self.object_tracking_ready = True
            self.get_logger().info("✅ Object tracking initialized successfully")

            # Create pose tracking timer to run at 50Hz
            self.pose_tracking_timer = self.create_timer(
                1.0 / POSE_TRACKING_HZ, self.track_poses
            )

            # Start pose tracking worker thread
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
        else:
            self.get_logger().error("❌ Object tracking initialization failed")
            self._exit_websocket_contexts()

    def stop_object_tracking(self) -> None:
        """Stop object tracking and clean up resources."""
        if not self.enable_object_tracking:
            self.get_logger().debug("Object tracking was not enabled - skipping stop")
            return

        if rclpy.ok():
            self.get_logger().info("Stopping object tracking...")

        # Mark object tracking as not ready
        self.object_tracking_ready = False

        # Stop pose tracking timer
        if hasattr(self, "pose_tracking_timer"):
            self.pose_tracking_timer.cancel()

        # Shutdown pose tracking thread
        self.pose_tracking_shutdown.set()
        self.pose_tracking_trigger.set()  # Unblock the thread
        if self.pose_tracking_thread and self.pose_tracking_thread.is_alive():
            self.pose_tracking_thread.join(timeout=1.0)
            if self.pose_tracking_thread.is_alive():
                if rclpy.ok():
                    self.get_logger().warning(
                        "Pose tracking thread did not shut down gracefully"
                    )

        # Reset pose tracking state
        self.pose_tracking_initialized = {
            "pick_target": False,
            "place_target": False,
        }
        self.target_poses = {
            "pick_target_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "place_target_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        }

        # Exit WebSocket contexts
        self._exit_websocket_contexts()

        if rclpy.ok():
            self.get_logger().info("Object tracking stopped")

    def _report_object_tracking_status(self) -> None:
        """Report the current status of object tracking for debugging."""
        if not self.enable_object_tracking:
            self.get_logger().info("📋 Object tracking status: DISABLED")
            return

        self.get_logger().info("📋 Object tracking status:")
        self.get_logger().info(f"  Ready: {self.object_tracking_ready}")
        self.get_logger().info(
            f"  Detection client: {'✓' if self.detection_client else '✗'}"
        )
        self.get_logger().info(f"  Pose client: {'✓' if self.pose_client else '✗'}")
        self.get_logger().info(
            f"  Detection context: {'✓' if self.detection_client_context else '✗'}"
        )
        self.get_logger().info(
            f"  Pose context: {'✓' if self.pose_client_context else '✗'}"
        )

        for target_key in ["pick_target", "place_target"]:
            initialized = self.pose_tracking_initialized.get(target_key, False)
            pose = self.target_poses.get(f"{target_key}_pose", [0, 0, 0, 0, 0, 0, 1])
            is_nonzero = any(
                abs(x) > 1e-6 for x in pose[:3]
            )  # Check if position is non-zero
            self.get_logger().info(
                f"  {target_key}: {'✓' if initialized else '✗'} initialized, "
                f"{'✓' if is_nonzero else '✗'} non-zero pose"
            )

    def generate_actions(self) -> None:
        """Generate actions using the RL client."""
        # Skip if policy client is not initialized
        if not self.rl_client:
            self.get_logger().info("RL client not initialized yet...")
            return

        # Check if we've reached the maximum frame limit and should terminate
        if self.max_frames_reached:
            self.get_logger().info("Maximum frames reached. Shutting down...")
            self.shutdown()
            # Request ROS shutdown to exit gracefully
            rclpy.shutdown()
            return

        # Initialize object tracking if enabled and not yet attempted
        if (
            self.enable_object_tracking
            and not self.object_tracking_initialization_attempted
            and not self.object_tracking_ready
        ):
            # Check if all required camera data is available
            static_rgb = self.images.get("static_top_rgb")
            static_depth = self.images.get("static_top_depth")
            static_k = self.camera_k_matrices.get("static_top_rgb")

            if (
                static_rgb is not None
                and static_depth is not None
                and static_k is not None
            ):
                # Mark as attempted to prevent multiple initialization attempts
                self.object_tracking_initialization_attempted = True

                self.get_logger().info(
                    "📸 Camera data ready! Starting object tracking initialization..."
                )
                self.get_logger().info(
                    f"  RGB shape: {static_rgb.shape}, "
                    f"Depth shape: {static_depth.shape}, "
                    f"K matrix length: {len(static_k)}"
                )

                # Start object tracking initialization
                self.start_object_tracking()

                if self.object_tracking_ready:
                    self.get_logger().info(
                        "✅ Object tracking ready - RL policy can start"
                    )
                    self._report_object_tracking_status()
                else:
                    self.get_logger().error(
                        "❌ Object tracking failed - using placeholder poses"
                    )
                    self._report_object_tracking_status()
            else:
                # Camera data not ready yet
                missing_data = []
                if static_rgb is None:
                    missing_data.append("RGB")
                if static_depth is None:
                    missing_data.append("Depth")
                if static_k is None:
                    missing_data.append("K matrix")

                self.get_logger().debug(
                    f"Waiting for camera data: missing {missing_data}"
                )
                return

        # Skip if object tracking is enabled but not ready
        if self.enable_object_tracking and not self.object_tracking_ready:
            self.get_logger().debug("Waiting for object tracking to be ready...")
            return

        # Get current observation
        obs = self.get_observation()
        if obs is None:
            self.get_logger().info("Waiting for complete observation...")
            return

        try:
            # Build observation payload for RL client
            observation_payload = self.build_observation_dict(obs)

            # Log state dimensions for debugging
            state_seq = observation_payload.get("state_sequence", [])

            # Get actions dict from server
            actions = self.rl_client.predict(observation_payload)  # type: ignore[union-attr]
            if actions is None:
                self.get_logger().warning(
                    "No action received from server (server returned error)"
                )
                return

            if (
                "expert_action" not in actions
                or "residual_action" not in actions
                or "final_action" not in actions
            ):
                self.get_logger().warning("Incomplete action received from server")
                self.get_logger().warning(
                    "Required fields: expert_action, residual_action, final_action"
                )
                return

            # Detect current task phase for action selection
            current_state = state_seq[-1] if state_seq else {}
            current_extra_obs = obs.get("extra_obs", {})

            # Phase-based action selection can be disabled via CLI flag
            if (
                self.use_phase_based_action_selection
                and current_state
                and "pick_target_pose" in current_state
            ):
                pick_target_pos = np.array(current_state["pick_target_pose"][:3])
                raw_phase = _detect_task_phase(pick_target_pos)

                # Stateful phase transition logic
                if (
                    self.current_phase == TaskPhase.PICK
                    and raw_phase == TaskPhase.PLACE
                ):
                    self.current_phase = TaskPhase.PLACE
                    if self.place_phase_start_frame is None:
                        self.place_phase_start_frame = self.frame_index
                elif (
                    self.current_phase == TaskPhase.PLACE
                    and raw_phase == TaskPhase.PICK
                ):
                    if (
                        self.place_phase_start_frame is not None
                        and (self.frame_index - self.place_phase_start_frame)
                        > PHASE_LOCK_FRAME_THRESHOLD
                    ):
                        # Lock phase to "place" after threshold
                        pass
                    else:
                        # Allow transition back for failed picks
                        self.current_phase = TaskPhase.PICK
                        self.place_phase_start_frame = None

                # Phase-based action selection
                if self.current_phase == TaskPhase.PICK:
                    # Use final action (expert + residual) during pick phase
                    final_action_dict = actions["final_action"]
                    action_type = "final (expert + residual)"
                else:
                    # Use expert action only during place phase
                    final_action_dict = actions["expert_action"]
                    action_type = "expert only"

                self.get_logger().debug(
                    f"Phase: {self.current_phase.value}, Action: {action_type}"
                )
            else:
                # Fallback to final action if phase detection fails or is disabled
                final_action_dict = actions["final_action"]
                if not self.use_phase_based_action_selection:
                    action_type = "final (phase selection disabled)"
                else:
                    action_type = "final (fallback)"
                    self.get_logger().warning(
                        "Could not detect task phase, using final action"
                    )

            if not isinstance(final_action_dict, dict):
                self.get_logger().warning(
                    f"Expected action to be dict, got {type(final_action_dict)}"
                )
                return

            # Log action information for debugging
            self.get_logger().debug("Received structured action from server:")
            for field_name, field_values in final_action_dict.items():
                self.get_logger().debug(f"  {field_name}: {field_values}")

            # ---------------------------------------------------- progress estimation
            # (same as before)
            if (
                self.progress_client
                and "static_top_rgb" in self.images
                and self.images["static_top_rgb"] is not None
            ):
                try:
                    rgb_image = cv2.cvtColor(
                        self.images["static_top_rgb"], cv2.COLOR_BGR2RGB
                    )
                    progress_images = {"static_top_rgb": rgb_image}
                    progress_response = self.progress_client.get_progress(
                        progress_images
                    )

                    progress_values = progress_response.get("progress", [])
                    if isinstance(progress_values, list) and progress_values:
                        self.latest_progress = float(progress_values[0])
                    elif isinstance(progress_values, (int, float)):
                        self.latest_progress = float(progress_values)
                except Exception as exc:
                    self.get_logger().warning(f"Failed to estimate progress: {exc}")

            # Essential logging: distance to target and reward components
            current_state = state_seq[-1] if state_seq else {}
            current_extra_obs = obs.get("extra_obs", {})

            # Calculate essential metrics
            ee_pos = np.array(current_state["right_arm_tool_pose"][:3])
            pick_target_pos = np.array(current_state["pick_target_pose"][:3])
            distance_to_target = np.linalg.norm(pick_target_pos - ee_pos)
            delta_pos = pick_target_pos - ee_pos

            # Calculate reward components (same as in _save_step)
            # Use modular reward calculation with the stateful phase
            _, reward_components = _calculate_step_reward(
                current_state, current_extra_obs, "live_run", self.current_phase
            )
            angle_error_rad = np.arccos(
                np.clip(
                    np.dot(
                        np.array(current_extra_obs["ee_x_axis"]),
                        np.array(current_extra_obs["target_x_axis"]),
                    ),
                    -1,
                    1,
                )
            )
            angle_error = np.degrees(angle_error_rad)
            downward = np.array([0, 0, -1])
            z_alignment = np.dot(np.array(current_extra_obs["ee_z_axis"]), downward)

            # Print essential info including phase
            print("================================")
            print(f"Distance to target: {distance_to_target:.4f} m")
            print(
                f"Delta (x,y,z): "
                f"({delta_pos[0]:.4f}, {delta_pos[1]:.4f}, {delta_pos[2]:.4f})"
            )
            print(f"Angle error (X-axis): {angle_error:.2f}°")
            print(f"Z-alignment: {z_alignment:.3f}")

            # Show current phase and action type
            if current_state and "pick_target_pose" in current_state:
                print(f"Task phase: {self.current_phase.value}")
                print(f"Action type: {action_type}")

            # Print action summary
            print("Action:")
            for field_name, field_values in final_action_dict.items():
                if isinstance(field_values, list):
                    print(
                        f"  {field_name}: ["
                        + ", ".join(f"{v:.3f}" for v in field_values)
                        + "]"
                    )
                else:
                    print(f"  {field_name}: {field_values}")
            print("================================")

            # Publish joint commands
            if not self.inference_mode:
                self.publish_joint_states(final_action_dict)

            # -------------------------------------------------------- save rollout step
            if self.record_enabled and self.rollout_dir is not None:
                self._save_step(actions)

        except Exception as e:
            self.get_logger().error(f"Error during action generation: {e}")

    def shutdown(self) -> None:
        """Clean up resources when shutting down."""
        # Stop object tracking first
        self.stop_object_tracking()

        if self.rl_client:
            self.close()

    # ------------------------------------------------------------------ rollout helpers

    def _save_step(self, actions: dict[str, Any]) -> None:
        """Save observation + reward structure."""
        assert self.rollout_dir is not None

        # Get current observation (includes extra_obs)
        current_obs = self.get_observation()
        if current_obs is None:
            self.get_logger().warning(
                "Cannot save step - no current observation available"
            )
            return

        # ------------------------- store camera images
        image_paths: dict[str, str] = {}
        for cam_name, img in self.images.items():
            if img is None:
                continue
            cam_dir = self.rollout_dir / cam_name
            cam_dir.mkdir(exist_ok=True)
            fname = f"{self.frame_index:06d}.jpg"
            fpath = cam_dir / fname

            # Handle different image types properly
            if "depth" in cam_name:
                resized_depth = center_crop_and_resize_image(
                    img, image_size=(RL_POLICY_IMAGE_SIZE, RL_POLICY_IMAGE_SIZE)
                )
                colorized_depth = colorize_depth_image(
                    resized_depth, min_depth=0.05, max_depth=2.0
                )
                cv2.imwrite(str(fpath), colorized_depth)
            else:
                resized_rgb = center_crop_and_resize_image(
                    img, image_size=(RL_POLICY_IMAGE_SIZE, RL_POLICY_IMAGE_SIZE)
                )
                bgr_img = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(fpath), bgr_img)

            image_paths[cam_name] = f"{cam_name}/{fname}"

        # ------------------------- reward computation
        if self.state_history is None:
            self.get_logger().warning("State history is None, cannot compute reward")
            return

        states_history = list(self.state_history)
        latest_state = states_history[-1]
        extra_obs = current_obs["extra_obs"]

        # Use modular reward calculation
        total_reward, reward_components = _calculate_step_reward(
            latest_state,
            extra_obs,
            self.rollout_dir.name if self.rollout_dir else "unknown",
            self.current_phase,
        )

        # ------------------------ episode timing
        if self.episode_start_time is None:
            self.episode_start_time = time.time()
        elapsed_ms: float = (time.time() - self.episode_start_time) * 1000.0

        # ------------------------- observation dict
        observation: dict = {
            "frame_index": self.frame_index,
            "states": states_history,
            "expert_action": actions["expert_action"],
            "residual_action": actions["residual_action"],
            "executed_action": actions["final_action"],
            "next_state": {},
            "extra_obs": extra_obs,  # Add extra_obs to observation frame
            "reward": total_reward,
            "reward_components": reward_components,
            "done": False,
            "terminated": False,
            "elapsed_ms": elapsed_ms,
        }

        # Conditionally add progress estimation
        if self.latest_progress is not None:
            observation["progress_estimation"] = self.latest_progress

        observation.update(image_paths)

        # Link with previous step for next_state -- use only robot state (most recent)
        if self.observations:
            self.observations[-1]["next_state"] = states_history[-1]

        self.observations.append(observation)
        self.frame_index += 1

        # Check if we've reached the maximum frame limit
        if self.record_enabled and self.frame_index >= self.max_frames:
            self.get_logger().info(
                f"Reached maximum frame limit ({self.max_frames}). "
                "Terminating rollout..."
            )
            self.max_frames_reached = True

    def _finalize_rollout(self) -> None:
        """Write rollout data to JSON file on shutdown."""
        if not self.record_enabled or self.rollout_dir is None:
            return

        # Clean incomplete observations
        complete_obs = [o for o in self.observations if o.get("next_state")]
        self.observations = complete_obs

        # Log rollout completion reason
        if self.max_frames_reached:
            self.get_logger().info(
                f"Rollout completed due to max frames limit ({self.max_frames}). "
                f"Total frames recorded: {len(self.observations)}"
            )
        else:
            self.get_logger().info(
                f"Rollout completed. Total frames recorded: {len(self.observations)}"
            )

        # Ask for grade
        feedback = ""  # Store feedback for saving
        try:
            print("\n" + "=" * 60)
            print("TRAJECTORY GRADING LEVELS:")
            print("=" * 60)
            for g, desc in GRADE_LEVEL.items():
                print(f"  {g}: {desc}")
            print("=" * 60)
            feedback = input(
                f"\nPlease rate the executed trajectory (0-{len(GRADE_LEVEL) - 1}): "
            ).strip()
            self._apply_final_grade(feedback)
        except Exception:  # pragma: no cover - non-interactive environments
            pass

        # ------------------------- metadata
        metadata = {
            "episode_name": self.rollout_dir.name,
            "policy_type": "rl",
            "model_id": self.model_id,
            "total_frames": len(self.observations),
            "max_frames": self.max_frames,
            "max_frames_reached": self.max_frames_reached,
            "horizon": self.horizon,
            "n_action_steps": self.n_action_steps,
            "n_obs_steps": self.n_obs_steps,
            "state_fields": self.state_fields,
            "action_fields": self.action_fields,
            "grade_level_mapping": {str(k): v for k, v in GRADE_LEVEL.items()},
            "grade_reward_mapping": {str(k): v for k, v in GRADE_LEVEL_MAPPING.items()},
            "shape_meta": {
                "obs": {
                    "state": {"fields": self.state_fields},
                    "images": {
                        cam: {
                            "dtype": "rgb",
                            "image_size": [
                                RL_POLICY_IMAGE_SIZE,
                                RL_POLICY_IMAGE_SIZE,
                            ],
                        }
                        for cam in self.camera_topics
                    },
                },
                "action": {"fields": self.action_fields},
            },
        }

        episode_data = {"metadata": metadata, "observations": self.observations}

        out_path = self.rollout_dir / "observations.json"
        out_path.write_text(custom_json_dumps(episode_data, max_indent_level=4))

        self.get_logger().info(f"✅ Episode saved: {out_path}")

        # Save feedback to separate file for compatibility
        if feedback:
            feedback_path = self.rollout_dir / "inference_grade.json"
            feedback_path.write_text(custom_json_dumps({"feedback": feedback}))

            self.get_logger().info(f"✅ Feedback saved: {feedback_path}")

    # ------------------------------------------------------------------ grading helper

    def _apply_final_grade(self, grade: str) -> None:
        if not self.observations:
            return
        try:
            gnum = int(grade)
            reward, done, terminated = GRADE_LEVEL_MAPPING.get(
                gnum, GRADE_LEVEL_MAPPING[0]
            )
        except ValueError:
            reward, done, terminated = GRADE_LEVEL_MAPPING[0]

        reward = float(reward)
        done = bool(done)
        terminated = bool(terminated)

        self.observations[-1]["reward"] = reward
        self.observations[-1]["reward_components"] = {"grade_reward": reward}
        self.observations[-1]["done"] = done
        self.observations[-1]["terminated"] = terminated


# print_vector function removed - now using structured dictionaries


def main() -> int:  # noqa: D103
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Zordi RL Runner - ROS Interface")
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record rollout data to disk.",
    )
    parser.add_argument(
        "--record-dir",
        type=str,
        default=DEFAULT_RECORD_DIR,
        help=(
            "Base directory where rollout data will be saved "
            f"(default: {DEFAULT_RECORD_DIR})"
        ),
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=250,
        help="Maximum number of frames to record before automatic termination (default: 250)",
    )
    parser.add_argument(
        "--progress-host",
        type=str,
        default=DEFAULT_PROGRESS_HOST,
        help="Progress estimation server host",
    )
    parser.add_argument(
        "--progress-port",
        type=int,
        default=DEFAULT_PROGRESS_PORT,
        help="Progress estimation server port",
    )
    parser.add_argument(
        "--use-progress",
        action="store_true",
        help="Enable progress estimation (requires progress server to be running)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_SERVER_HOST,
        help=f"RL server host (default: {DEFAULT_SERVER_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_SERVER_PORT,
        help=f"RL server port (default: {DEFAULT_SERVER_PORT})",
    )
    parser.add_argument(
        "--inference-mode",
        action="store_true",
        help="Run in inference mode (no joint commands)",
    )
    parser.add_argument(
        "--disable-phase-based-action-selection",
        action="store_true",
        help="Disable phase-based action selection (always use final action).",
    )
    parser.add_argument(
        "--disable-object-tracking",
        action="store_true",
        help="Disable object detection and pose tracking for pick/place targets",
    )
    parser.add_argument(
        "--detection-host",
        type=str,
        default=DETECTION_SERVER_HOST,
        help=f"Object detection server host (default: {DETECTION_SERVER_HOST})",
    )
    parser.add_argument(
        "--detection-port",
        type=int,
        default=DETECTION_SERVER_PORT,
        help=f"Object detection server port (default: {DETECTION_SERVER_PORT})",
    )
    parser.add_argument(
        "--pose-host",
        type=str,
        default=POSE_SERVER_HOST,
        help=f"Pose estimation server host (default: {POSE_SERVER_HOST})",
    )
    parser.add_argument(
        "--pose-port",
        type=int,
        default=POSE_SERVER_PORT,
        help=f"Pose estimation server port (default: {POSE_SERVER_PORT})",
    )
    parser.add_argument(
        "--calibration-file",
        type=str,
        default=DEFAULT_CALIBRATION_FILE,
        help=f"Camera calibration JSON file (default: {DEFAULT_CALIBRATION_FILE})",
    )

    args = parser.parse_args()

    # Initialize ROS
    rclpy.init()

    # Create the ZordiRLRunner node
    node = ZordiRLRunner(
        server_host=args.host,
        server_port=args.port,
        record=args.record,
        record_dir=args.record_dir,
        max_frames=args.max_frames,
        progress_host=args.progress_host,
        progress_port=args.progress_port,
        inference_mode=args.inference_mode,
        use_progress=args.use_progress,
        enable_object_tracking=not args.disable_object_tracking,
        use_phase_based_action_selection=not args.disable_phase_based_action_selection,
        detection_host=args.detection_host,
        detection_port=args.detection_port,
        pose_host=args.pose_host,
        pose_port=args.pose_port,
        calibration_file=args.calibration_file,
    )

    try:
        # Connect to RL server
        node.connect()

        # Spin to handle ROS callbacks and generate actions
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
