"""
Copyright 2025 Zordi, Inc. All rights reserved.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import sapien
import torch

from mani_skill.envs.tasks.tabletop.pick_box import PickBoxEnv
from mani_skill.examples.motionplanning.a1_galaxea.motionplanner import (
    A1GalaxeaMotionPlanningSolver,
)
from mani_skill.examples.motionplanning.a1_galaxea.solutions.utils import (
    colorize_depth_image,
)
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb,
    get_actor_obb,
)
from mani_skill.utils.structs import Actor, Link

# Right arm offset from table origin (in meters)
RIGHT_ARM_OFFSET = np.array([-0.025, -0.365, 0.005])
# Left arm offset from table origin (in meters)
LEFT_ARM_OFFSET = np.array([-0.025, 0.365, 0.005])


class A1GalaxeaDatasetGenerator(A1GalaxeaMotionPlanningSolver):
    """A1 motion planner that generates training dataset in the proper format."""

    def __init__(
        self,
        env,
        output_root_dir="~/training_data/galaxea_box_pnp_sim/",
        save_video=True,
        video_fps=30,
        use_table_origin=True,
        **kwargs,
    ):
        super().__init__(env, **kwargs)
        self.output_root_dir = Path(output_root_dir)
        self.observations: List[Dict[str, Any]] = []
        self.episode_name = ""
        self.episode_dir = None
        self.frame_counter = 0
        self.start_time = None
        self.last_timestamp = None
        self.segmentation_mapping = {}
        self.color_mapping = {}
        self.id_remapping = {}  # Maps original seg_id to simplified 1-5 range
        self.use_table_origin = use_table_origin
        self.table_origin = None  # Will be computed when needed

        # Video recording settings
        self.save_video = save_video
        self.video_fps = video_fps
        self.video_frames = []  # Store frames for default camera video
        self.static_top_video_frames = []  # Store frames for static top camera video
        self.eoat_video_frames = []  # Store frames for end effector camera video
        # Separate buffers for each gripper camera
        self.eoat_left_video_frames = []
        self.eoat_right_video_frames = []
        # Depth and segmentation video buffers
        self.static_top_depth_video_frames: list[np.ndarray] = []
        self.static_top_seg_video_frames: list[np.ndarray] = []
        self.eoat_left_depth_video_frames: list[np.ndarray] = []
        self.eoat_left_seg_video_frames: list[np.ndarray] = []
        self.eoat_right_depth_video_frames: list[np.ndarray] = []
        self.eoat_right_seg_video_frames: list[np.ndarray] = []

    def _compute_table_origin(self) -> np.ndarray:
        """Compute table origin position based on robot arm base positions."""
        if self.table_origin is not None:
            return self.table_origin

        # Get robot arm base positions
        if self.is_bimanual:
            left_robot = self.base_env.agent.agents[0].robot
            right_robot = self.base_env.agent.agents[1].robot

            left_base_pos = left_robot.pose.p.cpu().numpy().flatten()
            right_base_pos = right_robot.pose.p.cpu().numpy().flatten()
        else:
            # Single arm mode - use right arm
            right_robot = self.robot
            right_base_pos = right_robot.pose.p.cpu().numpy().flatten()

            # For single arm, estimate left arm position using bimanual config offsets
            cfg = (
                self.env.unwrapped.pick_box_configs
                if hasattr(self.env.unwrapped, "pick_box_configs")
                else None
            )
            if cfg and "left_arm" in cfg:
                left_base_pos = cfg["left_arm"]["pose"].p
            else:
                # Fallback: mirror right arm position
                left_base_pos = right_base_pos.copy()
                left_base_pos[1] = -left_base_pos[1]  # Mirror Y coordinate

        # Table origin is midpoint between arms minus the offsets
        right_arm_world_pos = right_base_pos
        left_arm_world_pos = left_base_pos if self.is_bimanual else left_base_pos

        # Calculate table origin such that:
        # right_arm_world_pos = table_origin + RIGHT_ARM_OFFSET
        # left_arm_world_pos = table_origin + LEFT_ARM_OFFSET
        # Take average to get best estimate
        table_origin_from_right = right_arm_world_pos - RIGHT_ARM_OFFSET
        table_origin_from_left = left_arm_world_pos - LEFT_ARM_OFFSET

        self.table_origin = (table_origin_from_right + table_origin_from_left) / 2

        print(f"🔧 Table origin computed: {self.table_origin}")
        print(f"🔧 Right arm world pos: {right_arm_world_pos}")
        print(f"🔧 Left arm world pos: {left_arm_world_pos}")
        print(
            f"🔧 Expected right arm table-relative: {self.table_origin + RIGHT_ARM_OFFSET}"
        )
        print(
            f"🔧 Expected left arm table-relative: {self.table_origin + LEFT_ARM_OFFSET}"
        )

        return self.table_origin

    def _transform_pose_to_table_origin(self, pose: np.ndarray) -> np.ndarray:
        """Transform a pose from world coordinates to table_origin coordinates.

        Args:
            pose: 7-element array [x, y, z, qx, qy, qz, qw] in world coordinates

        Returns:
            7-element array [x, y, z, qx, qy, qz, qw] in table_origin coordinates
        """
        if not self.use_table_origin:
            return pose

        table_origin = self._compute_table_origin()

        # Transform position relative to table origin
        transformed_pose = pose.copy()
        transformed_pose[:3] = pose[:3] - table_origin
        # Orientation (quaternion) remains the same

        return transformed_pose

    def start_episode(self, episode_name: str = None):
        """Start a new episode and create directory structure."""
        if episode_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            episode_name = timestamp

        self.episode_name = episode_name
        self.observations = []
        self.frame_counter = 0
        self.start_time = datetime.now()
        self.last_timestamp = None
        self.video_frames = []  # Reset video frames for new episode
        self.static_top_video_frames = []  # Reset static top video frames
        self.eoat_video_frames = []  # Reset end effector video frames
        self.eoat_left_video_frames = []  # Reset EOAT videos
        self.eoat_right_video_frames = []
        self.static_top_depth_video_frames = []
        self.static_top_seg_video_frames = []
        self.eoat_left_depth_video_frames = []
        self.eoat_left_seg_video_frames = []
        self.eoat_right_depth_video_frames = []
        self.eoat_right_seg_video_frames = []

        # Create episode directory structure
        date_str = self.start_time.strftime("%Y/%m/%d")
        self.episode_dir = (
            self.output_root_dir
            / "galaxea_box_pnp"
            / "galaxea-16hz_box_pnp"
            / "data"
            / date_str
            / episode_name
        )

        # Create camera directories
        (self.episode_dir / "static_top_rgb").mkdir(parents=True, exist_ok=True)
        (self.episode_dir / "static_top_depth").mkdir(parents=True, exist_ok=True)
        (self.episode_dir / "static_top_segmentation").mkdir(
            parents=True, exist_ok=True
        )
        (self.episode_dir / "static_top_segmentation_raw").mkdir(
            parents=True, exist_ok=True
        )
        (self.episode_dir / "eoat_left_top_rgb").mkdir(parents=True, exist_ok=True)
        (self.episode_dir / "eoat_right_top_rgb").mkdir(parents=True, exist_ok=True)
        (self.episode_dir / "eoat_right_top_depth").mkdir(parents=True, exist_ok=True)
        (self.episode_dir / "eoat_right_top_segmentation").mkdir(
            parents=True, exist_ok=True
        )
        (self.episode_dir / "eoat_right_top_segmentation_raw").mkdir(
            parents=True, exist_ok=True
        )
        # Additional directories for left end effector camera (bimanual support)
        (self.episode_dir / "eoat_left_top_depth").mkdir(parents=True, exist_ok=True)
        (self.episode_dir / "eoat_left_top_segmentation").mkdir(
            parents=True, exist_ok=True
        )
        (self.episode_dir / "eoat_left_top_segmentation_raw").mkdir(
            parents=True, exist_ok=True
        )

        # Create video directory if video recording is enabled
        if self.save_video:
            (self.episode_dir / "default_camera_video").mkdir(
                parents=True, exist_ok=True
            )

        # Extract segmentation mapping from environment
        self._extract_segmentation_mapping()

        print(f"Started episode: {episode_name}")
        print(f"Episode directory: {self.episode_dir}")
        print(
            f"Segmentation mapping extracted: {len(self.segmentation_mapping)} entities"
        )
        if self.save_video:
            print("Video recording enabled - frames will be saved from default camera")

    def _extract_segmentation_mapping(self):
        """Extract segmentation ID to entity mapping from environment."""
        # Get segmentation mapping from environment
        seg_id_map = self.env.unwrapped.segmentation_id_map

        # Clear previous mappings
        self.segmentation_mapping = {}
        self.color_mapping = {}
        self.id_remapping = {}

        # Define predetermined colors for key objects (BGR format)
        predefined_colors = {
            "b5box": [0, 0, 255],  # Red for the box to pick
            "basket": [0, 255, 0],  # Green for the target basket
            "gripper": [255, 0, 0],  # Blue for robot gripper
            "table": [128, 128, 128],  # Gray for table/surface
            "background": [0, 0, 0],  # Black for background
        }

        # Process each segmentation ID
        for seg_id, entity in seg_id_map.items():
            if seg_id == 0:
                # Background
                entity_info = {
                    "type": "background",
                    "name": "background",
                    "description": "Background/distant objects",
                }
                self.color_mapping[seg_id] = predefined_colors["background"]
            elif isinstance(entity, Actor):
                entity_info = {
                    "type": "actor",
                    "name": entity.name,
                    "description": f"Actor: {entity.name}",
                }
                # Assign predetermined colors based on name patterns
                if (
                    "box" in entity.name.lower()
                    or "b5box" in entity.name.lower()
                    or "cube" in entity.name.lower()
                ):
                    self.color_mapping[seg_id] = predefined_colors["b5box"]
                elif "basket" in entity.name.lower():
                    self.color_mapping[seg_id] = predefined_colors["basket"]
                elif "table" in entity.name.lower() or "ground" in entity.name.lower():
                    self.color_mapping[seg_id] = predefined_colors["table"]
                else:
                    # Use hash-based color for other actors
                    np.random.seed(int(seg_id))
                    self.color_mapping[seg_id] = np.random.randint(50, 255, 3).tolist()
            elif isinstance(entity, Link):
                entity_info = {
                    "type": "link",
                    "name": entity.name,
                    "description": f"Link: {entity.name}",
                }
                # Assign predetermined colors based on name patterns
                if "gripper" in entity.name.lower() or "finger" in entity.name.lower():
                    self.color_mapping[seg_id] = predefined_colors["gripper"]
                elif "table" in entity.name.lower() or "ground" in entity.name.lower():
                    self.color_mapping[seg_id] = predefined_colors["table"]
                else:
                    # Use hash-based color for other links
                    np.random.seed(
                        int(seg_id + 1000)
                    )  # Offset to avoid collision with actors
                    self.color_mapping[seg_id] = np.random.randint(50, 255, 3).tolist()
            else:
                entity_info = {
                    "type": "unknown",
                    "name": str(entity),
                    "description": f"Unknown entity: {entity}",
                }
                # Use hash-based color for unknown entities
                np.random.seed(int(seg_id + 2000))
                self.color_mapping[seg_id] = np.random.randint(50, 255, 3).tolist()

            self.segmentation_mapping[int(seg_id)] = entity_info

        # Create simplified ID remapping (1-5 for key objects, 0 for background)
        self._create_id_remapping()

        # Print mapping for debugging
        print("Segmentation mapping:")
        for seg_id, info in sorted(self.segmentation_mapping.items()):
            color = self.color_mapping[seg_id]
            simple_id = self.id_remapping.get(seg_id, 0)
            print(
                f"  ID {seg_id}: {info['name']} ({info['type']}) -> Color: {color}, Simple ID: {simple_id}"
            )

    def _create_id_remapping(self):
        """Create simplified ID remapping (1-5 for key objects, 0 for background)."""
        # Define priority order for key objects
        priority_patterns = [
            ("cube", "box", "b5box"),  # ID 1: Object to pick
            ("basket",),  # ID 2: Target container
            ("gripper", "finger"),  # ID 3: Robot gripper
            ("table", "ground"),  # ID 4: Surface
            ("goal",),  # ID 5: Goal/target markers
        ]

        # Initialize remapping with background
        self.id_remapping = {0: 0}  # Background stays 0

        # Assign simplified IDs based on priority
        next_simple_id = 1

        for patterns in priority_patterns:
            if next_simple_id > 5:
                break

            # Find entities matching current pattern
            for seg_id, entity_info in self.segmentation_mapping.items():
                if seg_id == 0:  # Skip background
                    continue

                entity_name = entity_info["name"].lower()

                # Check if entity matches any pattern
                for pattern in patterns:
                    if pattern in entity_name:
                        if seg_id not in self.id_remapping:
                            self.id_remapping[seg_id] = next_simple_id
                        break

            next_simple_id += 1

        # All other entities get ID 0 (background)
        for seg_id in self.segmentation_mapping.keys():
            if seg_id not in self.id_remapping:
                self.id_remapping[seg_id] = 0

    def _get_tooltip_pose(self) -> np.ndarray:
        """Get the tooltip 3D pose (position + quaternion in xyzw order).

        Returns:
            numpy array of shape (7,) containing [x, y, z, qx, qy, qz, qw]
        """
        # Find the tool_tip link
        tool_tip_link = None
        for link in self.robot.get_links():
            if link.get_name() == "tool_tip":
                tool_tip_link = link
                break

        if tool_tip_link is None:
            raise ValueError("Could not find 'tool_tip' link on the robot")

        # Get the pose of the tool tip link
        pose = tool_tip_link.pose

        # Extract position and quaternion
        position = pose.p  # [x, y, z]
        quaternion = pose.q  # [w, x, y, z] in SAPIEN format

        # Convert to numpy arrays
        if hasattr(position, "cpu"):
            position = position.cpu().numpy()
        if hasattr(quaternion, "cpu"):
            quaternion = quaternion.cpu().numpy()

        # Flatten arrays if needed
        if position.ndim > 1:
            position = position.flatten()
        if quaternion.ndim > 1:
            quaternion = quaternion.flatten()

        # SAPIEN uses [w, x, y, z] format, but we want [x, y, z, w] format
        # Convert from [w, x, y, z] to [x, y, z, w]
        quaternion_xyzw = np.array([
            quaternion[1],
            quaternion[2],
            quaternion[3],
            quaternion[0],
        ])

        # Combine position and quaternion: [x, y, z, qx, qy, qz, qw]
        tooltip_pose = np.concatenate([position, quaternion_xyzw])

        # Transform to table origin coordinates
        return self._transform_pose_to_table_origin(tooltip_pose)

    def _get_pick_target_pose(self) -> np.ndarray:
        """Get the pick target (b5box) 3D pose (position + quaternion in xyzw order).

        Returns:
            numpy array of shape (7,) containing [x, y, z, qx, qy, qz, qw]
        """
        # Get the b5box pose from the environment
        b5box_pose = self.env.unwrapped.b5box.pose

        # Extract position and quaternion
        position = b5box_pose.p  # [x, y, z]
        quaternion = b5box_pose.q  # [w, x, y, z] in SAPIEN format

        # Convert to numpy arrays
        if hasattr(position, "cpu"):
            position = position.cpu().numpy()
        if hasattr(quaternion, "cpu"):
            quaternion = quaternion.cpu().numpy()

        # Flatten arrays if needed
        if position.ndim > 1:
            position = position.flatten()
        if quaternion.ndim > 1:
            quaternion = quaternion.flatten()

        # SAPIEN uses [w, x, y, z] format, but we want [x, y, z, w] format
        # Convert from [w, x, y, z] to [x, y, z, w]
        quaternion_xyzw = np.array([
            quaternion[1],
            quaternion[2],
            quaternion[3],
            quaternion[0],
        ])

        # Combine position and quaternion: [x, y, z, qx, qy, qz, qw]
        pick_target_pose = np.concatenate([position, quaternion_xyzw])

        # Transform to table origin coordinates
        return self._transform_pose_to_table_origin(pick_target_pose)

    def _get_place_target_pose(self) -> np.ndarray:
        """Get the place target (basket) 3D pose (position + quaternion in xyzw order).

        Returns:
            numpy array of shape (7,) containing [x, y, z, qx, qy, qz, qw]
        """
        # Get the basket pose from the environment
        basket_pose = self.env.unwrapped.basket.pose

        # Extract position and quaternion
        position = basket_pose.p  # [x, y, z]
        quaternion = basket_pose.q  # [w, x, y, z] in SAPIEN format

        # Convert to numpy arrays
        if hasattr(position, "cpu"):
            position = position.cpu().numpy()
        if hasattr(quaternion, "cpu"):
            quaternion = quaternion.cpu().numpy()

        # Flatten arrays if needed
        if position.ndim > 1:
            position = position.flatten()
        if quaternion.ndim > 1:
            quaternion = quaternion.flatten()

        # SAPIEN uses [w, x, y, z] format, but we want [x, y, z, w] format
        # Convert from [w, x, y, z] to [x, y, z, w]
        quaternion_xyzw = np.array([
            quaternion[1],
            quaternion[2],
            quaternion[3],
            quaternion[0],
        ])

        # Combine position and quaternion: [x, y, z, qx, qy, qz, qw]
        place_target_pose = np.concatenate([position, quaternion_xyzw])

        # Transform to table origin coordinates
        return self._transform_pose_to_table_origin(place_target_pose)

    def _get_tool_pose(self, robot) -> np.ndarray:
        """Get the tool pose for a given robot (position + quaternion in xyzw order).

        Args:
            robot: Robot object to get tool pose from

        Returns:
            numpy array of shape (7,) containing [x, y, z, qx, qy, qz, qw]
        """
        # Find the tool_tip link for this robot
        tool_tip_link = None
        for link in robot.get_links():
            if link.get_name() == "tool_tip":
                tool_tip_link = link
                break

        if tool_tip_link is None:
            raise ValueError("Could not find 'tool_tip' link on the robot")

        # Get the pose of the tool tip link
        pose = tool_tip_link.pose

        # Extract position and quaternion
        position = pose.p  # [x, y, z]
        quaternion = pose.q  # [w, x, y, z] in SAPIEN format

        # Convert to numpy arrays
        if hasattr(position, "cpu"):
            position = position.cpu().numpy()
        if hasattr(quaternion, "cpu"):
            quaternion = quaternion.cpu().numpy()

        # Flatten arrays if needed
        if position.ndim > 1:
            position = position.flatten()
        if quaternion.ndim > 1:
            quaternion = quaternion.flatten()

        # SAPIEN uses [w, x, y, z] format, but we want [x, y, z, w] format
        # Convert from [w, x, y, z] to [x, y, z, w]
        quaternion_xyzw = np.array([
            quaternion[1],
            quaternion[2],
            quaternion[3],
            quaternion[0],
        ])

        # Combine position and quaternion: [x, y, z, qx, qy, qz, qw]
        tool_pose = np.concatenate([position, quaternion_xyzw])

        # Transform to table origin coordinates
        return self._transform_pose_to_table_origin(tool_pose)

    def save_frame_data(self, action: np.ndarray = None, phase: str = ""):
        """Save frame data including images and observations."""
        if self.episode_dir is None:
            raise ValueError("Episode not started. Call start_episode() first.")

        # Get current observations
        obs = self.env.get_obs()

        # Calculate timing
        current_time = datetime.now()
        elapsed_ms = (current_time - self.start_time).total_seconds() * 1000

        # Calculate delta_ms from last frame
        if self.last_timestamp is None:
            delta_ms = 0.0
        else:
            delta_ms = (current_time - self.last_timestamp).total_seconds() * 1000
        self.last_timestamp = current_time

        # Get robot states based on bimanual vs single arm mode
        if self.is_bimanual:
            # Get left arm joint states (agent 0)
            left_robot = self.base_env.agent.agents[0].robot
            joint_states_left = (
                left_robot.get_qpos()[0, :7].cpu().numpy()
            )  # 6 arm joints + 1 gripper

            # Get right arm joint states (agent 1)
            right_robot = self.base_env.agent.agents[1].robot
            joint_states_right = (
                right_robot.get_qpos()[0, :7].cpu().numpy()
            )  # 6 arm joints + 1 gripper

            # Get tool poses for both arms
            tool_pose_left = self._get_tool_pose(left_robot)
            tool_pose_right = self._get_tool_pose(right_robot)

        else:
            # Single arm mode - use zeros for left arm, actual values for right arm
            joint_states_left = np.zeros(7)
            joint_states_right = self.robot.get_qpos()[0, :7].cpu().numpy()

            # Get tool poses
            tool_pose_left = np.zeros(7)
            tool_pose_right = self._get_tool_pose(self.robot)

        # Get pick target pose (b5box)
        pick_target_pose = self._get_pick_target_pose()

        # Get place target pose (basket)
        place_target_pose = self._get_place_target_pose()

        # Debug: Print poses for first few frames
        if len(self.observations) < 3:
            coord_system = "table_origin" if self.use_table_origin else "world"
            print(
                f"🔍 [POSE DEBUG - {coord_system}] Tool pose left: pos=[{tool_pose_left[0]:.3f}, {tool_pose_left[1]:.3f}, {tool_pose_left[2]:.3f}], "
                f"quat=[{tool_pose_left[3]:.3f}, {tool_pose_left[4]:.3f}, {tool_pose_left[5]:.3f}, {tool_pose_left[6]:.3f}]"
            )
            print(
                f"🔍 [POSE DEBUG - {coord_system}] Tool pose right: pos=[{tool_pose_right[0]:.3f}, {tool_pose_right[1]:.3f}, {tool_pose_right[2]:.3f}], "
                f"quat=[{tool_pose_right[3]:.3f}, {tool_pose_right[4]:.3f}, {tool_pose_right[5]:.3f}, {tool_pose_right[6]:.3f}]"
            )
            print(
                f"🔍 [POSE DEBUG - {coord_system}] Pick target pose: pos=[{pick_target_pose[0]:.3f}, {pick_target_pose[1]:.3f}, {pick_target_pose[2]:.3f}], "
                f"quat=[{pick_target_pose[3]:.3f}, {pick_target_pose[4]:.3f}, {pick_target_pose[5]:.3f}, {pick_target_pose[6]:.3f}]"
            )
            print(
                f"🔍 [POSE DEBUG - {coord_system}] Place target pose: pos=[{place_target_pose[0]:.3f}, {place_target_pose[1]:.3f}, {place_target_pose[2]:.3f}], "
                f"quat=[{place_target_pose[3]:.3f}, {place_target_pose[4]:.3f}, {place_target_pose[5]:.3f}, {place_target_pose[6]:.3f}]"
            )

            if self.use_table_origin:
                table_origin = self._compute_table_origin()
                print(f"🔍 [COORD DEBUG] Table origin: {table_origin}")
                print(f"🔍 [COORD DEBUG] Right arm should be at: {RIGHT_ARM_OFFSET}")
                print(f"🔍 [COORD DEBUG] Left arm should be at: {LEFT_ARM_OFFSET}")

        # Frame numbering (6-digit format)
        frame_str = f"{self.frame_counter:06d}"

        # Save camera images
        image_paths = {}

        # Create observation entry in the new format
        observation = {
            "frame_index": len(self.observations),
            "delta_ms": delta_ms,
            "elapsed_ms": elapsed_ms,
            "joint_states_left": joint_states_left.tolist(),
            "joint_states_right": joint_states_right.tolist(),
            "tool_pose_left": tool_pose_left.tolist(),
            "tool_pose_right": tool_pose_right.tolist(),
            "pick_target_pose": pick_target_pose.tolist(),
            "place_target_pose": place_target_pose.tolist(),
        }

        if "sensor_data" in obs:
            sensor_data = obs["sensor_data"]

            # Debug: Print available cameras (only for first few frames)
            if len(self.observations) < 3:
                print(f"Available cameras: {list(sensor_data.keys())}")
                for cam_name, cam_data in sensor_data.items():
                    print(f"  {cam_name}: {list(cam_data.keys())}")

            # Save static top RGB
            if "static_top" in sensor_data and "rgb" in sensor_data["static_top"]:
                rgb_image = sensor_data["static_top"]["rgb"].cpu().numpy()
                if rgb_image.ndim == 4:
                    rgb_image = rgb_image[0]

                if rgb_image.dtype != np.uint8:
                    rgb_image = (rgb_image * 255).astype(np.uint8)

                bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                static_rgb_path = (
                    self.episode_dir / "static_top_rgb" / f"{frame_str}.jpg"
                )
                cv2.imwrite(str(static_rgb_path), bgr_image)
                image_paths["static_top_rgb"] = f"static_top_rgb/{frame_str}.jpg"

                # Store frame for static top video compilation
                if self.save_video:
                    self.static_top_video_frames.append(bgr_image)

            # Save static top depth
            if "static_top" in sensor_data and "depth" in sensor_data["static_top"]:
                depth_image = sensor_data["static_top"]["depth"].cpu().numpy()
                if depth_image.ndim == 4:
                    depth_image = depth_image[0]
                if depth_image.ndim == 3:
                    depth_image = depth_image[:, :, 0]

                # Convert depth to colorized format using Intel's HUE-based technique
                # ManiSkill depth is already in millimeters, just convert to uint16
                depth_mm = depth_image.astype(np.uint16)
                colorized_depth = colorize_depth_image(
                    depth_mm, min_depth=0.05, max_depth=2.0
                )

                static_depth_path = (
                    self.episode_dir / "static_top_depth" / f"{frame_str}.jpg"
                )
                cv2.imwrite(str(static_depth_path), colorized_depth)
                image_paths["static_top_depth"] = f"static_top_depth/{frame_str}.jpg"

                if self.save_video:
                    self.static_top_depth_video_frames.append(colorized_depth)

            # Save static top segmentation
            if (
                "static_top" in sensor_data
                and "segmentation" in sensor_data["static_top"]
            ):
                seg_image = sensor_data["static_top"]["segmentation"].cpu().numpy()
                if seg_image.ndim == 4:
                    seg_image = seg_image[0]
                if seg_image.ndim == 3:
                    seg_image = seg_image[:, :, 0]

                # Convert segmentation to colorized visualization
                # Segmentation values are typically small integers (0, 1, 2, ...)
                # We'll create a colorized version for visualization
                seg_colorized = self._colorize_segmentation(seg_image)

                static_seg_path = (
                    self.episode_dir / "static_top_segmentation" / f"{frame_str}.jpg"
                )
                cv2.imwrite(str(static_seg_path), seg_colorized)
                image_paths["static_top_segmentation"] = (
                    f"static_top_segmentation/{frame_str}.jpg"
                )

                # Save raw segmentation mask as PNG with simplified IDs (1-5)
                seg_raw = self._convert_to_simple_ids(seg_image)
                static_seg_raw_path = (
                    self.episode_dir
                    / "static_top_segmentation_raw"
                    / f"{frame_str}.png"
                )
                cv2.imwrite(str(static_seg_raw_path), seg_raw)
                image_paths["static_top_segmentation_raw"] = (
                    f"static_top_segmentation_raw/{frame_str}.png"
                )

                if self.save_video:
                    self.static_top_seg_video_frames.append(seg_colorized)

            # Save end effector camera RGB (eoat_right_top & eoat_left_top)
            # Handle right gripper camera
            if (
                "eoat_right_top" in sensor_data
                and "rgb" in sensor_data["eoat_right_top"]
            ):
                rgb_image = sensor_data["eoat_right_top"]["rgb"].cpu().numpy()
                if rgb_image.ndim == 4:
                    rgb_image = rgb_image[0]

                if rgb_image.dtype != np.uint8:
                    rgb_image = (rgb_image * 255).astype(np.uint8)

                bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

                # Save as eoat_right_top_rgb (main end effector camera)
                eoat_right_path = (
                    self.episode_dir / "eoat_right_top_rgb" / f"{frame_str}.jpg"
                )
                cv2.imwrite(str(eoat_right_path), bgr_image)
                image_paths["eoat_right_top_rgb"] = (
                    f"eoat_right_top_rgb/{frame_str}.jpg"
                )

                if self.save_video:
                    self.eoat_right_video_frames.append(bgr_image)

            # Handle left gripper camera RGB
            if "eoat_left_top" in sensor_data and "rgb" in sensor_data["eoat_left_top"]:
                rgb_image = sensor_data["eoat_left_top"]["rgb"].cpu().numpy()
                if rgb_image.ndim == 4:
                    rgb_image = rgb_image[0]

                if rgb_image.dtype != np.uint8:
                    rgb_image = (rgb_image * 255).astype(np.uint8)

                bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

                eoat_left_path = (
                    self.episode_dir / "eoat_left_top_rgb" / f"{frame_str}.jpg"
                )
                cv2.imwrite(str(eoat_left_path), bgr_image)
                image_paths["eoat_left_top_rgb"] = f"eoat_left_top_rgb/{frame_str}.jpg"

                if self.save_video:
                    self.eoat_left_video_frames.append(bgr_image)

            # Save depth for right & left gripper cameras
            if (
                "eoat_right_top" in sensor_data
                and "depth" in sensor_data["eoat_right_top"]
            ):
                depth_image = sensor_data["eoat_right_top"]["depth"].cpu().numpy()
                if depth_image.ndim == 4:
                    depth_image = depth_image[0]
                if depth_image.ndim == 3:
                    depth_image = depth_image[:, :, 0]

                depth_mm = depth_image.astype(np.uint16)
                colorized_depth = colorize_depth_image(
                    depth_mm, min_depth=0.05, max_depth=2.0
                )

                eoat_depth_path = (
                    self.episode_dir / "eoat_right_top_depth" / f"{frame_str}.jpg"
                )
                cv2.imwrite(str(eoat_depth_path), colorized_depth)
                image_paths["eoat_right_top_depth"] = (
                    f"eoat_right_top_depth/{frame_str}.jpg"
                )

                if self.save_video:
                    self.eoat_right_depth_video_frames.append(colorized_depth)

            if (
                "eoat_left_top" in sensor_data
                and "depth" in sensor_data["eoat_left_top"]
            ):
                depth_image = sensor_data["eoat_left_top"]["depth"].cpu().numpy()
                if depth_image.ndim == 4:
                    depth_image = depth_image[0]
                if depth_image.ndim == 3:
                    depth_image = depth_image[:, :, 0]

                depth_mm = depth_image.astype(np.uint16)
                colorized_depth = colorize_depth_image(
                    depth_mm, min_depth=0.05, max_depth=2.0
                )

                eoat_depth_path = (
                    self.episode_dir / "eoat_left_top_depth" / f"{frame_str}.jpg"
                )
                cv2.imwrite(str(eoat_depth_path), colorized_depth)
                image_paths["eoat_left_top_depth"] = (
                    f"eoat_left_top_depth/{frame_str}.jpg"
                )

                if self.save_video:
                    self.eoat_left_depth_video_frames.append(colorized_depth)

            # Save segmentation for right & left gripper cameras
            if (
                "eoat_right_top" in sensor_data
                and "segmentation" in sensor_data["eoat_right_top"]
            ):
                seg_image = sensor_data["eoat_right_top"]["segmentation"].cpu().numpy()
                if seg_image.ndim == 4:
                    seg_image = seg_image[0]
                if seg_image.ndim == 3:
                    seg_image = seg_image[:, :, 0]

                seg_colorized = self._colorize_segmentation(seg_image)

                eoat_seg_path = (
                    self.episode_dir
                    / "eoat_right_top_segmentation"
                    / f"{frame_str}.jpg"
                )
                cv2.imwrite(str(eoat_seg_path), seg_colorized)
                image_paths["eoat_right_top_segmentation"] = (
                    f"eoat_right_top_segmentation/{frame_str}.jpg"
                )

                seg_raw = self._convert_to_simple_ids(seg_image)
                eoat_seg_raw_path = (
                    self.episode_dir
                    / "eoat_right_top_segmentation_raw"
                    / f"{frame_str}.png"
                )
                cv2.imwrite(str(eoat_seg_raw_path), seg_raw)
                image_paths["eoat_right_top_segmentation_raw"] = (
                    f"eoat_right_top_segmentation_raw/{frame_str}.png"
                )

                if self.save_video:
                    self.eoat_right_seg_video_frames.append(seg_colorized)

            if (
                "eoat_left_top" in sensor_data
                and "segmentation" in sensor_data["eoat_left_top"]
            ):
                seg_image = sensor_data["eoat_left_top"]["segmentation"].cpu().numpy()
                if seg_image.ndim == 4:
                    seg_image = seg_image[0]
                if seg_image.ndim == 3:
                    seg_image = seg_image[:, :, 0]

                seg_colorized = self._colorize_segmentation(seg_image)

                eoat_seg_path = (
                    self.episode_dir / "eoat_left_top_segmentation" / f"{frame_str}.jpg"
                )
                cv2.imwrite(str(eoat_seg_path), seg_colorized)
                image_paths["eoat_left_top_segmentation"] = (
                    f"eoat_left_top_segmentation/{frame_str}.jpg"
                )

                seg_raw = self._convert_to_simple_ids(seg_image)
                eoat_seg_raw_path = (
                    self.episode_dir
                    / "eoat_left_top_segmentation_raw"
                    / f"{frame_str}.png"
                )
                cv2.imwrite(str(eoat_seg_raw_path), seg_raw)
                image_paths["eoat_left_top_segmentation_raw"] = (
                    f"eoat_left_top_segmentation_raw/{frame_str}.png"
                )

                if self.save_video:
                    self.eoat_left_seg_video_frames.append(seg_colorized)

        # Capture video frame from default human render camera
        if self.save_video:
            try:
                # Get default human render camera image
                video_frame = self.env.render()

                if video_frame is not None:
                    # Convert to numpy if it's a tensor
                    if hasattr(video_frame, "cpu"):
                        video_frame = video_frame.cpu().numpy()

                    # Handle different shapes
                    if video_frame.ndim == 4:
                        video_frame = video_frame[0]  # Remove batch dimension

                    # Ensure it's uint8
                    if video_frame.dtype != np.uint8:
                        video_frame = (video_frame * 255).astype(np.uint8)

                    # Convert RGB to BGR for OpenCV
                    if video_frame.shape[-1] == 3:
                        video_frame = cv2.cvtColor(video_frame, cv2.COLOR_RGB2BGR)

                    # Save individual video frame
                    video_frame_path = (
                        self.episode_dir / "default_camera_video" / f"{frame_str}.jpg"
                    )
                    cv2.imwrite(str(video_frame_path), video_frame)
                    image_paths["default_camera_video"] = (
                        f"default_camera_video/{frame_str}.jpg"
                    )

                    # Store frame for video compilation
                    self.video_frames.append(video_frame)

            except Exception as e:
                print(f"Warning: Could not capture video frame: {e}")

        # Add image paths to observation
        observation.update(image_paths)

        # Add the observation to the list
        self.observations.append(observation)
        self.frame_counter += 1

        if len(self.observations) % 10 == 0:
            print(f"Saved frame {len(self.observations)}: {phase}")

    def _colorize_segmentation(self, seg_image: np.ndarray) -> np.ndarray:
        """Convert segmentation mask to colorized visualization using predetermined colors.

        Args:
            seg_image: Segmentation mask with integer labels

        Returns:
            Colorized BGR image for visualization
        """
        # Get unique segment IDs
        unique_ids = np.unique(seg_image)

        # Create a colorized version
        height, width = seg_image.shape
        colorized = np.zeros((height, width, 3), dtype=np.uint8)

        # Apply predetermined colors to each segment
        for seg_id in unique_ids:
            mask = seg_image == seg_id
            if seg_id in self.color_mapping:
                # Use predetermined color
                colorized[mask] = self.color_mapping[seg_id]
            else:
                # Fallback: use hash-based color for unknown segments
                np.random.seed(int(seg_id))
                color = np.random.randint(50, 255, 3)
                colorized[mask] = color

        return colorized

    def _convert_to_simple_ids(self, seg_image: np.ndarray) -> np.ndarray:
        """Convert segmentation image to simplified IDs (1-5) as UINT8 PNG.

        Args:
            seg_image: Original segmentation mask with arbitrary integer labels

        Returns:
            Simplified segmentation mask with IDs 0-5 as UINT8
        """
        # Create output image
        height, width = seg_image.shape
        simple_seg = np.zeros((height, width), dtype=np.uint8)

        # Map each pixel to simplified ID
        for orig_id, simple_id in self.id_remapping.items():
            mask = seg_image == orig_id
            simple_seg[mask] = simple_id

        return simple_seg

    def _compile_video(self):
        """Compile video frames into MP4 video files for all cameras."""
        videos_to_compile = [
            ("default_camera", self.video_frames),
            ("static_top_camera", self.static_top_video_frames),
            ("eoat_left_top_camera", self.eoat_left_video_frames),
            ("eoat_right_top_camera", self.eoat_right_video_frames),
            ("static_top_depth", self.static_top_depth_video_frames),
            ("static_top_segmentation", self.static_top_seg_video_frames),
            ("eoat_left_top_depth", self.eoat_left_depth_video_frames),
            ("eoat_left_top_segmentation", self.eoat_left_seg_video_frames),
            ("eoat_right_top_depth", self.eoat_right_depth_video_frames),
            ("eoat_right_top_segmentation", self.eoat_right_seg_video_frames),
        ]

        for video_name, frames in videos_to_compile:
            if not frames:
                print(f"No {video_name} frames to compile")
                continue

            video_path = self.episode_dir / f"{self.episode_name}_{video_name}.mp4"

            try:
                # Get video dimensions from first frame
                height, width = frames[0].shape[:2]

                # Define codec and create VideoWriter
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(
                    str(video_path), fourcc, self.video_fps, (width, height)
                )

                # Write frames to video
                for frame in frames:
                    video_writer.write(frame)

                # Release video writer
                video_writer.release()

                print(f"{video_name.title()} video compiled successfully: {video_path}")
                print(f"Video contains {len(frames)} frames at {self.video_fps} FPS")

            except Exception as e:
                print(f"Error compiling {video_name} video: {e}")

    def follow_path(self, result, refine_steps: int = 0):
        """Override follow_path to save dataset at every control frame and handle bimanual actions."""
        n_step = result["position"].shape[0]
        print(f"Following path with {n_step} steps")

        for i in range(n_step + refine_steps):
            qpos = result["position"][min(i, n_step - 1)]
            if self.control_mode == "pd_joint_pos_vel":
                qvel = result["velocity"][min(i, n_step - 1)]
                action_np = np.hstack([qpos, qvel, self.gripper_state])
            else:
                action_np = np.hstack([qpos, self.gripper_state])

            # Format action for bimanual mode if necessary
            if hasattr(self.base_env.agent, "robot"):
                # Single arm
                action = action_np
            else:
                action = self._format_action_for_bimanual(action_np)

            obs, reward, terminated, truncated, info = self.env.step(action)

            # Save frame data
            self.save_frame_data(action_np, f"trajectory_step_{i}")

            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()

        return obs, reward, terminated, truncated, info

    def open_gripper(self):
        """Override open_gripper to save dataset at every control frame."""
        print(f"Opening gripper from {self.gripper_state} to {self.OPEN}")
        self.gripper_state = self.OPEN
        qpos = self.robot.get_qpos()[0, :6].cpu().numpy()

        for step in range(6):
            if self.control_mode == "pd_joint_pos":
                action_np = np.hstack([qpos, self.gripper_state])
            else:
                action_np = np.hstack([qpos, np.zeros_like(qpos), self.gripper_state])

            # Format for bimanual
            action = (
                action_np
                if hasattr(self.base_env.agent, "robot")
                else self._format_action_for_bimanual(action_np)
            )

            obs, reward, terminated, truncated, info = self.env.step(action)

            # Save frame data
            self.save_frame_data(action_np, f"open_gripper_step_{step}")

            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()

        return obs, reward, terminated, truncated, info

    def close_gripper(self, t: int = 6, gripper_state: float | None = None):
        """Override close_gripper to save dataset at every control frame."""
        old_gripper_state = self.gripper_state
        self.gripper_state = self.CLOSED if gripper_state is None else gripper_state
        print(f"Closing gripper from {old_gripper_state} to {self.gripper_state}")
        qpos = self.robot.get_qpos()[0, :6].cpu().numpy()

        for step in range(t):
            if self.control_mode == "pd_joint_pos":
                action_np = np.hstack([qpos, self.gripper_state])
            else:
                action_np = np.hstack([qpos, np.zeros_like(qpos), self.gripper_state])

            action = (
                action_np
                if hasattr(self.base_env.agent, "robot")
                else self._format_action_for_bimanual(action_np)
            )

            obs, reward, terminated, truncated, info = self.env.step(action)

            # Save frame data
            self.save_frame_data(action_np, f"close_gripper_step_{step}")

            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()

        return obs, reward, terminated, truncated, info

    def _get_camera_metadata(self) -> Dict:
        """Extract real camera intrinsic matrices and parameters from the environment."""
        # Get sensor parameters from the environment
        sensor_params = self.env.get_sensor_params()

        # Extract camera dimensions and intrinsics
        camera_metadata = {
            "k_mats": {},
            "camera_image_dimensions": {},
            "image_processing": {
                "resize_method": "cv2.INTER_LINEAR",
                "crop_method": "center_crop",
                "normalization": "0_to_1",
            },
        }

        # Get camera configurations and sensor data
        for sensor_name, sensor in self.env.scene.sensors.items():
            if hasattr(sensor, "camera") and hasattr(sensor, "config"):
                # Get intrinsic matrix from sensor parameters
                if sensor_name in sensor_params:
                    intrinsic_cv = sensor_params[sensor_name]["intrinsic_cv"]
                    # Convert from tensor to list (flatten 3x3 matrix row-wise)
                    if hasattr(intrinsic_cv, "cpu"):
                        intrinsic_cv = intrinsic_cv.cpu().numpy()
                    k_matrix = (
                        intrinsic_cv[0].flatten().tolist()
                    )  # Take first batch element

                    camera_metadata["k_mats"][sensor_name] = k_matrix

                    # Get camera dimensions
                    camera_metadata["camera_image_dimensions"][sensor_name] = {
                        "width": sensor.config.width,
                        "height": sensor.config.height,
                    }

                    print(f"✅ Extracted camera {sensor_name}: K={k_matrix[:3]}...")
                else:
                    print(f"⚠️ No sensor params found for {sensor_name}")

        # Add segmentation legends if they exist
        if hasattr(self, "segmentation_mapping") and self.segmentation_mapping:
            camera_metadata["segmentation_legend"] = {
                "segmentation_mapping": self.segmentation_mapping,
                "color_mapping": self.color_mapping,
                "id_remapping": self.id_remapping,
                "description": {
                    "segmentation_mapping": "Maps segmentation IDs to entity information (type, name, description)",
                    "color_mapping": "Maps segmentation IDs to BGR color values for visualization",
                    "id_remapping": "Maps original segmentation IDs to simplified IDs (0=background, 1-5=key objects)",
                },
            }
            print(
                f"✅ Added segmentation legend with {len(self.segmentation_mapping)} entities"
            )

        return camera_metadata

    def finish_episode(self):
        """Finish the episode and save the JSON file in the new format."""
        if self.episode_dir is None:
            raise ValueError("Episode not started.")

        # Get real camera metadata from environment
        camera_metadata = self._get_camera_metadata()

        # Create the new format with metadata and observations
        metadata = {
            "episode_name": self.episode_name,
            "robot_id": "a1_galaxea",
            **camera_metadata,
        }

        # Add table origin coordinate system information
        if self.use_table_origin:
            table_origin = self._compute_table_origin()
            metadata["coordinate_system"] = {
                "type": "table_origin",
                "table_origin": table_origin.tolist(),
                "right_arm_offset": RIGHT_ARM_OFFSET.tolist(),
                "left_arm_offset": LEFT_ARM_OFFSET.tolist(),
                "description": "All poses are relative to table_origin coordinate system",
            }
        else:
            metadata["coordinate_system"] = {
                "type": "world",
                "description": "All poses are in world coordinates",
            }

        episode_data = {
            "metadata": metadata,
            "observations": self.observations,
        }

        # Save episode data JSON file
        json_path = self.episode_dir / f"{self.episode_name}.json"
        with open(json_path, "w") as f:
            json.dump(episode_data, f, indent=2)

        print(f"Episode finished: {self.episode_name}")
        print(f"Total frames: {len(self.observations)}")
        print(f"Episode data saved to: {json_path}")

        # Compile video if video recording is enabled
        if self.save_video and (  # noqa: PLR0916
            len(self.video_frames) > 0
            or len(self.static_top_video_frames) > 0
            or len(self.eoat_left_video_frames) > 0
            or len(self.eoat_right_video_frames) > 0
            or len(self.static_top_depth_video_frames) > 0
            or len(self.static_top_seg_video_frames) > 0
            or len(self.eoat_left_depth_video_frames) > 0
            or len(self.eoat_left_seg_video_frames) > 0
            or len(self.eoat_right_depth_video_frames) > 0
            or len(self.eoat_right_seg_video_frames) > 0
        ):
            self._compile_video()

        # Reset for next episode
        self.observations = []
        self.episode_dir = None
        self.frame_counter = 0
        self.last_timestamp = None
        self.segmentation_mapping = {}
        self.color_mapping = {}
        self.id_remapping = {}
        self.video_frames = []
        self.static_top_video_frames = []
        self.eoat_video_frames = []
        self.eoat_left_video_frames = []
        self.eoat_right_video_frames = []
        self.static_top_depth_video_frames = []
        self.static_top_seg_video_frames = []
        self.eoat_left_depth_video_frames = []
        self.eoat_left_seg_video_frames = []
        self.eoat_right_depth_video_frames = []
        self.eoat_right_seg_video_frames = []


def generate_pick_box_dataset(
    env_config: dict,
    num_episodes: int = 1,
    output_root_dir: str = "~/training_data/galaxea_box_pnp_sim/",
    seed_start: int = 42,
    debug: bool = False,
    vis: bool = False,
    save_video: bool = True,
    video_fps: int = 30,
    use_table_origin: bool = True,
):
    """Generate multiple episodes of pick box dataset.

    Args:
        env_config: Environment configuration dictionary for gym.make()
        num_episodes: Number of episodes to generate
        output_root_dir: Root directory for dataset
        seed_start: Starting seed for episodes
        debug: Enable debug output
        vis: Enable visualization
        save_video: Enable video recording from default camera (default: True)
        video_fps: Frame rate for compiled video (default: 30)
    """
    for episode_idx in range(num_episodes):
        print(f"\n{'=' * 60}")
        print(f"GENERATING EPISODE {episode_idx + 1}/{num_episodes}")
        print(f"{'=' * 60}")

        # Create fresh environment for each episode to ensure proper randomization
        env = gym.make(**env_config)

        # Reset environment with unique seed
        env.reset(seed=seed_start + episode_idx)

        # Check if environment uses a1_galaxea (single or bimanual)
        robot_uids = env.unwrapped.robot_uids
        if isinstance(robot_uids, tuple):
            # Bimanual mode - check if all robots are a1_galaxea
            if not all(uid == "a1_galaxea" for uid in robot_uids):
                raise ValueError(
                    f"This generator only supports 'a1_galaxea', but got {robot_uids}."
                )
        # Single robot mode
        elif robot_uids != "a1_galaxea":
            raise ValueError(
                f"This generator only supports 'a1_galaxea', but got {robot_uids}."
            )

        # Get base pose from active agent (right arm in bimanual mode)
        if hasattr(env.unwrapped.agent, "robot"):
            # Single robot mode
            base_pose = env.unwrapped.agent.robot.pose
        else:
            # Bimanual mode - use right arm (active agent, index 1)
            base_pose = env.unwrapped.agent.agents[1].robot.pose

        # Create dataset generator
        generator = A1GalaxeaDatasetGenerator(
            env,
            output_root_dir=output_root_dir,
            save_video=save_video,
            video_fps=video_fps,
            use_table_origin=use_table_origin,
            debug=debug,
            vis=vis,
            base_pose=base_pose,
            visualize_target_grasp_pose=vis,
            print_env_info=debug,
        )

        # Start episode
        generator.start_episode()

        # Save initial frame
        generator.save_frame_data(phase="initial")

        FINGER_LENGTH = 0.025
        env_unwrapped = env.unwrapped

        try:
            # Compute grasp pose
            obb = get_actor_obb(env_unwrapped.b5box)
            approaching = np.array([0, 0, -1])
            # Get target_closing from active agent (right arm in bimanual mode)
            if hasattr(env.agent, "tcp"):
                # Single robot mode
                target_closing = (
                    env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1]
                    .cpu()
                    .numpy()
                )
            else:
                # Bimanual mode - use right arm (active agent, index 1)
                target_closing = (
                    env.agent.agents[1]
                    .tcp.pose.to_transformation_matrix()[0, :3, 1]
                    .cpu()
                    .numpy()
                )

            grasp_info = compute_grasp_info_by_obb(
                obb,
                approaching=approaching,
                target_closing=target_closing,
                depth=FINGER_LENGTH,
            )
            closing, _ = grasp_info["closing"], grasp_info["center"]
            # Build grasp pose using active agent (right arm in bimanual mode)
            if hasattr(env.agent, "build_grasp_pose"):
                # Single robot mode
                grasp_pose = env.agent.build_grasp_pose(
                    approaching, closing, env.b5box.pose.sp.p
                )
            else:
                # Bimanual mode - use right arm (active agent, index 1)
                grasp_pose = env.agent.agents[1].build_grasp_pose(
                    approaching, closing, env.b5box.pose.sp.p
                )

            # Execute pick and place sequence
            print("Moving to pre-grasp pose...")
            reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
            result = generator.move_to_pose_with_RRTConnect(reach_pose)
            if result == -1:
                print("Failed to reach pre-grasp pose")
                continue

            print("Descending to grasp pose...")
            result = generator.move_to_pose_with_RRTConnect(grasp_pose)
            if result == -1:
                print("Failed to reach grasp pose")
                continue

            print("Closing gripper...")
            generator.close_gripper()

            print("Lifting box...")
            # Get current_pos from active agent (right arm in bimanual mode)
            if hasattr(env.agent, "tcp"):
                # Single robot mode
                current_pos = env.agent.tcp.pose.p.cpu().numpy().flatten()
            else:
                # Bimanual mode - use right arm (active agent, index 1)
                current_pos = env.agent.agents[1].tcp.pose.p.cpu().numpy().flatten()
            lift_pose = sapien.Pose(current_pos + np.array([0, 0, 0.15]), grasp_pose.q)
            result = generator.move_to_pose_with_RRTConnect(lift_pose)
            if result == -1:
                print("Failed to lift box")
                continue

            print("Moving to above basket...")
            basket_center = env.basket.pose.sp.p
            lifted_height = lift_pose.p[2]
            above_basket_pose = sapien.Pose(
                np.array([basket_center[0], basket_center[1], lifted_height]),
                grasp_pose.q,
            )
            result = generator.move_to_pose_with_RRTConnect(above_basket_pose)
            if result == -1:
                print("Failed to move to above basket")
                continue

            print("Lowering into basket...")
            basket_drop_height = basket_center[2] + 0.15
            lower_pose = sapien.Pose(
                np.array([basket_center[0], basket_center[1], basket_drop_height]),
                grasp_pose.q,
            )
            result = generator.move_to_pose_with_RRTConnect(lower_pose)
            if result == -1:
                print("Failed to lower into basket")
                continue

            print("Opening gripper to release...")
            generator.open_gripper()

            print("Episode completed successfully!")

        except Exception as e:
            print(f"Episode failed with error: {e}")

        finally:
            # Always finish the episode to save data
            generator.finish_episode()
            generator.close()

            # Close environment to free resources
            env.close()

    print(f"\nDataset generation complete! Generated {num_episodes} episodes.")


if __name__ == "__main__":
    import argparse
    import os

    import gymnasium as gym

    import mani_skill

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate A1 Galaxea pick box dataset")
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=3,
        help="Number of episodes to generate (default: 3)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="~/training_data/galaxea_box_pnp_sim/",
        help="Root directory for dataset output (default: ~/training_data/galaxea_box_pnp_sim/)",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=42,
        help="Starting seed for episodes (default: 42)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output",
    )
    parser.add_argument(
        "--vis",
        action="store_true",
        help="Enable visualization",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Disable video recording",
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=30,
        help="Frame rate for compiled video (default: 30)",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default="PickBoxBimanual-v1",
        choices=["PickBox-v1", "PickBoxBimanual-v1"],
        help="Environment ID to use (default: PickBoxBimanual-v1)",
    )
    parser.add_argument(
        "--robot-uids",
        type=str,
        default="a1_galaxea",
        help="Robot UIDs to use (default: a1_galaxea)",
    )
    parser.add_argument(
        "--obs-mode",
        type=str,
        default="rgb+depth+segmentation",
        help="Observation mode (default: rgb+depth+segmentation)",
    )
    parser.add_argument(
        "--control-mode",
        type=str,
        default="pd_joint_pos",
        help="Control mode (default: pd_joint_pos)",
    )
    parser.add_argument(
        "--use-table-origin",
        action="store_true",
        default=True,
        help="Use table origin coordinate system instead of world coordinates",
    )
    parser.add_argument(
        "--no-table-origin",
        action="store_true",
        help="Use world coordinates instead of table origin coordinate system",
    )

    args = parser.parse_args()

    # Handle table origin flag
    use_table_origin = args.use_table_origin and not args.no_table_origin

    # Create environment configuration dictionary
    env_config = {
        "id": args.env_id,
        "robot_uids": args.robot_uids,
        "obs_mode": args.obs_mode,
        "render_mode": "rgb_array",
        "control_mode": args.control_mode,
        "verbose": args.debug,  # Enable verbose mode when debug is enabled
    }

    # Generate dataset
    generate_pick_box_dataset(
        env_config,
        num_episodes=args.n_episodes,
        output_root_dir=os.path.expanduser(args.output_dir),
        seed_start=args.seed_start,
        debug=args.debug,
        vis=args.vis,
        save_video=not args.no_video,
        video_fps=args.video_fps,
        use_table_origin=use_table_origin,
    )
