"""
Single-arm pick-and-place task that re-uses the b5box and basket assets from
BimanualPickPlace but keeps the reward shaping minimal (PickCube style).
The robot (Panda/Panda Wrist-Cam or A1 Galaxea) is positioned at the same
*right-arm* pose defined in the BimanualPickPlace configuration.

Task flow:
1. Reach the b5box.
2. Grasp it (binary test).
3. Move the box to the basket and release.

Success is declared when the box centre is within `goal_thresh` of the basket
centre **and** the robot is static.
"""

from typing import Any, Dict, Tuple, Union

import numpy as np
import sapien
import torch
from sapien import physx

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.agents.robots import A1Galaxea, Panda, PandaWristCam, XArm6Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.tasks.tabletop.bimanual_pick_place_cfgs import BIMANUAL_PICK_PLACE_CONFIG
from mani_skill.envs.tasks.tabletop.pick_box_cfgs import PICK_BOX_CONFIGS
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.geometry.rotation_conversions import quaternion_apply
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig

__all__ = ["PickBoxBimanualEnv", "PickBoxEnv"]

# A1 Galaxea table origin coordinate system constants
# Right arm offset from table origin (in meters)
RIGHT_ARM_OFFSET = torch.tensor([-0.025, -0.365, 0.005])
# Left arm offset from table origin (in meters)
LEFT_ARM_OFFSET = torch.tensor([-0.025, 0.365, 0.005])


@register_env("PickBox-v1", max_episode_steps=150)
class PickBoxEnv(BaseEnv):
    """Single-arm or bimanual pick-and-place task using b5box and basket assets."""

    SUPPORTED_ROBOTS = [
        "panda",
        "panda_wristcam",
        "a1_galaxea",
        "xarm6_robotiq",
        ("panda", "panda"),
        ("panda_wristcam", "panda_wristcam"),
        ("a1_galaxea", "a1_galaxea"),
        ("xarm6_robotiq", "xarm6_robotiq"),
    ]

    agent: Union[
        Panda,
        PandaWristCam,
        A1Galaxea,
        XArm6Robotiq,
        MultiAgent[Tuple[Panda, Panda]],
        MultiAgent[Tuple[PandaWristCam, PandaWristCam]],
        MultiAgent[Tuple[A1Galaxea, A1Galaxea]],
        MultiAgent[Tuple[XArm6Robotiq, XArm6Robotiq]],
    ]

    # Task constants (match BimanualPickPlace)
    # Scale factor applied to the STL/OBJ meshes for the b5box asset. The
    # original half-size was 5 cm; after an 80 % scale-down the new half-size
    # becomes 4 cm.
    b5box_half_size = 0.02  # 4 cm half-extent after scaling (will be overridden by config)
    goal_thresh = 0.015  # success threshold (metres) (will be overridden by config)

    # Base physical properties for randomization
    base_static_friction = 1.0
    base_dynamic_friction = 1.0
    base_density = 1000.0  # kg/m³
    base_linear_damping = 0.1  # Base linear damping coefficient

    # Visual variation parameters
    box_color_ranges = {
        "red": (0.2, 0.95),  # Red channel range (higher values)
        "green": (0.2, 0.95),  # Green channel range (lower max)
        "blue": (0.2, 0.95),  # Blue channel range (lower max)
        "alpha": (1.0, 1.0),  # Alpha channel (keep opaque)
    }

    table_color_ranges = {
        "red": (0.1, 0.95),  # Full red range for dramatic color variations
        "green": (0.1, 0.95),  # Full green range
        "blue": (0.1, 0.95),  # Full blue range
        "alpha": (1.0, 1.0),  # Alpha channel (keep opaque)
    }

    # Lighting variation parameters - moderate ranges for reasonable lighting
    ambient_light_ranges = {  # noqa: RUF012
        "min_intensity": (0.1, 0.3),  # Moderate ambient light range
        "max_intensity": (
            0.4,
            0.8,
        ),  # Reduced max ambient light for less extreme brightness
    }

    directional_light_ranges = {  # noqa: RUF012
        "intensity": (
            0.8,
            2.5,
        ),  # Reduced directional light intensity range for more reasonable lighting
        "direction_x": (-1.0, 1.0),  # X component of light direction
        "direction_y": (-1.0, 1.0),  # Y component of light direction
        "direction_z": (-1.0, -0.2),  # Z component (always downward, wider range)
    }

    # Store per-environment lighting directions (set at environment creation)
    _env_lighting_directions = None

    # Store reference to the directional light for updating direction
    _directional_light = None

    # Shadow box parameters for floating overhead obstacle
    shadow_box_ranges = {  # noqa: RUF012
        "probability": 0.5,  # 50% chance to spawn shadow box
        "width": (0.15, 0.35),  # Box width (x-axis)
        "depth": (0.15, 0.35),  # Box depth (y-axis)
        "height": (0.02, 0.08),  # Box thickness (z-axis)
        "x_position": (-0.1, 0.3),  # X position range
        "y_position": (-0.2, 0.2),  # Y position range
        "z_height": (1.5, 2.0),  # Height above table surface (1.5-2.0m above table)
        "rotation": (0, 2 * np.pi),  # Random rotation around Z-axis
    }

    # Class variable to track total environments created across all instances
    _total_environments_created = 0

    # B5box asset usage parameters
    use_real_b5box_probability = 0.8  # 80% chance to use real b5box assets instead of primitive box

    def __init__(
        self,
        *args,
        robot_uids="a1_galaxea",
        obs_mode="state",
        verbose=False,
        enable_shadow=True,
        bimanual=False,
        use_table_origin=True,
        **kwargs,
    ):
        self.verbose = verbose
        # Always enable shadows as requested
        self.enable_shadow = True

        # Initialize directional light reference
        self._directional_light = None

        # Handle bimanual mode
        self.bimanual = bimanual
        if self.bimanual:
            # Convert single robot_uids to tuple for bimanual mode
            if isinstance(robot_uids, str):
                robot_uids = (robot_uids, robot_uids)
            elif isinstance(robot_uids, (list, tuple)) and len(robot_uids) == 1:
                robot_uids = (robot_uids[0], robot_uids[0])
            self.robot_type = robot_uids[0]  # Use first robot type for config lookup
        else:
            # Single arm mode
            if isinstance(robot_uids, (list, tuple)):
                robot_uids = robot_uids[0]  # Use first robot for single arm
            self.robot_type = robot_uids

        # Resolve to a canonical UID string or tuple
        self.robot_uids = robot_uids

        # Table origin coordinate system
        self.use_table_origin = use_table_origin
        self._table_origin = None
        self._table_origin_computed = False

        # Use robot-specific configs like PickCube does
        if self.robot_type in PICK_BOX_CONFIGS:
            cfg = PICK_BOX_CONFIGS[self.robot_type]
            self.goal_thresh = cfg["goal_thresh"]
            self.b5box_half_size = cfg["cube_half_size"]
            self.cube_half_size = cfg["cube_half_size"]
            # Store camera configurations for sensor setup
            self.sensor_cam_eye_pos = cfg["sensor_cam_eye_pos"]
            self.sensor_cam_target_pos = cfg["sensor_cam_target_pos"]
            self.human_cam_eye_pos = cfg["human_cam_eye_pos"]
            self.human_cam_target_pos = cfg["human_cam_target_pos"]
        else:
            # Fallback to default values
            cfg = PICK_BOX_CONFIGS["panda"]
            self.goal_thresh = cfg["goal_thresh"]
            self.b5box_half_size = cfg["cube_half_size"]
            self.cube_half_size = cfg["cube_half_size"]
            self.sensor_cam_eye_pos = cfg["sensor_cam_eye_pos"]
            self.sensor_cam_target_pos = cfg["sensor_cam_target_pos"]
            self.human_cam_eye_pos = cfg["human_cam_eye_pos"]
            self.human_cam_target_pos = cfg["human_cam_target_pos"]

        # Initialize box_half_sizes early to avoid observation space generation errors
        # This will be updated later in _load_b5box based on the actual box configuration per scene
        self.box_half_sizes = [
            self.b5box_half_size,
            self.b5box_half_size,
            self.b5box_half_size,
        ]

        # Will store per-scene box dimensions after scene loading
        self.per_scene_box_half_sizes = None

        # Assets directory (same as bimanual task)
        self.asset_root = PACKAGE_ASSET_DIR / "tasks/a1_pick_place"

        # Note: use_real_b5box will be determined per sub-scene in _load_b5box using _batched_episode_rng

        if self.verbose:
            print("PickBox Environment Initialization:")
            print(f"  Robot: {robot_uids}")
            print(f"  Bimanual mode: {bimanual}")
            print(f"  Observation mode: {obs_mode}")
            print(f"  Cube half-size: {self.cube_half_size}")
            print(f"  Box half-size: {self.b5box_half_size}")
            print(f"  Goal threshold: {self.goal_thresh}")
            print(f"  Shadows enabled: {enable_shadow}")
            print("  B5box asset type will be randomized per sub-scene (30% real, 70% primitive)")
            print("  Both asset types have physics variation (friction, mass, damping, scale)")

        super().__init__(
            *args,
            robot_uids=robot_uids,
            obs_mode=obs_mode,
            enable_shadow=enable_shadow,
            **kwargs,
        )

        if self.verbose:
            print("  ✓ PickBox Environment initialization complete!")
            print("  Success criteria: object placed + released + stable")
            print("    - Object within goal threshold of basket center")
            print("    - Object released (not grasped)")
            print("    - Object stable (low velocity)")
            print("  Visual variations enabled:")
            print("    - Box color randomization (red-dominant)")
            print("    - Table color randomization (wood-like tones)")
            print("    - Lighting condition variations (ambient + directional)")
            # Show observation space info
            expected_features = self._get_expected_features()
            total_features = sum(expected_features.values())
            print(f"  Expected observation features: {total_features} total")
            for name, count in expected_features.items():
                print(f"    {name}: {count} features")

        # Box dimensions are now randomized per sub-scene during _load_b5box

    @property
    def left_agent(self) -> Union[Panda, PandaWristCam, A1Galaxea, XArm6Robotiq]:
        """Get the left agent (only available in bimanual mode)."""
        if self.bimanual:
            return self.agent.agents[0]
        else:
            return None

    @property
    def right_agent(self) -> Union[Panda, PandaWristCam, A1Galaxea, XArm6Robotiq]:
        """Get the right agent (active agent in single-arm mode, right agent in bimanual mode)."""
        if self.bimanual:
            return self.agent.agents[1]
        else:
            return self.agent

    @property
    def active_agent(self) -> Union[Panda, PandaWristCam, A1Galaxea, XArm6Robotiq]:
        """Get the active agent (right agent in both single-arm and bimanual modes)."""
        if self.bimanual:
            return self.agent.agents[1]  # Right agent is active in bimanual mode
        else:
            return self.agent

    @property
    def _default_sim_config(self):
        """Configure GPU memory settings to handle high collision pair requirements."""
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                # Increase found_lost_pairs_capacity to handle the reported requirement
                # Error indicated need for 127953602 pairs, so we set to 2**27 (134M pairs)
                found_lost_pairs_capacity=2**27,  # 134M pairs (increased from default 2**25)
                # Increase other memory buffers proportionally for complex scenes
                max_rigid_patch_count=2**20,  # 1M patches (increased from default 2**18)
                max_rigid_contact_count=2**21,  # 2M contacts (increased from default 2**19)
                # Increase heap capacity for additional memory overhead
                heap_capacity=2**27,  # 128MB (increased from default 2**26)
                # Increase temp buffer for memory operations
                temp_buffer_capacity=2**25,  # 32MB (increased from default 2**24)
            )
        )

    # ---------------------------------------------------------------------
    # Camera configuration (static top & human render) --------------------
    # ---------------------------------------------------------------------

    @property
    def _default_sensor_configs(self):
        # In bimanual mode, we need to explicitly add robot cameras since MultiAgent
        # sensor configs might not be automatically included
        if self.robot_type == "a1_galaxea" and self.bimanual:
            # For bimanual A1 Galaxea, don't add any cameras here
            # The end effector cameras will be automatically provided by the MultiAgent
            # from each robot's _sensor_configs and will be renamed in _setup_bimanual_camera_mounts
            # The static_top cameras are also provided by each robot
            return []
        elif self.robot_type == "a1_galaxea":
            return []  # Single-arm A1 Galaxea provides its own cameras
        else:
            # For other robots, use sensor camera configuration

            pose = sapien_utils.look_at(eye=self.sensor_cam_eye_pos, target=self.sensor_cam_target_pos)
            return [CameraConfig("base_camera", pose, 224, 224, fov=1.012, near=0.01, far=10)]

    @property
    def _default_human_render_camera_configs(self):
        # Use the same pose as static_top camera for A1 Galaxea, fallback for other robots
        if self.robot_type == "a1_galaxea":
            cfg = PICK_BOX_CONFIGS[self.robot_type]
            pose = sapien.Pose(
                p=cfg["static_top_cam_pos"],
                q=cfg["static_top_cam_quat"],
            )
        else:
            pose = sapien_utils.look_at(eye=self.human_cam_eye_pos, target=self.human_cam_target_pos)

        return CameraConfig("render_camera", pose, 448, 448, fov=1.012, near=0.01, far=10)

    # # ------------------------------------------------------------------
    # # Curriculum Learning Support -------------------------------------
    # # ------------------------------------------------------------------

    # def _apply_curriculum_config(self, options: dict) -> dict:
    #     """Apply curriculum level configuration to the environment.

    #     Args:
    #         options: Reset options that may contain curriculum configuration

    #     Returns:
    #         dict: The curriculum configuration applied
    #     """
    #     curriculum_config = {}

    #     # Extract curriculum parameters from options
    #     if "curriculum_level" in options:
    #         curriculum_config["curriculum_level"] = options["curriculum_level"]

    #     if "max_episode_steps" in options:
    #         curriculum_config["max_episode_steps"] = options["max_episode_steps"]
    #         # Note: max_episode_steps is handled by the curriculum wrapper

    #     if "box_dimensions" in options:
    #         curriculum_config["box_dimensions"] = options["box_dimensions"]
    #         # Update box dimensions for this episode
    #         self._update_box_dimensions(options["box_dimensions"])

    #     if "use_real_b5box_probability" in options:
    #         curriculum_config["use_real_b5box_probability"] = options[
    #             "use_real_b5box_probability"
    #         ]
    #         self.use_real_b5box_probability = options["use_real_b5box_probability"]

    #     if "physics_variation" in options:
    #         curriculum_config["physics_variation"] = options["physics_variation"]
    #         # Physics variations are applied during object creation

    #     if "visual_variation" in options:
    #         curriculum_config["visual_variation"] = options["visual_variation"]
    #         # Visual variations are applied during lighting setup

    #     if self.verbose and curriculum_config:
    #         print(f"  Applied curriculum config: {curriculum_config}")

    #     return curriculum_config

    # def _update_box_dimensions(self, box_dimensions: dict):
    #     """Update box dimensions for curriculum learning.

    #     Args:
    #         box_dimensions: Dictionary with 'type' and dimension parameters
    #     """
    #     if box_dimensions["type"] == "cube":
    #         # Basic cube - uniform dimensions
    #         size = box_dimensions.get("size", 0.02)
    #         self.box_half_sizes = [size, size, size]
    #     elif box_dimensions["type"] == "non_uniform":
    #         # Non-uniform cube (x-axis 3x longer)
    #         base_size = box_dimensions.get("base_size", 0.02)
    #         x_multiplier = box_dimensions.get("x_multiplier", 3.0)
    #         self.box_half_sizes = [base_size * x_multiplier, base_size, base_size]
    #     elif box_dimensions["type"] == "randomized":
    #         # Randomized dimensions - will be applied per scene in _load_primitive_b5box_for_scene
    #         base_size = box_dimensions.get("base_size", 0.02)
    #         self.box_half_sizes = [base_size, base_size, base_size]

    #     # Update per-scene box dimensions if they exist
    #     if hasattr(self, "per_scene_box_half_sizes") and self.per_scene_box_half_sizes:
    #         for i in range(len(self.per_scene_box_half_sizes)):
    #             self.per_scene_box_half_sizes[i] = self.box_half_sizes.copy()

    # ------------------------------------------------------------------
    # Table origin coordinate system -----------------------------------
    # ------------------------------------------------------------------

    def _compute_table_origin(self) -> torch.Tensor:
        """Compute table origin position based on robot arm base positions.

        Returns:
            torch.Tensor: Table origin position [x, y, z] in world coordinates
        """
        if self._table_origin_computed:
            return self._table_origin

        # Get robot arm base positions
        if self.bimanual:
            # Bimanual mode - use actual robot positions
            left_robot = self.left_agent.robot
            right_robot = self.right_agent.robot
            left_base_pos = left_robot.pose.p[0]
            right_base_pos = right_robot.pose.p[0]
        else:
            # Single arm mode - use right arm and estimate left arm position
            right_robot = self.active_agent.robot
            right_base_pos = right_robot.pose.p[0]

            # For single arm, estimate left arm position using bimanual config offsets
            cfg = PICK_BOX_CONFIGS[self.robot_type]
            if "left_arm" in cfg:
                # cfg["left_arm"]["pose"] is already a tensor from Pose.p
                # Ensure it has the same shape as right_base_pos by taking [0] if needed
                left_arm_pose = cfg["left_arm"]["pose"]
                if hasattr(left_arm_pose, "shape") and len(left_arm_pose.shape) > 1:
                    left_base_pos = left_arm_pose[0].to(self.device)
                else:
                    left_base_pos = torch.tensor(left_arm_pose, device=self.device)
            else:
                # Fallback: mirror right arm position
                left_base_pos = right_base_pos.clone()
                left_base_pos[1] = -left_base_pos[1]  # Mirror Y coordinate

        # Ensure offsets are on the same device
        right_arm_offset = RIGHT_ARM_OFFSET.to(self.device)
        left_arm_offset = LEFT_ARM_OFFSET.to(self.device)

        # Calculate table origin such that:
        # right_base_pos = table_origin + right_arm_offset
        # left_base_pos = table_origin + left_arm_offset
        # Take average to get best estimate
        table_origin_from_right = right_base_pos - right_arm_offset
        table_origin_from_left = left_base_pos - left_arm_offset

        self._table_origin = (table_origin_from_right + table_origin_from_left) / 2
        self._table_origin_computed = True

        if self.verbose:
            print(f"🔧 Table origin computed: {self._table_origin}")
            print(f"🔧 Right arm world pos: {right_base_pos}")
            print(f"🔧 Left arm world pos: {left_base_pos}")
            print(f"🔧 Expected right arm table-relative: {self._table_origin + right_arm_offset}")
            print(f"🔧 Expected left arm table-relative: {self._table_origin + left_arm_offset}")

        return self._table_origin

    def _transform_pose_to_table_origin(self, pose: torch.Tensor) -> torch.Tensor:
        """Transform poses from world coordinates to table_origin coordinates.

        Args:
            pose: Tensor of shape (num_envs, 7) containing [x, y, z, qx, qy, qz, qw] in world coordinates

        Returns:
            Tensor of shape (num_envs, 7) containing [x, y, z, qx, qy, qz, qw] in table_origin coordinates
        """
        if not self.use_table_origin:
            return pose

        table_origin = self._compute_table_origin()

        # Transform position relative to table origin
        transformed_pose = pose.clone()
        transformed_pose[:, :3] = pose[:, :3] - table_origin.unsqueeze(0)
        # Orientation (quaternion) remains the same

        return transformed_pose

    def _transform_position_to_table_origin(self, position: torch.Tensor) -> torch.Tensor:
        """Transform positions from world coordinates to table_origin coordinates.

        Args:
            position: Tensor of shape (num_envs, 3) containing [x, y, z] in world coordinates

        Returns:
            Tensor of shape (num_envs, 3) containing [x, y, z] in table_origin coordinates
        """
        if not self.use_table_origin:
            return position

        table_origin = self._compute_table_origin()
        return position - table_origin.unsqueeze(0)

    # ------------------------------------------------------------------
    # Scene & robot loading --------------------------------------------
    # ------------------------------------------------------------------

    def _load_agent(self, options: dict):
        if self.bimanual:
            # Load both arms using poses from config
            cfg = PICK_BOX_CONFIGS[self.robot_type]
            left_pose = cfg["left_arm"]["pose"]
            right_pose = cfg["right_arm"]["pose"]
            super()._load_agent(options, [left_pose, right_pose])

            # Set up camera mounts for bimanual A1 Galaxea
            if self.robot_type == "a1_galaxea":
                self._setup_bimanual_camera_mounts()
        else:
            # Use bimanual config positioning for single right arm
            cfg = PICK_BOX_CONFIGS[self.robot_type]
            right_pose = cfg["right_arm"]["pose"]
            super()._load_agent(options, right_pose)

    def _setup_bimanual_camera_mounts(self):
        """Set up camera mounts for bimanual A1 Galaxea end effector cameras."""
        # The MultiAgent automatically merges sensor configs from both robots
        # Each A1 Galaxea robot provides an "end_effector_camera" - we need to rename them
        # to "eoat_left_top" and "eoat_right_top" to match the expected naming convention
        # Also need to remove duplicate static_top cameras

        # Find and rename the end effector cameras
        end_effector_cameras = []
        static_top_cameras = []

        for sensor_config in self.agent.sensor_configs:
            if sensor_config.uid == "end_effector_camera":
                end_effector_cameras.append(sensor_config)
            elif sensor_config.uid == "static_top":
                static_top_cameras.append(sensor_config)

        # Rename the end effector cameras based on their mount (left vs right arm)
        if len(end_effector_cameras) == 2:
            # Determine which camera belongs to which arm by checking the mount
            left_camera = None
            right_camera = None

            for camera in end_effector_cameras:
                # Check if this camera is mounted on the left or right arm
                if camera.mount == self.left_agent.robot.links_map["galaxea_eoat_set"]:
                    left_camera = camera
                elif camera.mount == self.right_agent.robot.links_map["galaxea_eoat_set"]:
                    right_camera = camera

            # Rename the cameras
            if left_camera:
                left_camera.uid = "eoat_left_top"
            if right_camera:
                right_camera.uid = "eoat_right_top"

        # Remove duplicate static_top cameras, keeping only the first one
        if len(static_top_cameras) > 1:
            # Create a new sensor_configs list without the duplicate static_top cameras
            new_sensor_configs = []
            static_top_added = False

            for sensor_config in self.agent.sensor_configs:
                if sensor_config.uid == "static_top":
                    if not static_top_added:
                        new_sensor_configs.append(sensor_config)
                        static_top_added = True
                    # Skip duplicate static_top cameras
                else:
                    new_sensor_configs.append(sensor_config)

            self.agent.sensor_configs = new_sensor_configs

    def _load_scene(self, options: dict):
        # Table, b5box, basket, and goal site (for visualization) ----------
        self.table = self._create_table()
        self.b5box = self._load_b5box()
        self.basket = self._load_basket()
        self.goal_site = self._create_goal_site()

        # Create floating shadow box for more interesting lighting
        self.shadow_box = self._create_shadow_box()

        # Setup lighting at environment creation (not per episode)
        self._setup_env_creation_lighting()

    def _load_lighting(self, options: dict):
        """Override base lighting setup - we handle lighting in _setup_env_creation_lighting."""
        # Don't call super()._load_lighting() to avoid default directional lights
        # Our lighting is set up in _setup_env_creation_lighting() which is called from _load_scene()
        pass

    # ------------------------------------------------------------------
    # Asset builders ----------------------------------------------------
    # ------------------------------------------------------------------

    def _create_table(self):
        cfg = BIMANUAL_PICK_PLACE_CONFIG["scene"]["table"]
        builder = self.scene.create_actor_builder()
        # Use a high-friction PhysX material so objects on the table do not
        # slide easily during manipulation.
        MAX_FRIC = 2.0  # normal friction coefficient for PhysX materials
        table_fric_mat = physx.PhysxMaterial(
            static_friction=MAX_FRIC,
            dynamic_friction=MAX_FRIC,
            restitution=0.0,
        )

        # Use default table color - will be randomized per episode in _initialize_episode
        default_table_color = [0.6, 0.5, 0.4, 1.0]  # Default brownish color

        builder.add_box_collision(
            half_size=cfg["size"],
            material=table_fric_mat,
        )
        builder.add_box_visual(half_size=cfg["size"], material=default_table_color)
        builder.set_initial_pose(cfg["pose"].sp)

        if self.verbose:
            print("    Table created with default color (will be randomized per episode)")

        return builder.build_static(name="table")

    def _load_b5box(self):
        """Create either a primitive box or load the real b5box asset based on per-sub-scene randomization.

        For primitive box: creates a red box with x-axis 3x longer than other dimensions.
        For real b5box: loads the actual b5box asset files (STL collision + OBJ visual).

        Each sub-scene (parallel environment) gets its own random decision using _batched_episode_rng.
        """
        # Use batched episode RNG to make consistent decisions across different numbers of parallel environments
        random_values = self._batched_episode_rng.random()
        use_real_b5box_decisions = random_values < self.use_real_b5box_probability

        # Create separate objects for each sub-scene based on their individual decisions
        real_b5box_objects = []
        primitive_b5box_objects = []

        # Store per-scene box dimensions
        self.per_scene_box_half_sizes = []

        for i in range(self.num_envs):
            if bool(use_real_b5box_decisions[i]):
                # Create real b5box for this sub-scene
                real_obj, real_box_dims = self._load_real_b5box_for_scene(i)
                real_b5box_objects.append(real_obj)
                primitive_b5box_objects.append(None)  # placeholder
                self.per_scene_box_half_sizes.append(real_box_dims)
            else:
                # Create primitive box for this sub-scene
                primitive_obj, primitive_box_dims = self._load_primitive_b5box_for_scene(i)
                primitive_b5box_objects.append(primitive_obj)
                real_b5box_objects.append(None)  # placeholder
                self.per_scene_box_half_sizes.append(primitive_box_dims)

        # Merge all objects (filtering out None placeholders)
        all_objects = [obj for obj in real_b5box_objects + primitive_b5box_objects if obj is not None]

        if self.verbose:
            real_count = sum(use_real_b5box_decisions)
            primitive_count = self.num_envs - real_count
            print(f"    B5box asset distribution: {real_count} real, {primitive_count} primitive")
            print("    Both asset types now have physics variation (friction, mass, damping, scale)")

        return Actor.merge(all_objects, name="cube")

    def _load_real_b5box_for_scene(self, scene_idx: int):
        """Load the actual b5box object using STL collision file and OBJ visual (matching bimanual approach)."""
        builder = self.scene.create_actor_builder()

        # Apply physics randomization similar to primitive box
        # Scale randomization (±10% variation)
        scale_multiplier = self._batched_episode_rng[scene_idx].uniform(0.9, 1.1)
        scale = [scale_multiplier, scale_multiplier, scale_multiplier]

        # Friction randomization
        scene_friction_multiplier = 0.8 + 0.4 * self._batched_episode_rng[scene_idx].random()  # [0.8, 1.2]
        randomized_static_friction = self.base_static_friction * scene_friction_multiplier
        randomized_dynamic_friction = self.base_dynamic_friction * scene_friction_multiplier

        box_material = physx.PhysxMaterial(
            static_friction=randomized_static_friction,
            dynamic_friction=randomized_dynamic_friction,
            restitution=0.1,
        )

        # Mass/density randomization
        mass_multiplier = 0.8 + 0.4 * self._batched_episode_rng[scene_idx].random()  # [0.8, 1.2]
        randomized_density = 500 * mass_multiplier  # Base density from bimanual environment

        # Use STL file for collision (with randomized parameters)
        collision_file = str(self.asset_root / "b5box/b5box_collision.stl")
        builder.add_convex_collision_from_file(
            filename=collision_file,
            scale=scale,
            density=randomized_density,
            material=box_material,
        )

        # Use visual mesh for rendering (with matching scale)
        visual_file = str(self.asset_root / "b5box/b5box.obj")
        builder.add_visual_from_file(
            filename=visual_file,
            scale=scale,
        )

        # Set initial pose to avoid warnings (will be repositioned in episode init)
        builder.set_initial_pose(sapien.Pose(p=[0, 0, self.b5box_half_size]))

        # Set scene index for this specific sub-scene
        builder.set_scene_idxs([scene_idx])

        # Build dynamic object
        cube = builder.build(name=f"cube_{scene_idx}")

        # Apply damping randomization after creation (like primitive box)
        damping_multiplier = 0.8 + 0.4 * self._batched_episode_rng[scene_idx].random()  # [0.8, 1.2]
        randomized_damping = self.base_linear_damping * damping_multiplier
        cube.linear_damping = randomized_damping

        # Real b5box approximate dimensions scaled by the randomization
        # The real b5box is approximately 5cm x 2.5cm x 1.5cm (half-sizes)
        base_real_box_half_sizes = [0.0625, 0.016, 0.024]  # Base dimensions
        real_box_half_sizes = [
            base_real_box_half_sizes[0] * scale_multiplier,
            base_real_box_half_sizes[1] * scale_multiplier,
            base_real_box_half_sizes[2] * scale_multiplier,
        ]

        if self.verbose:
            print("    Using REAL B5BOX ASSETS (with physics variation):")
            print("      Collision file: b5box_collision.stl")
            print("      Visual file: b5box.obj")
            print(f"      Scale multiplier: {scale_multiplier:.3f}")
            print(f"      Friction multiplier: {scene_friction_multiplier:.3f} (narrowed)")
            print(f"      Static friction: {randomized_static_friction:.3f}")
            print(f"      Dynamic friction: {randomized_dynamic_friction:.3f}")
            print(f"      Mass multiplier: {mass_multiplier:.3f} (narrowed)")
            print(f"      Density: {randomized_density:.1f} kg/m³")
            print(f"      Damping multiplier: {damping_multiplier:.3f} (narrowed)")
            print(f"      Damping: {randomized_damping:.4f}")
            print(f"      Scaled half-sizes: {real_box_half_sizes}")

        return cube, real_box_half_sizes

    def _load_primitive_b5box_for_scene(self, scene_idx: int):
        """Create a primitive red box with x-axis 3x longer than other dimensions (original approach)."""
        # Apply uniform sampling for 10% +/- variability on each dimension using batched episode RNG
        x_multiplier = self._batched_episode_rng[scene_idx].uniform(0.9, 1.2)
        y_multiplier = self._batched_episode_rng[scene_idx].uniform(0.9, 1.2)
        z_multiplier = self._batched_episode_rng[scene_idx].uniform(0.9, 1.2)

        box_half_sizes = [
            3 * self.b5box_half_size * x_multiplier,  # x-axis: 3x longer with variation
            self.b5box_half_size * y_multiplier,  # y-axis: same as original with variation
            self.b5box_half_size * z_multiplier,  # z-axis: same as original with variation
        ]  # [width, depth, height] half-sizes - longer in x

        # Create box with base physical material (friction will be randomized per scene)
        # Apply some friction randomization at the scene level using batched episode RNG
        scene_friction_multiplier = 0.5 + self._batched_episode_rng[scene_idx].random()  # [0.5, 1.5]
        randomized_static_friction = self.base_static_friction * scene_friction_multiplier
        randomized_dynamic_friction = self.base_dynamic_friction * scene_friction_multiplier

        box_material = physx.PhysxMaterial(
            static_friction=randomized_static_friction,
            dynamic_friction=randomized_dynamic_friction,
            restitution=0.1,
        )

        # Print dimension info for first few scenes to verify randomization (only if verbose)
        if self.verbose and scene_idx < 5:  # Print for first 5 scenes
            print(f"Scene {scene_idx} box dimensions:")
            print(f"  Multipliers - X: {x_multiplier:.3f}, Y: {y_multiplier:.3f}, Z: {z_multiplier:.3f}")
            print(f"  Final half-sizes: [{box_half_sizes[0]:.4f}, {box_half_sizes[1]:.4f}, {box_half_sizes[2]:.4f}]")

        if self.verbose:
            print(f"    Using PRIMITIVE BOX for scene {scene_idx}:")
            print(f"      Scene-level friction multiplier: {scene_friction_multiplier:.3f}")
            print(f"      Scene static friction: {randomized_static_friction:.3f}")
            print(f"      Scene dynamic friction: {randomized_dynamic_friction:.3f}")
            print("      Dimensional variation multipliers:")
            print(f"        X-axis (length): {x_multiplier:.3f}")
            print(f"        Y-axis (width): {y_multiplier:.3f}")
            print(f"        Z-axis (height): {z_multiplier:.3f}")
            print(f"      Final box half-sizes: {box_half_sizes}")

        # Apply mass/density randomization at creation time (±50% variation)
        mass_multiplier = 0.5 + self._batched_episode_rng[scene_idx].random()  # [0.5, 1.5]
        randomized_density = self.base_density * mass_multiplier

        # Generate random box color using batched episode RNG
        box_color = [
            self._batched_episode_rng[scene_idx].uniform(*self.box_color_ranges["red"]),
            self._batched_episode_rng[scene_idx].uniform(*self.box_color_ranges["green"]),
            self._batched_episode_rng[scene_idx].uniform(*self.box_color_ranges["blue"]),
            self._batched_episode_rng[scene_idx].uniform(*self.box_color_ranges["alpha"]),
        ]

        if self.verbose:
            print(
                f"      Box color: RGBA({box_color[0]:.3f}, {box_color[1]:.3f}, {box_color[2]:.3f}, {box_color[3]:.3f})"
            )

        builder = self.scene.create_actor_builder()
        builder.add_box_collision(
            half_size=box_half_sizes,
            material=box_material,
            density=randomized_density,
        )
        builder.add_box_visual(
            half_size=box_half_sizes,
            material=box_color,  # Use random color instead of hardcoded red
        )
        builder.set_initial_pose(sapien.Pose(p=[0, 0, box_half_sizes[2]]))

        # Set scene index for this specific sub-scene
        builder.set_scene_idxs([scene_idx])

        cube = builder.build(name=f"cube_{scene_idx}")

        # Apply damping randomization after creation but before GPU initialization
        damping_multiplier = 0.5 + self._batched_episode_rng[scene_idx].random()  # [0.5, 1.5]
        randomized_damping = self.base_linear_damping * damping_multiplier
        cube.linear_damping = randomized_damping

        if self.verbose:
            print(f"      Creation-time mass multiplier: {mass_multiplier:.3f}")
            print(f"      Creation-time density: {randomized_density:.1f} kg/m³")
            print(f"      Creation-time damping multiplier: {damping_multiplier:.3f}")
            print(f"      Creation-time damping: {randomized_damping:.4f}")

        return cube, box_half_sizes

    def _load_basket(self):
        builder = self.scene.create_actor_builder()
        # Bottom
        builder.add_box_collision(
            pose=sapien.Pose(p=[0, 0, 0]),
            half_size=[0.12, 0.09, 0.004],
        )
        # Walls (south/north/east/west)
        builder.add_box_collision(pose=sapien.Pose(p=[-0.13, 0, 0.06]), half_size=[0.005, 0.1, 0.05])
        builder.add_box_collision(pose=sapien.Pose(p=[0.13, 0, 0.06]), half_size=[0.005, 0.1, 0.05])
        builder.add_box_collision(pose=sapien.Pose(p=[0.0, 0.1, 0.06]), half_size=[0.12, 0.005, 0.05])
        builder.add_box_collision(pose=sapien.Pose(p=[0.0, -0.1, 0.06]), half_size=[0.12, 0.005, 0.05])
        visual_file = str(self.asset_root / "basket/basket.obj")
        builder.add_visual_from_file(filename=visual_file, scale=[1.0, 1.0, 1.0])
        builder.set_initial_pose(sapien.Pose(p=[-0.2, 0, 0.01]))
        return builder.build_kinematic(name="basket")

    def _create_goal_site(self):
        goal_site = actors.build_sphere(
            self.scene,
            radius=0.02,
            color=[0, 1, 0, 0.0],  # Set alpha to 0.0 to make completely transparent
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(goal_site)
        return goal_site

    def _create_shadow_box(self):
        """Create a floating rectangular box above the scene to cast shadows."""
        # Check if we should create a shadow box (70% chance)
        if np.random.random() > self.shadow_box_ranges["probability"]:
            if self.verbose:
                print("    Shadow box: Not created (random variation)")
            return None

        # Generate random dimensions
        width = np.random.uniform(*self.shadow_box_ranges["width"])
        depth = np.random.uniform(*self.shadow_box_ranges["depth"])
        height = np.random.uniform(*self.shadow_box_ranges["height"])

        # Get table surface height
        scene_cfg = BIMANUAL_PICK_PLACE_CONFIG["scene"]
        table_surface_z = scene_cfg["table"]["pose"].p[..., 2] + scene_cfg["table"]["size"][2]
        if hasattr(table_surface_z, "item"):  # Handle tensor case
            table_surface_z = table_surface_z.item()

        # Generate random position relative to table surface
        x_pos = np.random.uniform(*self.shadow_box_ranges["x_position"])
        y_pos = np.random.uniform(*self.shadow_box_ranges["y_position"])
        z_height_above_table = np.random.uniform(*self.shadow_box_ranges["z_height"])
        z_pos = float(table_surface_z + z_height_above_table)  # Position relative to table surface

        # Generate random rotation around Z-axis
        rotation_z = np.random.uniform(*self.shadow_box_ranges["rotation"])

        # Create a semi-transparent dark box
        shadow_color = [0.2, 0.2, 0.2, 0.6]  # Dark gray, semi-transparent

        builder = self.scene.create_actor_builder()
        builder.add_box_visual(half_size=[width / 2, depth / 2, height / 2], material=shadow_color)

        # Set pose with random rotation - ensure all values are float
        quat = [
            float(np.cos(rotation_z / 2)),
            0.0,
            0.0,
            float(np.sin(rotation_z / 2)),
        ]  # Rotation around Z-axis
        builder.set_initial_pose(sapien.Pose(p=[float(x_pos), float(y_pos), float(z_pos)], q=quat))

        shadow_box = builder.build_kinematic(name="shadow_box")

        if self.verbose:
            print("    Shadow box created:")
            print(f"      Dimensions: {width:.3f} x {depth:.3f} x {height:.3f}")
            print(f"      Position: ({x_pos:.3f}, {y_pos:.3f}, {z_pos:.3f})")
            print(f"      Height above table: {z_height_above_table:.3f}m")
            print(f"      Rotation: {rotation_z:.3f} rad ({rotation_z * 180 / np.pi:.1f}°)")

        # Add to hidden objects so it doesn't interfere with task logic
        self._hidden_objects.append(shadow_box)
        return shadow_box

    # ------------------------------------------------------------------
    # Episode initialisation -------------------------------------------
    # ------------------------------------------------------------------

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)

            if self.verbose and b > 0:
                print(f"\nInitializing episode for {b} environments...")
                print(f"  Environment indices: {env_idx[: min(5, b)].tolist()}")  # Show first 5

            # Apply curriculum level configuration if provided
            # curriculum_config = self._apply_curriculum_config(options)

            # Home pose ---------------------------------------------------
            if self.bimanual:
                # Set both arms to their respective home positions
                cfg = PICK_BOX_CONFIGS[self.robot_type]
                left_qpos = cfg["left_arm"]["home_qpos"]
                right_qpos = cfg["right_arm"]["home_qpos"]

                # Set left arm to home position (static)
                self.left_agent.robot.set_qpos(torch.tensor(left_qpos, device=self.device).repeat(b, 1))
                # Set right arm to home position (active)
                self.right_agent.robot.set_qpos(torch.tensor(right_qpos, device=self.device).repeat(b, 1))
            else:
                # Single arm mode - use right arm configuration
                cfg = PICK_BOX_CONFIGS[self.robot_type]
                right_qpos = cfg["right_arm"]["home_qpos"]
                self.active_agent.robot.set_qpos(torch.tensor(right_qpos, device=self.device).repeat(b, 1))

            # Scene parameters ------------------------------------------
            scene_cfg = BIMANUAL_PICK_PLACE_CONFIG["scene"]
            # Pose.p is of shape (num_envs, 3); take the z-component robustly
            table_surface_z = scene_cfg["table"]["pose"].p[..., 2] + scene_cfg["table"]["size"][2]

            # Randomise b5box position ----------------------------------
            box_spawn = scene_cfg["b5box"]["spawn_region"]
            box_xyz = torch.zeros((b, 3))
            box_xyz[:, 0] = box_spawn["center"][0] + (torch.rand((b,)) * 2 - 1) * box_spawn["half_size"][0]
            box_xyz[:, 1] = box_spawn["center"][1] + (torch.rand((b,)) * 2 - 1) * box_spawn["half_size"][1]
            # Use actual box height from per-scene box dimensions
            if self.per_scene_box_half_sizes is not None:
                # Use per-scene box dimensions
                for i, env_id in enumerate(env_idx):
                    actual_box_height = self.per_scene_box_half_sizes[env_id][2]  # Z-dimension half-size
                    box_xyz[i, 2] = table_surface_z + actual_box_height
            else:
                # Fallback to default (shouldn't happen in normal operation)
                actual_box_height = self.box_half_sizes[2]  # Z-dimension half-size
                box_xyz[:, 2] = table_surface_z + actual_box_height
            box_qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.b5box.set_pose(Pose.create_from_pq(box_xyz, box_qs))

            # Fixed basket position (no randomization) ----------------------------------
            basket_spawn = scene_cfg["basket"]["spawn_region"]
            basket_xyz = torch.zeros((b, 3))
            # Use center position without randomization
            basket_xyz[:, 0] = basket_spawn["center"][0]
            basket_xyz[:, 1] = basket_spawn["center"][1]
            basket_xyz[:, 2] = table_surface_z + 0.004  # basket bottom height (lowered by 5cm)
            # 90° rotation around Z (match original)
            z_rot = torch.tensor(np.pi / 2, device=self.device)
            basket_qs = torch.zeros((b, 4), device=self.device)
            basket_qs[:, 0] = torch.cos(z_rot / 2)
            basket_qs[:, 3] = torch.sin(z_rot / 2)
            self.basket.set_pose(Pose.create_from_pq(basket_xyz, basket_qs))

            # Goal site for visualization ------------------------------
            goal_xyz = basket_xyz.clone()
            goal_xyz[:, 2] = table_surface_z + 0.10
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

            # Per-episode visual randomization (table color and lighting)
            self._randomize_table_color_per_episode(env_idx)
            self._randomize_lighting_per_episode(env_idx)

            # Initialize transport timer for reward decay mechanism
            if not hasattr(self, "transport_timer"):
                self.transport_timer = torch.zeros(self.num_envs, device=self.device)
            # Reset transport timer for environments being reset
            self.transport_timer[env_idx] = 0.0

            # Physical properties (mass, damping, friction) are randomized at creation time
            # since they cannot be modified after GPU simulation initialization

            if self.verbose and b > 0:
                print(f"  ✓ Episode initialized for {b} environments")
                print(f"    Mode: {'Bimanual' if self.bimanual else 'Single-arm (right)'}")
                print(f"    Box position (first env): {box_xyz[0]}")
                print(f"    Basket position (first env): {basket_xyz[0]}")
                print(f"    Goal position (first env): {goal_xyz[0]}")
                if self.per_scene_box_half_sizes is not None:
                    print(f"    Box half-sizes for env 0: {self.per_scene_box_half_sizes[env_idx[0]]}")
                    expected_z = table_surface_z + self.per_scene_box_half_sizes[env_idx[0]][2]
                else:
                    print(f"    Box half-sizes for env 0: {self.box_half_sizes}")
                    expected_z = table_surface_z + self.box_half_sizes[2]

                print(f"    Box z-placement: {box_xyz[0][2]:.4f} (should be table + box height)")
                print(
                    f"    Expected box z-placement: {expected_z.item() if hasattr(expected_z, 'item') else expected_z:.4f}"
                )
                print(f"    Lighting direction (fixed): {self._env_lighting_directions}")

                if self.bimanual:
                    print("    Left arm: Static at home position")
                    print("    Right arm: Active (controlled by agent)")

    def _randomize_table_color_per_episode(self, env_idx: torch.Tensor):
        """Randomize table color for each episode reset."""
        # Generate random table color using proper random number generator
        # Use the first environment's RNG for simplicity since table is shared
        if len(env_idx) > 0:
            first_env_idx = env_idx[0].item()
            rng = self._batched_episode_rng[first_env_idx]

            table_color = [
                rng.uniform(*self.table_color_ranges["red"]),
                rng.uniform(*self.table_color_ranges["green"]),
                rng.uniform(*self.table_color_ranges["blue"]),
                rng.uniform(*self.table_color_ranges["alpha"]),
            ]

            # Update table visual appearance using the correct SAPIEN approach
            try:
                # Access the underlying SAPIEN actors from the ManiSkill Actor wrapper
                for sapien_actor in self.table._objs:
                    # Get the render body component
                    render_body = sapien_actor.find_component_by_type(sapien.render.RenderBodyComponent)
                    if render_body:
                        # Get render shapes from the render body
                        render_shapes = render_body.render_shapes
                        if render_shapes:
                            for shape in render_shapes:
                                # Update the material's base color
                                material = shape.material
                                material.set_base_color(table_color)

                if self.verbose:
                    print(
                        f"    Table color randomized: RGBA({table_color[0]:.3f}, {table_color[1]:.3f}, {table_color[2]:.3f}, {table_color[3]:.3f})"
                    )
            except Exception as e:
                if self.verbose:
                    print(f"    Warning: Could not update table color: {e}")
                    print(f"    Table object type: {type(self.table)}")
                    if hasattr(self.table, "_objs"):
                        print(
                            f"    Underlying SAPIEN actor type: {type(self.table._objs[0]) if self.table._objs else 'None'}"
                        )
                    else:
                        print(f"    Table attributes: {dir(self.table)[:10]}...")  # Show first 10 attributes

    def _randomize_lighting_per_episode(self, env_idx: torch.Tensor):
        """Randomize lighting for each episode reset."""
        if len(env_idx) > 0:
            first_env_idx = env_idx[0].item()
            rng = self._batched_episode_rng[first_env_idx]

            # Generate random ambient light intensity
            ambient_intensity = rng.uniform(
                self.ambient_light_ranges["min_intensity"][0],
                self.ambient_light_ranges["max_intensity"][1],
            )

            ambient_color = [ambient_intensity, ambient_intensity, ambient_intensity]

            # Set ambient light - this works reliably without accumulation
            self.scene.set_ambient_light(ambient_color)

            if self.verbose:
                print("    Lighting randomized per episode:")
                print(f"      Ambient: {ambient_color}")
                print("      ✓ Applied ambient light successfully")
                print("      Note: Directional light remains fixed to prevent accumulation issues")

    # ------------------------------------------------------------------
    # Observation extras -----------------------------------------------
    # ------------------------------------------------------------------

    def _get_obs_extra(self, info: Dict):
        b = len(self.scene.sub_scenes)

        # Calculate gripper x-axis in world frame
        local_x_axis = torch.tensor([1, 0, 0], device=self.device).expand(b, 3)
        gripper_x_axis = quaternion_apply(self.active_agent.tcp_pose.q, local_x_axis)

        # Calculate box x-axis in world frame
        box_x_axis = quaternion_apply(self.b5box.pose.q, local_x_axis)

        # Calculate angle between box and gripper x-axes
        box_x_axis_norm = box_x_axis / torch.linalg.norm(box_x_axis, dim=-1, keepdim=True)
        gripper_x_axis_norm = gripper_x_axis / torch.linalg.norm(gripper_x_axis, dim=-1, keepdim=True)
        dot_product = torch.sum(box_x_axis_norm * gripper_x_axis_norm, dim=-1, keepdim=True)
        dot_product = torch.clamp(dot_product, -1.0, 1.0)
        angle_between_axes = torch.acos(dot_product)

        # Apply table origin transformations if enabled
        tcp_pose = self._transform_pose_to_table_origin(self.active_agent.tcp_pose.raw_pose)
        goal_pos = self._transform_position_to_table_origin(self.goal_site.pose.p)
        obj_pose = self._transform_pose_to_table_origin(self.b5box.pose.raw_pose)
        tcp_to_obj_pos = self._transform_position_to_table_origin(
            self.b5box.pose.p
        ) - self._transform_position_to_table_origin(self.active_agent.tcp_pose.p)
        obj_to_goal_pos = goal_pos - self._transform_position_to_table_origin(self.b5box.pose.p)

        obs = dict(
            is_grasped=info["is_grasped"],
            tcp_pose=tcp_pose,
            goal_pos=goal_pos,
            obj_pose=obj_pose,
            tcp_to_obj_pos=tcp_to_obj_pos,
            obj_to_goal_pos=obj_to_goal_pos,
            box_x_axis_in_world=self._get_box_axes_in_world(),
            gripper_x_axis=gripper_x_axis,
            angle_between_axes=angle_between_axes,
            is_obj_stable=info["is_obj_stable"],
        )

        # Add bimanual-specific observations if in bimanual mode
        if self.bimanual:
            left_tcp_pose = self._transform_pose_to_table_origin(self.left_agent.tcp_pose.raw_pose)
            right_tcp_pose = self._transform_pose_to_table_origin(self.right_agent.tcp_pose.raw_pose)
            left_tcp_to_obj_pos = self._transform_position_to_table_origin(
                self.b5box.pose.p
            ) - self._transform_position_to_table_origin(self.left_agent.tcp_pose.p)
            right_tcp_to_obj_pos = self._transform_position_to_table_origin(
                self.b5box.pose.p
            ) - self._transform_position_to_table_origin(self.right_agent.tcp_pose.p)

            obs.update({
                "left_tcp_pose": left_tcp_pose,
                "right_tcp_pose": right_tcp_pose,
                "left_tcp_to_obj_pos": left_tcp_to_obj_pos,
                "right_tcp_to_obj_pos": right_tcp_to_obj_pos,
            })

        return obs

    def _get_box_axes_in_world(self):
        """Get the box's local axes (X, Y, Z) expressed in world coordinates.

        This helps the agent understand the box's orientation for better grasping.
        Returns: tensor of shape (num_envs, 9) representing [x_axis, y_axis, z_axis]
        """
        box_pose = self.b5box.pose

        # Define local axes
        x_axis = torch.tensor([1, 0, 0], device=self.device).expand(len(self.scene.sub_scenes), 3)

        # Transform to world frame using quaternion
        world_x_axis = quaternion_apply(box_pose.q, x_axis)  # Longest dimension

        return world_x_axis

    # ------------------------------------------------------------------
    # Task evaluation & reward -----------------------------------------
    # ------------------------------------------------------------------

    def evaluate(self):
        # Proximity check
        is_obj_placed = torch.linalg.norm(self.goal_site.pose.p - self.b5box.pose.p, axis=1) <= self.goal_thresh
        is_grasped = self.active_agent.is_grasping(self.b5box)
        is_robot_static = self.active_agent.is_static(0.2)

        # Check if object is stable (not moving much)
        obj_velocity = self.b5box.linear_velocity
        obj_angular_velocity = self.b5box.angular_velocity
        obj_linear_speed = torch.linalg.norm(obj_velocity, axis=1)
        obj_angular_speed = torch.linalg.norm(obj_angular_velocity, axis=1)

        # Object is stable if both linear and angular velocities are low
        is_obj_stable = (obj_linear_speed <= 0.1) & (obj_angular_speed <= 0.5)  # 0.1 m/s, 0.5 rad/s

        # Success: object is placed, released, and stable
        return {
            "success": is_obj_placed & ~is_grasped & is_obj_stable,
            "is_obj_placed": is_obj_placed,
            "is_robot_static": is_robot_static,  # Keep for debugging/analysis
            "is_grasped": is_grasped,
            "is_obj_stable": is_obj_stable,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_to_obj_dist = torch.linalg.norm(self.b5box.pose.p - self.active_agent.tcp_pose.p, axis=1)

        # TCP-to-object proximity reward throughout task, except after releasing
        # Stop rewarding TCP proximity when object is placed AND not grasped (i.e., after release)
        is_grasped = info["is_grasped"]
        is_obj_placed = info["is_obj_placed"]
        should_reward_tcp_proximity = ~(is_obj_placed & ~is_grasped)  # NOT (placed AND not grasped)

        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward * should_reward_tcp_proximity.float()

        # Binary grasp reward
        reward += is_grasped.float()

        # Lifting reward: high reward for lifting the box above the table
        # Get table surface height
        scene_cfg = BIMANUAL_PICK_PLACE_CONFIG["scene"]
        table_pose_z = torch.as_tensor(scene_cfg["table"]["pose"].p[..., 2], device=self.device)
        table_size_z = torch.as_tensor(scene_cfg["table"]["size"][2], device=self.device)
        table_surface_z = table_pose_z + table_size_z

        # Calculate how high the box is above the table surface
        box_height_above_table = self.b5box.pose.p[:, 2] - table_surface_z
        # Give lifting reward when grasped and lifted (scale by 0.05m = 5cm max lift)
        lifting_reward = torch.clamp(box_height_above_table / 0.05, 0.0, 1.0)
        reward += lifting_reward * is_grasped.float() * 2.0  # 2x weight for lifting

        # Check if box is lifted above 3cm threshold
        is_lifted_above_threshold = box_height_above_table >= 0.03  # 3cm threshold

        # Transport / place reward
        obj_to_goal_dist = torch.linalg.norm(self.goal_site.pose.p - self.b5box.pose.p, axis=1)
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)

        # During transport phase: reward only when grasped AND lifted above 3cm
        transport_reward = place_reward * is_grasped.float() * is_lifted_above_threshold.float()

        # Decay transport reward if held too long in place (Option 2)
        if not hasattr(self, "transport_timer"):
            self.transport_timer = torch.zeros(self.num_envs, device=self.device)

        is_transporting = is_grasped & is_lifted_above_threshold & (obj_to_goal_dist < 0.05)
        self.transport_timer = torch.where(
            is_transporting,
            self.transport_timer + 1,
            torch.zeros_like(self.transport_timer),
        )

        # Decay transport reward after holding for too long (20 steps = ~0.4 seconds at 50Hz)
        transport_decay = torch.clamp(1.0 - self.transport_timer / 20.0, 0.1, 1.0)  # Decay over 20 steps
        transport_reward = transport_reward * transport_decay
        reward += transport_reward

        # Penalize holding object still when it should be released (Option 1)
        is_near_goal = obj_to_goal_dist < (self.goal_thresh * 2.0)
        is_holding_still = torch.linalg.norm(self.b5box.linear_velocity, axis=1) < 0.1  # Very still
        holding_still_penalty = 0.5 * is_grasped.float() * is_near_goal.float() * is_holding_still.float()
        reward -= holding_still_penalty

        # After release phase: continue rewarding good placement even when not grasped
        # This ensures the robot doesn't lose reward for releasing the object in the right place
        post_release_reward = place_reward * (~is_grasped).float() * is_lifted_above_threshold.float()
        reward += post_release_reward * 0.5  # Lower weight than transport reward

        # X-axis alignment reward: reward alignment of box and gripper x-axes
        # Calculate gripper x-axis in world frame
        local_x_axis = torch.tensor([1, 0, 0], device=self.device).expand(len(self.scene.sub_scenes), 3)
        gripper_x_axis = quaternion_apply(self.active_agent.tcp_pose.q, local_x_axis)
        # Calculate box x-axis in world frame
        box_x_axis = quaternion_apply(self.b5box.pose.q, local_x_axis)
        # Calculate angle between axes
        box_x_axis_norm = box_x_axis / torch.linalg.norm(box_x_axis, dim=-1, keepdim=True)
        gripper_x_axis_norm = gripper_x_axis / torch.linalg.norm(gripper_x_axis, dim=-1, keepdim=True)
        dot_product = torch.sum(box_x_axis_norm * gripper_x_axis_norm, dim=-1)
        dot_product = torch.clamp(dot_product, -1.0, 1.0)
        angle_between_axes = torch.acos(dot_product)
        # Handle flipped axes by taking min(angle, π - angle) for alignment reward
        alignment_angle = torch.min(angle_between_axes, torch.pi - angle_between_axes)
        alignment_reward = 1 - torch.tanh(5 * alignment_angle)
        reward += 0.1 * alignment_reward * should_reward_tcp_proximity.float()

        # Gripper openness reward: encourage keeping gripper open when far from object
        # Get gripper joint positions (assuming they're at the end of qpos)
        robot_qpos = self.active_agent.robot.get_qpos()
        if self.robot_uids in ["panda", "panda_wristcam"]:
            # For Panda robots, gripper joints are the last 2 joints
            gripper_qpos = robot_qpos[..., -2:]  # [left_finger, right_finger]
            # Gripper openness is the sum of finger positions (more open = higher values)
            gripper_openness = torch.sum(gripper_qpos, dim=-1)
            max_gripper_openness = 0.08  # Panda gripper max opening (~4cm per finger)
        elif self.robot_uids == "a1_galaxea":
            # A1 Galaxea gripper is *flipped*: open ≈ -0.01 rad, closed ≈ +0.03 rad (per finger).
            # Let s = sum(qpos_last_two).  We map s=0.06 → 0 (fully closed) and s=-0.02 →1 (fully open).
            gripper_qpos = robot_qpos[..., -2:]
            gripper_qpos_sum = torch.sum(gripper_qpos, dim=-1)

            closed_sum = 0.06  # 0.03 + 0.03  (fully closed)
            open_sum = -0.02  # -0.01 + -0.01 (fully open)
            range_sum = closed_sum - open_sum  # 0.08

            gripper_openness = torch.clamp((closed_sum - gripper_qpos_sum) / range_sum, 0.0, 1.0)
            # gripper_openness is *already normalized* ∈[0,1]
            normalized_gripper_openness = gripper_openness  # skip re-normalizing below
            max_gripper_openness = 1.0  # since already scaled
        else:
            # Fallback for other robots
            gripper_qpos = robot_qpos[..., -2:]
            gripper_openness = torch.sum(gripper_qpos, dim=-1)
            max_gripper_openness = 0.08

        # Normalize (for non-A1 robots)
        if self.robot_uids != "a1_galaxea":
            normalized_gripper_openness = torch.clamp(gripper_openness / max_gripper_openness, 0.0, 1.0)

        # Gripper reward logic based on context
        # Check if object is near goal and at good height (ready for release)
        is_near_goal = obj_to_goal_dist < (self.goal_thresh * 2.0)  # 2x goal threshold
        is_at_good_height = box_height_above_table >= 0.05  # 5cm above table
        ready_for_release = is_near_goal & is_at_good_height

        # RELEASE REWARD: Encourage releasing when object is well-positioned
        # This is the key addition - reward for opening gripper when near goal
        release_reward = 3.0 * normalized_gripper_openness * ready_for_release.float()
        reward += release_reward

        # CONTINUED GRASP PENALTY: Penalize keeping gripper closed when ready for release
        continued_grasp_penalty = (1.0 - normalized_gripper_openness) * 0.6
        reward -= continued_grasp_penalty * ready_for_release.float()

        # STANDARD GRIPPER REWARDS (when not ready for release)
        not_ready_for_release = ~ready_for_release

        # Reward open gripper when NOT grasping and NOT ready for release
        should_reward_open_gripper = ~is_grasped & not_ready_for_release
        open_gripper_reward = normalized_gripper_openness * 0.02
        reward += open_gripper_reward * should_reward_open_gripper.float()

        # Penalty for closing gripper when far from object and not grasping
        should_penalize_close_gripper = ~is_grasped & not_ready_for_release
        premature_close_penalty = (1.0 - normalized_gripper_openness) * 0.2
        reward -= premature_close_penalty * should_penalize_close_gripper.float()

        # Object stability bonus when placed and released
        # Reward when object is stable (not moving much) in the goal location
        stability_reward = 1 - torch.tanh(5 * torch.linalg.norm(self.b5box.linear_velocity, axis=1))
        reward += stability_reward * info["is_obj_placed"].float() * (~is_grasped).float()

        # Debug output for release reward (removed for performance)

        # Success bonus
        reward[info["success"]] = 5.0
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # Updated max reward calculation with new release reward components:
        # - reaching_reward: 1.0 (when approaching object)
        # - is_grasped: 1.0 (grasp bonus)
        # - lifting_reward: 2.0 (when grasped and lifted)
        # - transport_reward: 1.0 (when grasped and lifted)
        # - post_release_reward: 0.5 (when not grasped and lifted)
        # - alignment_reward: 0.1 (when should_reward_tcp_proximity)
        # - release_reward: 3.0 (when ready for release)
        # - stability_reward: 1.0 (when object is stable and placed after release)
        # - success_bonus: 5.0
        # - holding_still_penalty: -0.5 (when grasped, near goal, and holding still)
        # Total max positive reward: 1.0 + 1.0 + 2.0 + 1.0 + 0.5 + 0.1 + 3.0 + 1.0 + 5.0 = 14.6
        # Note: holding_still_penalty reduces reward when robot stabilizes while grasped
        return self.compute_dense_reward(obs, action, info) / 14.6

    def inspect_observation_space(self, verbose=True):
        """Inspect the observation space structure and validate all features are present.

        Args:
            verbose (bool): If True, print detailed information about each observation component.

        Returns:
            dict: Dictionary containing observation space information
        """
        # Reset environment to get a sample observation
        obs, info = self.reset()

        # Get current observation by taking a step
        action = self.action_space.sample()
        obs, reward, terminated, truncated, info = self.step(action)

        obs_info = {
            "obs_mode": "state",
            "total_features": 0,
            "components": {},
        }

        if verbose:
            print("=== Observation Space Inspection ===")
            print("Observation Mode: state")
            print(f"Environment: {self.__class__.__name__}")
            print(f"Robot: {self.robot_uids}")
            print(f"Device: {self.device}")
            print()

        # Inspect each observation component
        if isinstance(obs, dict):
            # Handle dictionary observations (state_dict or visual modes)
            for key, value in obs.items():
                if isinstance(value, torch.Tensor):
                    shape = tuple(value.shape)
                    dtype = value.dtype
                    feature_count = value.numel() // len(self.scene.sub_scenes)

                    obs_info["components"][key] = {
                        "shape": shape,
                        "dtype": str(dtype),
                        "features_per_env": feature_count,
                        "total_features": value.numel(),
                    }
                    obs_info["total_features"] += feature_count

                    if verbose:
                        print(f"  {key:20s}: {shape} ({dtype}) - {feature_count} features/env")
                elif isinstance(value, dict):
                    # Handle nested dictionaries (like 'extra' or 'agent')
                    if verbose:
                        print(f"  {key:20s}: Dictionary with {len(value)} sub-components")
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, torch.Tensor):
                            shape = tuple(sub_value.shape)
                            dtype = sub_value.dtype
                            feature_count = sub_value.numel() // len(self.scene.sub_scenes)

                            full_key = f"{key}.{sub_key}"
                            obs_info["components"][full_key] = {
                                "shape": shape,
                                "dtype": str(dtype),
                                "features_per_env": feature_count,
                                "total_features": sub_value.numel(),
                            }
                            obs_info["total_features"] += feature_count

                            if verbose:
                                print(f"    {sub_key:18s}: {shape} ({dtype}) - {feature_count} features/env")
                        elif verbose:
                            print(f"    {sub_key:18s}: {type(sub_value).__name__}")
                else:
                    obs_info["components"][key] = {
                        "type": type(value).__name__,
                        "value": value
                        if not hasattr(value, "__len__") or len(str(value)) < 100
                        else f"{type(value).__name__}(...)",
                    }
                    if verbose:
                        print(f"  {key:20s}: {type(value).__name__}")
        else:
            # Handle tensor observations (flattened state mode)
            shape = tuple(obs.shape)
            dtype = obs.dtype
            feature_count = obs.numel() // len(self.scene.sub_scenes)

            obs_info["components"]["flattened_state"] = {
                "shape": shape,
                "dtype": str(dtype),
                "features_per_env": feature_count,
                "total_features": obs.numel(),
            }
            obs_info["total_features"] = feature_count

            if verbose:
                print(f"  flattened_state    : {shape} ({dtype}) - {feature_count} features/env")

        if verbose:
            print(f"\nTotal Features per Environment: {obs_info['total_features']}")
            print()

        # Validate expected features based on obs_mode
        expected_features = self._get_expected_features()
        missing_features = []

        for feature_name, expected_count in expected_features.items():
            # Handle both direct keys and nested keys (with dot notation)
            found = False
            actual_count = 0

            # Check direct key first
            if feature_name in obs_info["components"]:
                found = True
                actual_count = obs_info["components"][feature_name].get("features_per_env", 0)
            else:
                # Check nested keys (e.g., extra.semantic_features)
                for comp_key in obs_info["components"]:
                    if comp_key.endswith(f".{feature_name}") or comp_key == f"extra.{feature_name}":
                        found = True
                        actual_count = obs_info["components"][comp_key].get("features_per_env", 0)
                        break

            if not found:
                missing_features.append(feature_name)
            elif "features_per_env" in obs_info["components"].get(feature_name, {}) or any(
                comp_key.endswith(f".{feature_name}") for comp_key in obs_info["components"]
            ):
                if actual_count != expected_count:
                    if verbose:
                        print(f"WARNING: {feature_name} has {actual_count} features, expected {expected_count}")

        if missing_features:
            if verbose:
                print(f"MISSING FEATURES: {missing_features}")
            obs_info["missing_features"] = missing_features
        elif verbose:
            print("✓ All expected features are present!")

        return obs_info

    def _get_expected_features(self):
        """Get expected feature counts based on obs_mode."""
        features = {
            "is_grasped": 1,
            "tcp_pose": 7,
            "goal_pos": 3,
            "obj_pose": 7,
            "tcp_to_obj_pos": 3,
            "obj_to_goal_pos": 3,
            "box_x_axis_in_world": 3,
            "gripper_x_axis": 3,
            "angle_between_axes": 1,
            "is_obj_stable": 1,
        }

        # Add bimanual-specific features if in bimanual mode
        if self.bimanual:
            features.update({
                "left_tcp_pose": 7,
                "right_tcp_pose": 7,
                "left_tcp_to_obj_pos": 3,
                "right_tcp_to_obj_pos": 3,
            })

        return features

    def print_observation_details(self, obs=None):
        """Print detailed information about the current observation.

        Args:
            obs (dict, optional): Observation dictionary. If None, will get current observation.
        """
        if obs is None:
            action = self.action_space.sample()
            obs, reward, terminated, truncated, info = self.step(action)

        print("\n=== Observation Details ===")
        print("Observation Mode: state")
        print(f"Number of environments: {len(self.scene.sub_scenes)}")

        # Check if obs is a dictionary
        if not isinstance(obs, dict):
            print("Observation type: Tensor (flattened)")
            print(f"Observation shape: {obs.shape}")
            print(f"Total features: {obs.shape[1] if obs.dim() > 1 else obs.shape[0]}")
            print("Features are embedded in flattened tensor")
            return

        # Handle dictionary observations
        print("Observation type: Dictionary")
        print(f"Top-level keys: {list(obs.keys())}")

        # Print details for each top-level key
        for key, value in obs.items():
            print(f"\n{key}:")
            if isinstance(value, dict):
                print(f"  Type: Dictionary with {len(value)} keys")
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, torch.Tensor):
                        print(f"    {sub_key}: {sub_value.shape} ({sub_value.dtype})")
                    else:
                        print(f"    {sub_key}: {type(sub_value).__name__}")
            elif isinstance(value, torch.Tensor):
                print(f"  Shape: {value.shape}")
                print(f"  Dtype: {value.dtype}")
                print(f"  Device: {value.device}")
                print(f"  Range: [{value.min().item():.4f}, {value.max().item():.4f}]")

                # Print first few values for small tensors
                if value.numel() <= 20:
                    print(f"  Values: {value.flatten()}")
                else:
                    print(f"  First 5 values: {value.flatten()[:5]}")
            else:
                print(f"  Value: {value}")

    def _setup_env_creation_lighting(self):
        """Setup lighting at environment creation with randomized directions per environment."""
        # Use a properly seeded random number generator for lighting randomization
        # This ensures different lighting for each environment creation
        import time

        rng = np.random.RandomState(int(time.time() * 1000000) % (2**32))

        # Generate random ambient light intensity
        ambient_intensity = rng.uniform(
            self.ambient_light_ranges["min_intensity"][0],
            self.ambient_light_ranges["max_intensity"][1],
        )

        ambient_color = [ambient_intensity, ambient_intensity, ambient_intensity]

        # Set ambient light
        self.scene.set_ambient_light(ambient_color)

        # Generate random directional light parameters (fixed per environment creation)
        directional_intensity = rng.uniform(
            self.directional_light_ranges["intensity"][0],
            self.directional_light_ranges["intensity"][1],
        )

        directional_color = [
            directional_intensity,
            directional_intensity,
            directional_intensity,
        ]

        # Generate random direction (fixed per environment creation)
        direction = [
            rng.uniform(
                self.directional_light_ranges["direction_x"][0],
                self.directional_light_ranges["direction_x"][1],
            ),
            rng.uniform(
                self.directional_light_ranges["direction_y"][0],
                self.directional_light_ranges["direction_y"][1],
            ),
            rng.uniform(
                self.directional_light_ranges["direction_z"][0],
                self.directional_light_ranges["direction_z"][1],
            ),
        ]

        # Store the direction for this environment instance
        self._env_lighting_directions = direction

        # Add single directional light with enhanced shadow settings and store reference
        self._directional_light = self.scene.add_directional_light(
            direction=direction,
            color=directional_color,
            shadow=True,  # Always enable shadows as requested
            shadow_scale=10,  # Increased shadow coverage for stronger shadows
            shadow_map_size=2048,  # High resolution shadows for crisp edges
        )

        if self.verbose:
            print(f"    ✓ Directional light created and stored: {self._directional_light is not None}")

        if self.verbose:
            print("    Environment creation lighting setup (enhanced shadows):")
            print(f"      Ambient light: {ambient_color}")
            print(f"      Directional light: {directional_color}, direction: {direction}")
            print("      Shadows: Always enabled with enhanced settings")
            print("      Shadow quality: 4096x4096 map, scale=15 (stronger shadows)")
            print("      Lighting direction: Fixed per environment creation")


@register_env("PickBoxBimanual-v1", max_episode_steps=100)
class PickBoxBimanualEnv(PickBoxEnv):
    """Bimanual pick-and-place task using b5box and basket assets with static left arm."""

    def __init__(self, *args, **kwargs):
        # Force bimanual mode
        kwargs["bimanual"] = True
        super().__init__(*args, **kwargs)
