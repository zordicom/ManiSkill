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

import math
from typing import Any, Dict, Union

import numpy as np
import sapien
import torch
from sapien import physx

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.robots import A1Galaxea, Panda, PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.tasks.tabletop.bimanual_pick_place_cfgs import (
    BIMANUAL_PICK_PLACE_CONFIG,
    ROBOT_CONFIGS,
)
from mani_skill.envs.tasks.tabletop.pick_box_cfgs import PICK_BOX_CONFIGS
from mani_skill.envs.tasks.tabletop.pick_cube_cfgs import PICK_CUBE_CONFIGS
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.building import actors
from mani_skill.utils.geometry.rotation_conversions import quaternion_apply
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

__all__ = ["PickBoxEnv"]


@register_env("PickBox-v1", max_episode_steps=50)
class PickBoxEnv(BaseEnv):
    """Single-arm pick-and-place task using b5box and basket assets."""

    SUPPORTED_ROBOTS = [
        "panda",
        "panda_wristcam",
        "a1_galaxea",
    ]

    agent: Union[Panda, PandaWristCam, A1Galaxea]

    # Task constants (match BimanualPickPlace)
    # Scale factor applied to the STL/OBJ meshes for the b5box asset. The
    # original half-size was 5 cm; after an 80 % scale-down the new half-size
    # becomes 4 cm.
    b5box_half_size = (
        0.012  # 4 cm half-extent after scaling (will be overridden by config)
    )
    goal_thresh = 0.015  # success threshold (metres) (will be overridden by config)

    # Base physical properties for randomization
    base_static_friction = 1.0
    base_dynamic_friction = 1.0
    base_density = 1000.0  # kg/m³
    base_linear_damping = 0.1  # Base linear damping coefficient

    def __init__(
        self, *args, robot_uids="a1_galaxea", obs_mode="state", verbose=False, **kwargs
    ):
        self.verbose = verbose
        # Resolve to a canonical UID string
        self.robot_uids = robot_uids
        self.robot_type = robot_uids  # alias for config lookup

        # Use robot-specific configs like PickCube does
        if robot_uids in PICK_BOX_CONFIGS:
            cfg = PICK_BOX_CONFIGS[robot_uids]
            self.goal_thresh = cfg["goal_thresh"]
            self.b5box_half_size = cfg["cube_half_size"]
            self.cube_half_size = cfg["cube_half_size"]
        else:
            # Fallback to default values
            cfg = PICK_BOX_CONFIGS["panda"]
            self.goal_thresh = cfg["goal_thresh"]
            self.b5box_half_size = cfg["cube_half_size"]
            self.cube_half_size = cfg["cube_half_size"]

        # Initialize box_half_sizes early to avoid observation space generation errors
        # This will be updated later in _load_b5box based on the actual box configuration
        self.box_half_sizes = [
            self.cube_half_size,
            self.cube_half_size,
            self.cube_half_size,
        ]

        # Assets directory (same as bimanual task)
        self.asset_root = PACKAGE_ASSET_DIR / "tasks/a1_pick_place"

        if self.verbose:
            print("PickBox Environment Initialization:")
            print(f"  Robot: {robot_uids}")
            print(f"  Observation mode: {obs_mode}")
            print(f"  Cube half-size: {self.cube_half_size}")
            print(f"  Box half-size: {self.b5box_half_size}")
            print(f"  Goal threshold: {self.goal_thresh}")

        super().__init__(*args, robot_uids=robot_uids, obs_mode=obs_mode, **kwargs)

        if self.verbose:
            print("  ✓ PickBox Environment initialization complete!")
            # Show observation space info
            expected_features = self._get_expected_features()
            total_features = sum(expected_features.values())
            print(f"  Expected observation features: {total_features} total")
            for name, count in expected_features.items():
                print(f"    {name}: {count} features")

    # ---------------------------------------------------------------------
    # Camera configuration (static top & human render) --------------------
    # ---------------------------------------------------------------------

    @property
    def _default_sensor_configs(self):
        cam_cfg = BIMANUAL_PICK_PLACE_CONFIG["cameras"]["static_top"]
        return [
            # RGB camera (existing)
            CameraConfig(
                "base_camera",
                cam_cfg["pose"].sp,
                cam_cfg["width"],
                cam_cfg["height"],
                cam_cfg["fov"],
                0.01,
                100,
            ),
            # For depth, we would need to enable depth in the obs_mode
            # The depth is handled by the shader system, not camera config
        ]

    @property
    def _default_human_render_camera_configs(self):
        cam_cfg = BIMANUAL_PICK_PLACE_CONFIG["cameras"]["human_render"]
        return CameraConfig(
            "render_camera",
            cam_cfg["pose"].sp,
            cam_cfg["width"],
            cam_cfg["height"],
            cam_cfg["fov"],
            0.01,
            100,
        )

    # ------------------------------------------------------------------
    # Scene & robot loading --------------------------------------------
    # ------------------------------------------------------------------

    def _load_agent(self, options: dict):
        # Use bimanual config positioning for all robots
        right_pose = ROBOT_CONFIGS[self.robot_type]["right_arm"]["pose"].sp
        super()._load_agent(options, right_pose)

    def _load_scene(self, options: dict):
        # Table, b5box, basket, and goal site (for visualisation) ----------
        self.table = self._create_table()
        self.b5box = self._load_b5box()
        self.basket = self._load_basket()
        self.goal_site = self._create_goal_site()

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

        builder.add_box_collision(
            half_size=cfg["size"],
            material=table_fric_mat,
        )
        builder.add_box_visual(half_size=cfg["size"])
        builder.set_initial_pose(cfg["pose"].sp)
        return builder.build_static(name="table")

    def _load_b5box(self):
        """Create a red box with x-axis 3x longer than other dimensions.

        For simplicity we keep the original variable name (b5box) so that the
        rest of the task logic remains unchanged.
        """
        # Option 1: Uniform cube (previous implementation)
        # cube = actors.build_cube(
        #     self.scene,
        #     half_size=self.b5box_half_size,
        #     color=[1, 0, 0, 1],
        #     name="cube",
        #     initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        # )

        # Update box_half_sizes for uniform cube
        # self.box_half_sizes = [
        #     self.b5box_half_size,
        #     self.b5box_half_size,
        #     self.b5box_half_size,
        # ]

        # Option 2: Non-uniform box (longer in x-axis)
        self.box_half_sizes = [
            3 * self.b5box_half_size,  # x-axis: 3x longer
            self.b5box_half_size,  # y-axis: same as original
            self.b5box_half_size,  # z-axis: same as original
        ]  # [width, depth, height] half-sizes - longer in x

        # Create box with base physical material (friction will be randomized per episode)
        # Apply some friction randomization at the scene level (affects all environments)
        scene_friction_multiplier = 0.5 + np.random.random()  # [0.5, 1.5]
        randomized_static_friction = (
            self.base_static_friction * scene_friction_multiplier
        )
        randomized_dynamic_friction = (
            self.base_dynamic_friction * scene_friction_multiplier
        )

        self.box_material = physx.PhysxMaterial(
            static_friction=randomized_static_friction,
            dynamic_friction=randomized_dynamic_friction,
            restitution=0.1,
        )

        if self.verbose:
            print(
                f"    Scene-level friction multiplier: {scene_friction_multiplier:.3f}"
            )
            print(f"    Scene static friction: {randomized_static_friction:.3f}")
            print(f"    Scene dynamic friction: {randomized_dynamic_friction:.3f}")

        # Apply mass/density randomization at creation time (±50% variation)
        mass_multiplier = 0.5 + np.random.random()  # [0.5, 1.5]
        randomized_density = self.base_density * mass_multiplier

        builder = self.scene.create_actor_builder()
        builder.add_box_collision(
            half_size=self.box_half_sizes,
            material=self.box_material,
            density=randomized_density,
        )
        builder.add_box_visual(
            half_size=self.box_half_sizes,
            material=[1, 0, 0, 1],  # Red color
        )
        builder.set_initial_pose(sapien.Pose(p=[0, 0, self.box_half_sizes[2]]))
        cube = builder.build(name="cube")

        # Apply damping randomization after creation but before GPU initialization
        damping_multiplier = 0.5 + np.random.random()  # [0.5, 1.5]
        randomized_damping = self.base_linear_damping * damping_multiplier
        cube.linear_damping = randomized_damping

        if self.verbose:
            print(f"    Creation-time mass multiplier: {mass_multiplier:.3f}")
            print(f"    Creation-time density: {randomized_density:.1f} kg/m³")
            print(f"    Creation-time damping multiplier: {damping_multiplier:.3f}")
            print(f"    Creation-time damping: {randomized_damping:.4f}")

        return cube

    def _load_basket(self):
        builder = self.scene.create_actor_builder()
        # Bottom
        builder.add_box_collision(
            pose=sapien.Pose(p=[0, 0, 0]),
            half_size=[0.12, 0.09, 0.004],
        )
        # Walls (south/north/east/west)
        builder.add_box_collision(
            pose=sapien.Pose(p=[-0.13, 0, 0.06]), half_size=[0.005, 0.1, 0.05]
        )
        builder.add_box_collision(
            pose=sapien.Pose(p=[0.13, 0, 0.06]), half_size=[0.005, 0.1, 0.05]
        )
        builder.add_box_collision(
            pose=sapien.Pose(p=[0.0, 0.1, 0.06]), half_size=[0.12, 0.005, 0.05]
        )
        builder.add_box_collision(
            pose=sapien.Pose(p=[0.0, -0.1, 0.06]), half_size=[0.12, 0.005, 0.05]
        )
        visual_file = str(self.asset_root / "basket/basket.obj")
        builder.add_visual_from_file(filename=visual_file, scale=[1.0, 1.0, 1.0])
        builder.set_initial_pose(sapien.Pose(p=[-0.2, 0, 0.01]))
        return builder.build_kinematic(name="basket")

    def _create_goal_site(self):
        goal_site = actors.build_sphere(
            self.scene,
            radius=0.02,
            color=[0, 1, 0, 0.5],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(goal_site)
        return goal_site

    # ------------------------------------------------------------------
    # Episode initialisation -------------------------------------------
    # ------------------------------------------------------------------

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)

            if self.verbose and b > 0:
                print(f"\nInitializing episode for {b} environments...")
                print(
                    f"  Environment indices: {env_idx[: min(5, b)].tolist()}"
                )  # Show first 5

            # Home pose ---------------------------------------------------
            home_qpos = ROBOT_CONFIGS[self.robot_type]["right_arm"]["home_qpos"]
            self.agent.robot.set_qpos(
                torch.tensor(home_qpos, device=self.device).repeat(b, 1)
            )

            # Scene parameters ------------------------------------------
            scene_cfg = BIMANUAL_PICK_PLACE_CONFIG["scene"]
            # Pose.p is of shape (num_envs, 3); take the z-component robustly
            table_surface_z = (
                scene_cfg["table"]["pose"].p[..., 2] + scene_cfg["table"]["size"][2]
            )

            # Randomise b5box position ----------------------------------
            box_spawn = scene_cfg["b5box"]["spawn_region"]
            box_xyz = torch.zeros((b, 3))
            box_xyz[:, 0] = (
                box_spawn["center"][0]
                + (torch.rand((b,)) * 2 - 1) * box_spawn["half_size"][0]
            )
            box_xyz[:, 1] = (
                box_spawn["center"][1]
                + (torch.rand((b,)) * 2 - 1) * box_spawn["half_size"][1]
            )
            box_xyz[:, 2] = table_surface_z + self.b5box_half_size
            box_qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.b5box.set_pose(Pose.create_from_pq(box_xyz, box_qs))

            # Randomise basket position ----------------------------------
            basket_spawn = scene_cfg["basket"]["spawn_region"]
            basket_xyz = torch.zeros((b, 3))
            basket_xyz[:, 0] = (
                basket_spawn["center"][0]
                + (torch.rand((b,)) * 2 - 1) * basket_spawn["half_size"][0]
            )
            basket_xyz[:, 1] = (
                basket_spawn["center"][1]
                + (torch.rand((b,)) * 2 - 1) * basket_spawn["half_size"][1]
            )
            basket_xyz[:, 2] = table_surface_z + 0.054  # basket bottom height
            # 90° rotation around Z (match original)
            z_rot = torch.tensor(np.pi / 2, device=self.device)
            basket_qs = torch.zeros((b, 4), device=self.device)
            basket_qs[:, 0] = torch.cos(z_rot / 2)
            basket_qs[:, 3] = torch.sin(z_rot / 2)
            self.basket.set_pose(Pose.create_from_pq(basket_xyz, basket_qs))

            # Goal site for visualisation ------------------------------
            goal_xyz = basket_xyz.clone()
            goal_xyz[:, 2] = table_surface_z + 0.15
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

            # Physical properties (mass, damping, friction) are randomized at creation time
            # since they cannot be modified after GPU simulation initialization

            if self.verbose and b > 0:
                print(f"  ✓ Episode initialized for {b} environments")
                print(f"    Box position (first env): {box_xyz[0]}")
                print(f"    Basket position (first env): {basket_xyz[0]}")
                print(f"    Goal position (first env): {goal_xyz[0]}")

    # ------------------------------------------------------------------
    # Observation extras -----------------------------------------------
    # ------------------------------------------------------------------

    def _get_obs_extra(self, info: Dict):
        b = len(self.scene.sub_scenes)

        # Calculate gripper x-axis in world frame
        local_x_axis = torch.tensor([1, 0, 0], device=self.device).expand(b, 3)
        gripper_x_axis = quaternion_apply(self.agent.tcp_pose.q, local_x_axis)

        # Calculate box x-axis in world frame
        box_x_axis = quaternion_apply(self.b5box.pose.q, local_x_axis)

        # Calculate angle between box and gripper x-axes
        box_x_axis_norm = box_x_axis / torch.linalg.norm(
            box_x_axis, dim=-1, keepdim=True
        )
        gripper_x_axis_norm = gripper_x_axis / torch.linalg.norm(
            gripper_x_axis, dim=-1, keepdim=True
        )
        dot_product = torch.sum(
            box_x_axis_norm * gripper_x_axis_norm, dim=-1, keepdim=True
        )
        dot_product = torch.clamp(dot_product, -1.0, 1.0)
        angle_between_axes = torch.acos(dot_product)

        if self.verbose and hasattr(self, "_verbose_obs_count"):
            self._verbose_obs_count += 1
            if self._verbose_obs_count % 100 == 1:  # Print every 100 observations
                print(f"\nObservation {self._verbose_obs_count} (first env):")
                print(f"  Box x-axis: {box_x_axis[0]}")
                print(f"  Gripper x-axis: {gripper_x_axis[0]}")
                print(
                    f"  Angle between axes: {angle_between_axes[0].item():.3f} rad ({angle_between_axes[0].item() * 180 / math.pi:.1f}°)"
                )
        elif self.verbose:
            self._verbose_obs_count = 1

        obs = dict(
            is_grasped=info["is_grasped"],
            tcp_pose=self.agent.tcp_pose.raw_pose,
            goal_pos=self.goal_site.pose.p,
            obj_pose=self.b5box.pose.raw_pose,
            tcp_to_obj_pos=self.b5box.pose.p - self.agent.tcp_pose.p,
            obj_to_goal_pos=self.goal_site.pose.p - self.b5box.pose.p,
            box_x_axis_in_world=self._get_box_axes_in_world(),
            gripper_x_axis=gripper_x_axis,
            angle_between_axes=angle_between_axes,
        )
        return obs

    def _get_box_axes_in_world(self):
        """Get the box's local axes (X, Y, Z) expressed in world coordinates.

        This helps the agent understand the box's orientation for better grasping.
        Returns: tensor of shape (num_envs, 9) representing [x_axis, y_axis, z_axis]
        """
        box_pose = self.b5box.pose

        # Define local axes
        x_axis = torch.tensor([1, 0, 0], device=self.device).expand(
            len(self.scene.sub_scenes), 3
        )

        # Transform to world frame using quaternion
        world_x_axis = quaternion_apply(box_pose.q, x_axis)  # Longest dimension

        return world_x_axis

    # ------------------------------------------------------------------
    # Task evaluation & reward -----------------------------------------
    # ------------------------------------------------------------------

    def evaluate(self):
        # Proximity check
        is_obj_placed = (
            torch.linalg.norm(self.goal_site.pose.p - self.b5box.pose.p, axis=1)
            <= self.goal_thresh
        )
        is_grasped = self.agent.is_grasping(self.b5box)
        is_robot_static = self.agent.is_static(0.2)
        return {
            "success": is_obj_placed & is_robot_static,
            "is_obj_placed": is_obj_placed,
            "is_robot_static": is_robot_static,
            "is_grasped": is_grasped,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_to_obj_dist = torch.linalg.norm(
            self.b5box.pose.p - self.agent.tcp_pose.p, axis=1
        )

        # TCP-to-object proximity reward throughout task, except after releasing
        # Stop rewarding TCP proximity when object is placed AND not grasped (i.e., after release)
        is_grasped = info["is_grasped"]
        is_obj_placed = info["is_obj_placed"]
        should_reward_tcp_proximity = ~(
            is_obj_placed & ~is_grasped
        )  # NOT (placed AND not grasped)

        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward * should_reward_tcp_proximity.float()

        # Binary grasp reward
        reward += is_grasped.float()

        # Lifting reward: high reward for lifting the box above the table
        # Get table surface height
        scene_cfg = BIMANUAL_PICK_PLACE_CONFIG["scene"]
        table_pose_z = torch.tensor(
            scene_cfg["table"]["pose"].p[..., 2], device=self.device
        )
        table_size_z = torch.tensor(scene_cfg["table"]["size"][2], device=self.device)
        table_surface_z = table_pose_z + table_size_z

        # Calculate how high the box is above the table surface
        box_height_above_table = self.b5box.pose.p[:, 2] - table_surface_z
        # Give lifting reward when grasped and lifted (scale by 0.05m = 5cm max lift)
        lifting_reward = torch.clamp(box_height_above_table / 0.05, 0.0, 1.0)
        reward += lifting_reward * is_grasped.float() * 2.0  # 2x weight for lifting

        # Check if box is lifted above 3cm threshold
        is_lifted_above_threshold = box_height_above_table >= 0.03  # 3cm threshold

        # Transport / place reward ONLY when grasped AND lifted above 3cm
        obj_to_goal_dist = torch.linalg.norm(
            self.goal_site.pose.p - self.b5box.pose.p, axis=1
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        # Goal-related reward only kicks in after lifting above 3cm
        reward += place_reward * is_grasped.float() * is_lifted_above_threshold.float()

        # X-axis alignment reward: reward alignment of box and gripper x-axes
        # Calculate gripper x-axis in world frame
        local_x_axis = torch.tensor([1, 0, 0], device=self.device).expand(
            len(self.scene.sub_scenes), 3
        )
        gripper_x_axis = quaternion_apply(self.agent.tcp_pose.q, local_x_axis)
        # Calculate box x-axis in world frame
        box_x_axis = quaternion_apply(self.b5box.pose.q, local_x_axis)
        # Calculate angle between axes
        box_x_axis_norm = box_x_axis / torch.linalg.norm(
            box_x_axis, dim=-1, keepdim=True
        )
        gripper_x_axis_norm = gripper_x_axis / torch.linalg.norm(
            gripper_x_axis, dim=-1, keepdim=True
        )
        dot_product = torch.sum(box_x_axis_norm * gripper_x_axis_norm, dim=-1)
        dot_product = torch.clamp(dot_product, -1.0, 1.0)
        angle_between_axes = torch.acos(dot_product)
        # Handle flipped axes by taking min(angle, π - angle) for alignment reward
        alignment_angle = torch.min(angle_between_axes, torch.pi - angle_between_axes)
        alignment_reward = 1 - torch.tanh(5 * alignment_angle)
        reward += 0.1 * alignment_reward * should_reward_tcp_proximity.float()

        # Gripper openness reward: encourage keeping gripper open when far from object
        # Get gripper joint positions (assuming they're at the end of qpos)
        robot_qpos = self.agent.robot.get_qpos()
        if self.robot_uids in ["panda", "panda_wristcam"]:
            # For Panda robots, gripper joints are the last 2 joints
            gripper_qpos = robot_qpos[..., -2:]  # [left_finger, right_finger]
            # Gripper openness is the sum of finger positions (more open = higher values)
            gripper_openness = torch.sum(gripper_qpos, dim=-1)
            max_gripper_openness = 0.08  # Panda gripper max opening (~4cm per finger)
        elif self.robot_uids == "a1_galaxea":
            # For A1 Galaxea, gripper joints are the last 2 joints
            gripper_qpos = robot_qpos[..., -2:]  # [left_finger, right_finger]
            # Gripper openness is the sum of finger positions (more open = higher values)
            gripper_openness = torch.sum(gripper_qpos, dim=-1)
            max_gripper_openness = -0.03  # A1 gripper max opening
        else:
            # Fallback for other robots
            gripper_qpos = robot_qpos[..., -2:]
            gripper_openness = torch.sum(gripper_qpos, dim=-1)
            max_gripper_openness = 0.08

        # Normalize gripper openness to [0, 1] range
        normalized_gripper_openness = torch.clamp(
            gripper_openness / max_gripper_openness, 0.0, 1.0
        )

        # Only reward open gripper when NOT grasping and when far from object
        should_reward_open_gripper = ~is_grasped  # NOT grasping

        # Reward for keeping gripper open when appropriate
        open_gripper_reward = normalized_gripper_openness * 0.01  # Weight of 0.5
        reward += open_gripper_reward * should_reward_open_gripper.float()

        # Small penalty for closing gripper when far from object and not grasping
        premature_close_penalty = (
            1.0 - normalized_gripper_openness
        ) * 0.3  # Weight of 0.3
        reward -= premature_close_penalty * should_reward_open_gripper.float()

        # Static bonus when placed
        qvel = self.agent.robot.get_qvel()
        if self.robot_uids in ["panda", "panda_wristcam", "a1_galaxea"]:
            qvel = qvel[..., :-2]  # strip gripper joints if present
        static_reward = 1 - torch.tanh(5 * torch.linalg.norm(qvel, axis=1))
        reward += static_reward * info["is_obj_placed"].float()

        # Success bonus
        reward[info["success"]] = 5.0
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs, action, info) / 8.8

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
        if isinstance(obs, dict):  # noqa: PLR1702
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
                        print(
                            f"  {key:20s}: {shape} ({dtype}) - {feature_count} features/env"
                        )
                elif isinstance(value, dict):
                    # Handle nested dictionaries (like 'extra' or 'agent')
                    if verbose:
                        print(
                            f"  {key:20s}: Dictionary with {len(value)} sub-components"
                        )
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, torch.Tensor):
                            shape = tuple(sub_value.shape)
                            dtype = sub_value.dtype
                            feature_count = sub_value.numel() // len(
                                self.scene.sub_scenes
                            )

                            full_key = f"{key}.{sub_key}"
                            obs_info["components"][full_key] = {
                                "shape": shape,
                                "dtype": str(dtype),
                                "features_per_env": feature_count,
                                "total_features": sub_value.numel(),
                            }
                            obs_info["total_features"] += feature_count

                            if verbose:
                                print(
                                    f"    {sub_key:18s}: {shape} ({dtype}) - {feature_count} features/env"
                                )
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
                print(
                    f"  flattened_state    : {shape} ({dtype}) - {feature_count} features/env"
                )

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
                actual_count = obs_info["components"][feature_name].get(
                    "features_per_env", 0
                )
            else:
                # Check nested keys (e.g., extra.semantic_features)
                for comp_key in obs_info["components"]:
                    if (
                        comp_key.endswith(f".{feature_name}")
                        or comp_key == f"extra.{feature_name}"
                    ):
                        found = True
                        actual_count = obs_info["components"][comp_key].get(
                            "features_per_env", 0
                        )
                        break

            if not found:
                missing_features.append(feature_name)
            elif "features_per_env" in obs_info["components"].get(
                feature_name, {}
            ) or any(
                comp_key.endswith(f".{feature_name}")
                for comp_key in obs_info["components"]
            ):
                if actual_count != expected_count:
                    if verbose:
                        print(
                            f"WARNING: {feature_name} has {actual_count} features, expected {expected_count}"
                        )

        if missing_features:
            if verbose:
                print(f"MISSING FEATURES: {missing_features}")
            obs_info["missing_features"] = missing_features
        elif verbose:
            print("✓ All expected features are present!")

        return obs_info

    def _get_expected_features(self):
        """Get expected feature counts based on obs_mode."""
        return {
            "is_grasped": 1,
            "tcp_pose": 7,
            "goal_pos": 3,
            "obj_pose": 7,
            "tcp_to_obj_pos": 3,
            "obj_to_goal_pos": 3,
            "box_x_axis_in_world": 3,
            "gripper_x_axis": 3,
            "angle_between_axes": 1,
        }

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
