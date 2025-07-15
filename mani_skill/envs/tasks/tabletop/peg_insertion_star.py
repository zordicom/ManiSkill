"""
Copyright 2025 Zordi, Inc. All rights reserved.
"""

from typing import Any, Dict, Union

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.agents.robots.panda import PandaWristCam
from mani_skill.agents.robots.xarm6 import XArm6Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Actor, Pose
from mani_skill.utils.structs.types import SimConfig


def _build_star_shaped_peg(
    scene: ManiSkillScene, outer_radius: float, inner_radius: float, height: float
):
    """Build a star-shaped peg using multiple cylinders."""
    builder = scene.create_actor_builder()

    # Create star shape using 5 cylinders arranged in a star pattern
    num_points = 5
    angle_step = 2 * np.pi / num_points

    # Central core
    builder.add_cylinder_collision(radius=inner_radius, half_length=height / 2)

    # Star points
    for i in range(num_points):
        angle = i * angle_step
        # Position each point cylinder
        x = (outer_radius + inner_radius) / 2 * np.cos(angle)
        y = (outer_radius + inner_radius) / 2 * np.sin(angle)

        point_pose = sapien.Pose(p=[x, y, 0], q=euler2quat(0, 0, angle))

        # Add collision for each point
        builder.add_cylinder_collision(
            pose=point_pose, radius=inner_radius * 0.6, half_length=height / 2
        )

    # Visual materials
    core_mat = sapien.render.RenderMaterial(
        base_color=sapien_utils.hex2rgba("#F39C12"), roughness=0.4, specular=0.6
    )
    point_mat = sapien.render.RenderMaterial(
        base_color=sapien_utils.hex2rgba("#E74C3C"), roughness=0.4, specular=0.6
    )

    # Add visual components
    # Central core
    builder.add_cylinder_visual(
        radius=inner_radius, half_length=height / 2, material=core_mat
    )

    # Star points
    for i in range(num_points):
        angle = i * angle_step
        x = (outer_radius + inner_radius) / 2 * np.cos(angle)
        y = (outer_radius + inner_radius) / 2 * np.sin(angle)

        point_pose = sapien.Pose(p=[x, y, 0], q=euler2quat(0, 0, angle))

        builder.add_cylinder_visual(
            pose=point_pose,
            radius=inner_radius * 0.6,
            half_length=height / 2,
            material=point_mat,
        )

    return builder


def _build_star_block_with_hole(
    scene: ManiSkillScene, outer_radius: float, inner_radius: float, block_height: float
):
    """Build a block with a star-shaped hole."""
    builder = scene.create_actor_builder()

    # Block dimensions
    block_size = outer_radius * 2.5
    wall_thickness = 0.02

    # Create the main block material
    main_mat = sapien.render.RenderMaterial(
        base_color=sapien_utils.hex2rgba("#8B4513"), roughness=0.6, specular=0.3
    )

    # Create the block structure around the star hole
    # We'll build it as a series of segments that form the negative space of the star

    # Bottom part of the block
    bottom_pose = sapien.Pose(p=[0, 0, -block_height / 2 + wall_thickness / 2])
    builder.add_box_collision(
        pose=bottom_pose, half_size=[block_size / 2, block_size / 2, wall_thickness / 2]
    )
    builder.add_box_visual(
        pose=bottom_pose,
        half_size=[block_size / 2, block_size / 2, wall_thickness / 2],
        material=main_mat,
    )

    # Create walls around the star hole using multiple segments
    num_segments = 20  # Higher resolution for better approximation
    segment_thickness = 0.015

    for i in range(num_segments):
        angle = i * 2 * np.pi / num_segments

        # Calculate distance from center for this angle
        # Simple approximation of star shape
        star_angle = (angle * 5) % (2 * np.pi)  # 5-pointed star
        star_radius = inner_radius + (outer_radius - inner_radius) * (
            0.5 + 0.5 * np.cos(star_angle)
        )

        # Position segment at star boundary
        segment_radius = star_radius + segment_thickness
        x = segment_radius * np.cos(angle)
        y = segment_radius * np.sin(angle)

        segment_pose = sapien.Pose(p=[x, y, 0])
        builder.add_box_collision(
            pose=segment_pose,
            half_size=[segment_thickness / 2, segment_thickness / 2, block_height / 2],
        )
        builder.add_box_visual(
            pose=segment_pose,
            half_size=[segment_thickness / 2, segment_thickness / 2, block_height / 2],
            material=main_mat,
        )

    # Outer walls
    outer_wall_radius = block_size / 2 - wall_thickness / 2
    for i in range(num_segments):
        angle = i * 2 * np.pi / num_segments
        x = outer_wall_radius * np.cos(angle)
        y = outer_wall_radius * np.sin(angle)

        wall_pose = sapien.Pose(p=[x, y, 0])
        builder.add_box_collision(
            pose=wall_pose,
            half_size=[wall_thickness / 2, wall_thickness / 2, block_height / 2],
        )
        builder.add_box_visual(
            pose=wall_pose,
            half_size=[wall_thickness / 2, wall_thickness / 2, block_height / 2],
            material=main_mat,
        )

    return builder


@register_env("PegInsertionStar-v1", max_episode_steps=200)
class PegInsertionStarEnv(BaseEnv):
    """
    **Task Description:**
    Pick up a star-shaped peg and insert it into a matching star-shaped hole in a wooden block.
    The peg has a 5-pointed star cross-section requiring very precise alignment.

    **Randomizations:**
    - Peg height is randomized between 0.06 and 0.10 meters
    - Outer radius is randomized between 0.025 and 0.035 meters
    - Inner radius is randomized between 0.015 and 0.020 meters
    - Peg and block positions are randomized on the table
    - Block orientation is randomized around z-axis

    **Success Conditions:**
    - The peg tip is inserted at least 0.05m into the hole
    - The peg is properly aligned with the star hole orientation (within 5 degrees)
    - All star points are within the hole boundaries
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PegInsertionStar-v1_rt.mp4"
    SUPPORTED_ROBOTS = ["panda_wristcam", "xarm6_robotiq"]
    agent: Union[PandaWristCam, XArm6Robotiq]
    _clearance = 0.002

    def __init__(
        self,
        *args,
        robot_uids="panda_wristcam",
        num_envs=1,
        reconfiguration_freq=None,
        **kwargs,
    ):
        if reconfiguration_freq is None:
            if num_envs == 1:
                reconfiguration_freq = 1
            else:
                reconfiguration_freq = 0
        super().__init__(
            *args,
            robot_uids=robot_uids,
            num_envs=num_envs,
            reconfiguration_freq=reconfiguration_freq,
            **kwargs,
        )

    @property
    def _default_sim_config(self):
        return SimConfig()

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at([0, -0.5, 0.4], [0, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.7, -0.7, 1.1], [0.05, -0.1, 0.4])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        with torch.device(self.device):
            self.table_scene = TableSceneBuilder(self)
            self.table_scene.build()

            # Randomize star-shaped peg dimensions
            peg_heights = self._batched_episode_rng.uniform(0.06, 0.10)
            outer_radii = self._batched_episode_rng.uniform(0.025, 0.035)
            inner_radii = self._batched_episode_rng.uniform(0.015, 0.020)

            # Store dimensions for use in reward computation
            self.peg_heights = common.to_tensor(peg_heights)
            self.outer_radii = common.to_tensor(outer_radii)
            self.inner_radii = common.to_tensor(inner_radii)

            # Create peg tip offset (bottom of peg)
            peg_tip_offsets = torch.zeros((self.num_envs, 3))
            peg_tip_offsets[:, 2] = -self.peg_heights / 2
            self.peg_tip_offsets = Pose.create_from_pq(p=peg_tip_offsets)

            # Create hole center offset
            hole_center_offsets = torch.zeros((self.num_envs, 3))
            self.hole_center_offsets = Pose.create_from_pq(p=hole_center_offsets)

            # Create star point offsets for collision checking
            star_point_offsets = []
            for i in range(5):
                angle = i * 2 * np.pi / 5
                point_offsets = torch.zeros((self.num_envs, 3))
                point_offsets[:, 0] = self.outer_radii * np.cos(angle)
                point_offsets[:, 1] = self.outer_radii * np.sin(angle)
                point_offsets[:, 2] = -self.peg_heights / 2
                star_point_offsets.append(Pose.create_from_pq(p=point_offsets))
            self.star_point_offsets = star_point_offsets

            # Build pegs and blocks for each environment
            pegs = []
            blocks = []

            for i in range(self.num_envs):
                scene_idxs = [i]
                peg_height = peg_heights[i]
                outer_radius = outer_radii[i]
                inner_radius = inner_radii[i]

                # Create star-shaped peg
                peg_builder = _build_star_shaped_peg(
                    self.scene, outer_radius, inner_radius, peg_height
                )
                peg_builder.initial_pose = sapien.Pose(p=[0, 0, 0.15])
                peg_builder.set_scene_idxs(scene_idxs)
                peg = peg_builder.build(f"peg_{i}")
                self.remove_from_state_dict_registry(peg)

                # Create block with star-shaped hole
                block_height = 0.06
                block_builder = _build_star_block_with_hole(
                    self.scene,
                    outer_radius + self._clearance,
                    inner_radius + self._clearance,
                    block_height,
                )
                block_builder.initial_pose = sapien.Pose(p=[0, 0.35, block_height / 2])
                block_builder.set_scene_idxs(scene_idxs)
                block = block_builder.build_kinematic(f"block_{i}")
                self.remove_from_state_dict_registry(block)

                pegs.append(peg)
                blocks.append(block)

            self.peg = Actor.merge(pegs, "peg")
            self.block = Actor.merge(blocks, "block")

            # Register merged actors
            self.add_to_state_dict_registry(self.peg)
            self.add_to_state_dict_registry(self.block)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Initialize peg position
            peg_xy = randomization.uniform(
                low=torch.tensor([-0.2, -0.3]),
                high=torch.tensor([0.2, -0.1]),
                size=(b, 2),
            )
            peg_pos = torch.zeros((b, 3))
            peg_pos[:, :2] = peg_xy
            peg_pos[:, 2] = self.peg_heights[env_idx] / 2 + 0.01  # On table surface

            # Random orientation around z-axis
            peg_quat = randomization.random_quaternions(
                b, self.device, lock_x=True, lock_y=True, bounds=(0, 2 * np.pi)
            )
            self.peg.set_pose(Pose.create_from_pq(peg_pos, peg_quat))

            # Initialize block position
            block_xy = randomization.uniform(
                low=torch.tensor([-0.1, 0.25]),
                high=torch.tensor([0.1, 0.45]),
                size=(b, 2),
            )
            block_pos = torch.zeros((b, 3))
            block_pos[:, :2] = block_xy
            block_pos[:, 2] = 0.03  # Half height of block

            # Random orientation around z-axis
            block_quat = randomization.random_quaternions(
                b, self.device, lock_x=True, lock_y=True, bounds=(0, 2 * np.pi)
            )
            self.block.set_pose(Pose.create_from_pq(block_pos, block_quat))

            # Initialize robot
            if self.robot_uids == "xarm6_robotiq":
                qpos = np.array([
                    0.0,
                    0.22,
                    -1.23,
                    0.0,
                    1.01,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ])
            else:
                qpos = np.array([
                    0.0,
                    np.pi / 8,
                    0,
                    -np.pi * 5 / 8,
                    0,
                    np.pi * 3 / 4,
                    -np.pi / 4,
                    0.04,
                    0.04,
                ])
            qpos = self._episode_rng.normal(0, 0.02, (b, len(qpos))) + qpos
            if self.robot_uids == "xarm6_robotiq":
                qpos[:, -6:] = 0.0
            else:
                qpos[:, -2:] = 0.04
            self.agent.robot.set_qpos(qpos)
            self.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))

    @property
    def peg_tip_pos(self):
        return self.peg.pose.p + self.peg_tip_offsets.p

    @property
    def peg_tip_pose(self):
        return self.peg.pose * self.peg_tip_offsets

    @property
    def hole_center_pose(self):
        return self.block.pose * self.hole_center_offsets

    @property
    def star_point_poses(self):
        return [self.peg.pose * offset for offset in self.star_point_offsets]

    def has_peg_inserted(self):
        """Check if star-shaped peg is properly inserted into hole."""
        peg_tip_at_hole = (self.hole_center_pose.inv() * self.peg_tip_pose).p

        # Check if peg is inserted deep enough (z-axis is up)
        depth_flag = peg_tip_at_hole[:, 2] <= -0.05

        # Check rotational alignment (star needs precise orientation)
        # Get relative orientation between peg and hole
        relative_pose = self.hole_center_pose.inv() * self.peg.pose
        relative_quat = relative_pose.q

        # Extract z-axis rotation (yaw) from quaternion
        yaw = torch.atan2(
            2
            * (
                relative_quat[:, 3] * relative_quat[:, 2]
                + relative_quat[:, 0] * relative_quat[:, 1]
            ),
            1 - 2 * (relative_quat[:, 1] ** 2 + relative_quat[:, 2] ** 2),
        )

        # For 5-pointed star, alignment should be within 72 degrees (2*pi/5)
        star_angle = 2 * np.pi / 5
        yaw_aligned = torch.abs(yaw % star_angle) < (
            star_angle / 8
        )  # Within ~9 degrees

        # Check if all star points are within reasonable bounds
        star_points_valid = True
        for i, star_point_pose in enumerate(self.star_point_poses):
            point_at_hole = (self.hole_center_pose.inv() * star_point_pose).p
            point_dist = torch.linalg.norm(point_at_hole[:, :2], dim=1)
            point_valid = point_dist <= (self.outer_radii + self._clearance)
            star_points_valid = star_points_valid & point_valid

        return depth_flag & yaw_aligned & star_points_valid, peg_tip_at_hole

    def evaluate(self):
        success, peg_tip_at_hole = self.has_peg_inserted()
        return dict(success=success, peg_tip_at_hole=peg_tip_at_hole)

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self.obs_mode_struct.use_state:
            obs.update(
                peg_pose=self.peg.pose.raw_pose,
                peg_height=self.peg_heights,
                outer_radius=self.outer_radii,
                inner_radius=self.inner_radii,
                hole_center_pose=self.hole_center_pose.raw_pose,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # Stage 1: Reach and grasp the peg
        gripper_pos = self.agent.tcp.pose.p
        peg_grasp_pos = self.peg.pose.p + torch.tensor([0, 0, 0.02], device=self.device)

        reach_dist = torch.linalg.norm(gripper_pos - peg_grasp_pos, dim=1)
        reach_reward = 1 - torch.tanh(3.0 * reach_dist)

        is_grasped = self.agent.is_grasping(self.peg, max_angle=30)
        reward = reach_reward + is_grasped

        # Stage 2: Lift the peg
        peg_height = self.peg.pose.p[:, 2]
        lifted = peg_height > 0.12
        lift_reward = 2 * lifted.float()
        reward += lift_reward * is_grasped

        # Stage 3: Align peg with hole center
        peg_tip_at_hole = (self.hole_center_pose.inv() * self.peg_tip_pose).p
        xy_alignment = torch.linalg.norm(peg_tip_at_hole[:, :2], dim=1)

        position_alignment_reward = 4 * (1 - torch.tanh(12.0 * xy_alignment))
        reward += position_alignment_reward * is_grasped * lifted

        # Stage 4: Rotational alignment (critical for star shape)
        relative_pose = self.hole_center_pose.inv() * self.peg.pose
        relative_quat = relative_pose.q
        yaw = torch.atan2(
            2
            * (
                relative_quat[:, 3] * relative_quat[:, 2]
                + relative_quat[:, 0] * relative_quat[:, 1]
            ),
            1 - 2 * (relative_quat[:, 1] ** 2 + relative_quat[:, 2] ** 2),
        )

        # Calculate alignment error for 5-pointed star
        star_angle = 2 * np.pi / 5
        yaw_error = torch.abs(yaw % star_angle)
        yaw_error = torch.minimum(yaw_error, star_angle - yaw_error)

        rotation_alignment_reward = 5 * (1 - yaw_error / (star_angle / 2))
        well_positioned = xy_alignment < 0.01
        reward += rotation_alignment_reward * is_grasped * lifted * well_positioned

        # Stage 5: Insert peg into hole
        insertion_depth = torch.clamp(-peg_tip_at_hole[:, 2], 0, 0.05)
        insertion_reward = 8 * (insertion_depth / 0.05)

        well_aligned = (xy_alignment < 0.008) & (yaw_error < star_angle / 8)
        reward += insertion_reward * is_grasped * well_aligned

        # Success bonus
        reward[info["success"]] = 35

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs, action, info) / 35
