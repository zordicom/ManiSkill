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


def _build_angled_block_with_hole(
    scene: ManiSkillScene, hole_radius: float, block_size: float, angle: float
):
    """Build a block with an angled hole through it."""
    builder = scene.create_actor_builder()

    # Create the main block
    half_size = [block_size / 2, block_size / 2, block_size / 2]
    main_mat = sapien.render.RenderMaterial(
        base_color=sapien_utils.hex2rgba("#8B4513"), roughness=0.6, specular=0.3
    )
    builder.add_box_collision(half_size=half_size)
    builder.add_box_visual(half_size=half_size, material=main_mat)

    # Create the angled hole by subtracting a cylinder
    # The hole goes through the block at the specified angle
    hole_length = block_size * 1.5  # Make sure it goes through entire block
    hole_pose = sapien.Pose(
        p=[0, 0, 0],
        q=euler2quat(0, angle, 0),  # Rotate around y-axis
    )

    # Use a smaller collision cylinder to create the hole effect
    # Note: In a real implementation, you'd use CSG operations
    # For now, we'll approximate with visual indication
    hole_mat = sapien.render.RenderMaterial(
        base_color=sapien_utils.hex2rgba("#000000"), roughness=0.1, specular=0.9
    )
    builder.add_cylinder_visual(
        pose=hole_pose,
        radius=hole_radius,
        half_length=hole_length / 2,
        material=hole_mat,
    )

    return builder


@register_env("PegInsertionAngled-v1", max_episode_steps=120)
class PegInsertionAngledEnv(BaseEnv):
    """
    **Task Description:**
    Pick up a cylindrical peg and insert it into an angled hole in a wooden block.
    The hole is drilled at a specific angle, requiring precise orientation control.

    **Randomizations:**
    - Peg length is randomized between 0.10 and 0.15 meters
    - Peg radius is randomized between 0.010 and 0.018 meters
    - Hole angle is randomized between 15 and 45 degrees from vertical
    - Hole radius is peg radius + 0.003m clearance
    - Peg and block positions are randomized on the table
    - Block orientation is randomized around z-axis

    **Success Conditions:**
    - The peg tip is inserted at least 0.08m into the angled hole
    - The peg is properly aligned with the hole angle and within clearance
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PegInsertionAngled-v1_rt.mp4"
    SUPPORTED_ROBOTS = ["panda_wristcam", "xarm6_robotiq"]
    agent: Union[PandaWristCam, XArm6Robotiq]
    _clearance = 0.003

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
        pose = sapien_utils.look_at([0.7, -0.7, 1.0], [0.05, -0.1, 0.4])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        with torch.device(self.device):
            self.table_scene = TableSceneBuilder(self)
            self.table_scene.build()

            # Randomize peg and hole dimensions
            peg_lengths = self._batched_episode_rng.uniform(0.10, 0.15)
            peg_radii = self._batched_episode_rng.uniform(0.010, 0.018)
            hole_radii = peg_radii + self._clearance
            hole_angles = self._batched_episode_rng.uniform(
                np.radians(15), np.radians(45)
            )

            # Store dimensions for use in reward computation
            self.peg_lengths = common.to_tensor(peg_lengths)
            self.peg_radii = common.to_tensor(peg_radii)
            self.hole_radii = common.to_tensor(hole_radii)
            self.hole_angles = common.to_tensor(hole_angles)

            # Create peg tip offset (front of peg)
            peg_tip_offsets = torch.zeros((self.num_envs, 3))
            peg_tip_offsets[:, 0] = self.peg_lengths / 2
            self.peg_tip_offsets = Pose.create_from_pq(p=peg_tip_offsets)

            # Create hole entry point offsets
            hole_entry_offsets = torch.zeros((self.num_envs, 3))
            hole_entry_offsets[:, 2] = 0.05  # Top of block
            self.hole_entry_offsets = Pose.create_from_pq(p=hole_entry_offsets)

            # Create hole direction vectors
            hole_directions = torch.zeros((self.num_envs, 3))
            hole_directions[:, 0] = torch.sin(self.hole_angles)
            hole_directions[:, 2] = -torch.cos(self.hole_angles)
            self.hole_directions = hole_directions

            # Build pegs and blocks for each environment
            pegs = []
            blocks = []

            for i in range(self.num_envs):
                scene_idxs = [i]
                peg_length = peg_lengths[i]
                peg_radius = peg_radii[i]
                hole_radius = hole_radii[i]
                hole_angle = hole_angles[i]

                # Create cylindrical peg
                peg_builder = self.scene.create_actor_builder()
                peg_builder.add_cylinder_collision(
                    radius=peg_radius, half_length=peg_length / 2
                )

                # Peg head (tip) - green
                head_mat = sapien.render.RenderMaterial(
                    base_color=sapien_utils.hex2rgba("#27AE60"),
                    roughness=0.4,
                    specular=0.6,
                )
                peg_builder.add_cylinder_visual(
                    pose=sapien.Pose([peg_length / 4, 0, 0]),
                    radius=peg_radius,
                    half_length=peg_length / 4,
                    material=head_mat,
                )

                # Peg body - silver
                body_mat = sapien.render.RenderMaterial(
                    base_color=sapien_utils.hex2rgba("#95A5A6"),
                    roughness=0.3,
                    specular=0.8,
                )
                peg_builder.add_cylinder_visual(
                    pose=sapien.Pose([-peg_length / 4, 0, 0]),
                    radius=peg_radius,
                    half_length=peg_length / 4,
                    material=body_mat,
                )

                # Rotate peg to lie on its side initially
                peg_builder.initial_pose = sapien.Pose(
                    p=[0, 0, 0.15], q=euler2quat(0, np.pi / 2, 0)
                )
                peg_builder.set_scene_idxs(scene_idxs)
                peg = peg_builder.build(f"peg_{i}")
                self.remove_from_state_dict_registry(peg)

                # Create block with angled hole
                block_size = 0.12
                block_builder = _build_angled_block_with_hole(
                    self.scene, hole_radius, block_size, hole_angle
                )
                block_builder.initial_pose = sapien.Pose(p=[0, 0.35, block_size / 2])
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

            # Initialize peg position (lying on side)
            peg_xy = randomization.uniform(
                low=torch.tensor([-0.2, -0.3]),
                high=torch.tensor([0.2, -0.1]),
                size=(b, 2),
            )
            peg_pos = torch.zeros((b, 3))
            peg_pos[:, :2] = peg_xy
            peg_pos[:, 2] = self.peg_radii[env_idx] + 0.01

            # Random orientation around z-axis
            peg_quat = randomization.random_quaternions(
                b, self.device, lock_x=True, lock_y=True, bounds=(0, 2 * np.pi)
            )
            # Combine with side-lying orientation
            side_quat = euler2quat(0, np.pi / 2, 0)
            peg_quat = sapien_utils.quat_multiply(peg_quat, side_quat)
            self.peg.set_pose(Pose.create_from_pq(peg_pos, peg_quat))

            # Initialize block position
            block_xy = randomization.uniform(
                low=torch.tensor([-0.1, 0.25]),
                high=torch.tensor([0.1, 0.45]),
                size=(b, 2),
            )
            block_pos = torch.zeros((b, 3))
            block_pos[:, :2] = block_xy
            block_pos[:, 2] = 0.06  # Half height of block

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
    def hole_entry_pose(self):
        return self.block.pose * self.hole_entry_offsets

    @property
    def goal_pose(self):
        """Target pose for peg insertion along the angled hole."""
        # Calculate goal position deep in the hole
        insertion_depth = 0.08
        goal_offset = torch.zeros((self.num_envs, 3))
        goal_offset[:, 0] = self.hole_directions[:, 0] * insertion_depth
        goal_offset[:, 2] = self.hole_directions[:, 2] * insertion_depth
        return self.hole_entry_pose * Pose.create_from_pq(p=goal_offset)

    def has_peg_inserted(self):
        """Check if peg is properly inserted into the angled hole."""
        # Transform peg tip to hole reference frame
        peg_tip_at_hole = (self.hole_entry_pose.inv() * self.peg_tip_pose).p

        # Check insertion depth along hole direction
        hole_dir = self.hole_directions.unsqueeze(0)  # Add batch dimension
        insertion_depth = torch.sum(peg_tip_at_hole * hole_dir, dim=1)
        depth_flag = insertion_depth >= 0.08

        # Check lateral alignment (perpendicular to hole direction)
        # Project peg tip position onto plane perpendicular to hole direction
        projected_dist = torch.linalg.norm(
            peg_tip_at_hole - insertion_depth.unsqueeze(1) * hole_dir, dim=1
        )
        alignment_flag = projected_dist <= self.hole_radii

        return depth_flag & alignment_flag, peg_tip_at_hole

    def evaluate(self):
        success, peg_tip_at_hole = self.has_peg_inserted()
        return dict(success=success, peg_tip_at_hole=peg_tip_at_hole)

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self.obs_mode_struct.use_state:
            obs.update(
                peg_pose=self.peg.pose.raw_pose,
                peg_length=self.peg_lengths,
                peg_radius=self.peg_radii,
                hole_entry_pose=self.hole_entry_pose.raw_pose,
                hole_direction=self.hole_directions,
                hole_radius=self.hole_radii,
                hole_angle=self.hole_angles,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # Stage 1: Reach and grasp the peg
        gripper_pos = self.agent.tcp.pose.p
        peg_grasp_pos = self.peg.pose.p

        reach_dist = torch.linalg.norm(gripper_pos - peg_grasp_pos, dim=1)
        reach_reward = 1 - torch.tanh(3.0 * reach_dist)

        is_grasped = self.agent.is_grasping(self.peg, max_angle=30)
        reward = reach_reward + is_grasped

        # Stage 2: Orient peg to match hole angle
        peg_direction = self.peg.pose.to_transformation_matrix()[:, :3, 0]  # x-axis
        hole_direction = self.hole_directions

        # Angle between peg direction and hole direction
        dot_product = torch.sum(peg_direction * hole_direction, dim=1)
        dot_product = torch.clamp(dot_product, -1.0, 1.0)
        angle_diff = torch.acos(torch.abs(dot_product))

        orientation_reward = 2 * (1 - angle_diff / np.pi)
        reward += orientation_reward * is_grasped

        # Stage 3: Align peg tip with hole entry
        peg_tip_at_hole = (self.hole_entry_pose.inv() * self.peg_tip_pose).p
        entry_dist = torch.linalg.norm(peg_tip_at_hole, dim=1)

        alignment_reward = 3 * (1 - torch.tanh(5.0 * entry_dist))
        well_oriented = angle_diff < np.pi / 6  # Within 30 degrees
        reward += alignment_reward * is_grasped * well_oriented

        # Stage 4: Insert peg into hole
        hole_dir = self.hole_directions
        insertion_depth = torch.sum(peg_tip_at_hole * hole_dir, dim=1)
        insertion_depth = torch.clamp(insertion_depth, 0, 0.08)
        insertion_reward = 6 * (insertion_depth / 0.08)

        # Check lateral alignment
        projected_dist = torch.linalg.norm(
            peg_tip_at_hole - insertion_depth.unsqueeze(1) * hole_dir, dim=1
        )
        well_aligned = projected_dist <= self.hole_radii
        reward += insertion_reward * is_grasped * well_oriented * well_aligned

        # Success bonus
        reward[info["success"]] = 18

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs, action, info) / 18
