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
from mani_skill.utils.geometry.rotation_conversions import quaternion_multiply
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Actor, Pose
from mani_skill.utils.structs.types import SimConfig


def _build_l_shaped_peg(
    scene: ManiSkillScene, length1: float, length2: float, radius: float
):
    """Build an L-shaped peg with two perpendicular cylindrical segments."""
    builder = scene.create_actor_builder()

    # Vertical segment
    vertical_pose = sapien.Pose(p=[0, 0, length1 / 2])
    builder.add_cylinder_collision(
        pose=vertical_pose, radius=radius, half_length=length1 / 2
    )

    # Horizontal segment
    horizontal_pose = sapien.Pose(
        p=[length2 / 2, 0, length1], q=euler2quat(0, np.pi / 2, 0)
    )
    builder.add_cylinder_collision(
        pose=horizontal_pose, radius=radius, half_length=length2 / 2
    )

    # Visual materials
    vertical_mat = sapien.render.RenderMaterial(
        base_color=sapien_utils.hex2rgba("#E67E22"), roughness=0.4, specular=0.6
    )
    horizontal_mat = sapien.render.RenderMaterial(
        base_color=sapien_utils.hex2rgba("#3498DB"), roughness=0.4, specular=0.6
    )

    # Add visual components
    builder.add_cylinder_visual(
        pose=vertical_pose,
        radius=radius,
        half_length=length1 / 2,
        material=vertical_mat,
    )
    builder.add_cylinder_visual(
        pose=horizontal_pose,
        radius=radius,
        half_length=length2 / 2,
        material=horizontal_mat,
    )

    return builder


def _build_l_shaped_hole_block(
    scene: ManiSkillScene, hole_length1: float, hole_length2: float, hole_radius: float
):
    """Build a block with an L-shaped hole."""
    builder = scene.create_actor_builder()

    # Main block dimensions
    block_size = max(hole_length1, hole_length2) + 0.06  # Extra margin
    block_height = hole_length1 + 0.04

    # Create the main block
    half_size = [block_size / 2, block_size / 2, block_height / 2]
    main_mat = sapien.render.RenderMaterial(
        base_color=sapien_utils.hex2rgba("#8B4513"), roughness=0.6, specular=0.3
    )

    # Build block using multiple segments to approximate the L-shaped hole
    # We'll create the solid parts around the hole
    segment_thickness = 0.02

    # Bottom part of the block (below vertical hole)
    bottom_pose = sapien.Pose(p=[0, 0, -block_height / 2 + segment_thickness / 2])
    builder.add_box_collision(
        pose=bottom_pose,
        half_size=[block_size / 2, block_size / 2, segment_thickness / 2],
    )
    builder.add_box_visual(
        pose=bottom_pose,
        half_size=[block_size / 2, block_size / 2, segment_thickness / 2],
        material=main_mat,
    )

    # Sides around vertical hole
    side_poses = [
        sapien.Pose(p=[hole_radius + segment_thickness / 2, 0, 0]),
        sapien.Pose(p=[-hole_radius - segment_thickness / 2, 0, 0]),
        sapien.Pose(p=[0, hole_radius + segment_thickness / 2, 0]),
        sapien.Pose(p=[0, -hole_radius - segment_thickness / 2, 0]),
    ]

    for pose in side_poses:
        builder.add_box_collision(
            pose=pose,
            half_size=[segment_thickness / 2, segment_thickness / 2, block_height / 2],
        )
        builder.add_box_visual(
            pose=pose,
            half_size=[segment_thickness / 2, segment_thickness / 2, block_height / 2],
            material=main_mat,
        )

    # Top part around horizontal hole
    top_pose = sapien.Pose(p=[0, 0, block_height / 2 - segment_thickness / 2])
    builder.add_box_collision(
        pose=top_pose, half_size=[block_size / 2, block_size / 2, segment_thickness / 2]
    )
    builder.add_box_visual(
        pose=top_pose,
        half_size=[block_size / 2, block_size / 2, segment_thickness / 2],
        material=main_mat,
    )

    return builder


@register_env("PegInsertionLShaped-v1", max_episode_steps=150)
class PegInsertionLShapedEnv(BaseEnv):
    """
    **Task Description:**
    Pick up an L-shaped peg and insert it into a matching L-shaped hole in a wooden block.
    The peg has two perpendicular cylindrical segments forming an L-shape.

    **Randomizations:**
    - Vertical segment length is randomized between 0.06 and 0.10 meters
    - Horizontal segment length is randomized between 0.04 and 0.08 meters
    - Peg radius is randomized between 0.008 and 0.015 meters
    - Hole radius is peg radius + 0.002m clearance
    - Peg and block positions are randomized on the table
    - Block orientation is randomized around z-axis

    **Success Conditions:**
    - Both segments of the L-shaped peg are properly inserted into the hole
    - The peg is fully seated with the horizontal segment at the top
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PegInsertionLShaped-v1_rt.mp4"
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
        pose = sapien_utils.look_at([0, -0.5, 0.5], [0, 0, 0.15])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.8, -0.8, 1.2], [0.05, -0.1, 0.4])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        with torch.device(self.device):
            self.table_scene = TableSceneBuilder(self)
            self.table_scene.build()

            # Randomize L-shaped peg dimensions
            length1 = self._batched_episode_rng.uniform(0.06, 0.10)  # Vertical segment
            length2 = self._batched_episode_rng.uniform(
                0.04, 0.08
            )  # Horizontal segment
            peg_radii = self._batched_episode_rng.uniform(0.008, 0.015)
            hole_radii = peg_radii + self._clearance

            # Store dimensions
            self.peg_length1 = common.to_tensor(length1)
            self.peg_length2 = common.to_tensor(length2)
            self.peg_radii = common.to_tensor(peg_radii)
            self.hole_radii = common.to_tensor(hole_radii)

            # Create key point offsets
            # Vertical segment tip (bottom of peg)
            vertical_tip_offsets = torch.zeros((self.num_envs, 3))
            vertical_tip_offsets[:, 2] = 0  # Bottom of vertical segment
            self.vertical_tip_offsets = Pose.create_from_pq(p=vertical_tip_offsets)

            # Horizontal segment tip (end of horizontal segment)
            horizontal_tip_offsets = torch.zeros((self.num_envs, 3))
            horizontal_tip_offsets[:, 0] = self.peg_length2
            horizontal_tip_offsets[:, 2] = self.peg_length1
            self.horizontal_tip_offsets = Pose.create_from_pq(p=horizontal_tip_offsets)

            # L-joint position (corner of the L)
            l_joint_offsets = torch.zeros((self.num_envs, 3))
            l_joint_offsets[:, 2] = self.peg_length1
            self.l_joint_offsets = Pose.create_from_pq(p=l_joint_offsets)

            # Hole entry point (top of block)
            hole_entry_offsets = torch.zeros((self.num_envs, 3))
            hole_entry_offsets[:, 2] = (self.peg_length1 + 0.04) / 2
            self.hole_entry_offsets = Pose.create_from_pq(p=hole_entry_offsets)

            # Build pegs and blocks for each environment
            pegs = []
            blocks = []

            for i in range(self.num_envs):
                scene_idxs = [i]
                len1 = length1[i]
                len2 = length2[i]
                peg_radius = peg_radii[i]
                hole_radius = hole_radii[i]

                # Create L-shaped peg
                peg_builder = _build_l_shaped_peg(self.scene, len1, len2, peg_radius)
                peg_builder.initial_pose = sapien.Pose(p=[0, 0, 0.15])
                peg_builder.set_scene_idxs(scene_idxs)
                peg = peg_builder.build(f"peg_{i}")
                self.remove_from_state_dict_registry(peg)

                # Create block with L-shaped hole
                block_builder = _build_l_shaped_hole_block(
                    self.scene, len1, len2, hole_radius
                )
                block_builder.initial_pose = sapien.Pose(p=[0, 0.4, (len1 + 0.04) / 2])
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
            peg_pos[:, 2] = 0.05  # Above table surface

            # Random orientation - lying on side
            peg_quat = randomization.random_quaternions(
                b, self.device, lock_x=True, lock_y=False, bounds=(0, 2 * np.pi)
            )
            # Rotate to lie on side
            side_quat = euler2quat(0, np.pi / 2, 0)
            peg_quat = quaternion_multiply(peg_quat, side_quat)
            self.peg.set_pose(Pose.create_from_pq(peg_pos, peg_quat))

            # Initialize block position
            block_xy = randomization.uniform(
                low=torch.tensor([-0.1, 0.3]),
                high=torch.tensor([0.1, 0.5]),
                size=(b, 2),
            )
            block_pos = torch.zeros((b, 3))
            block_pos[:, :2] = block_xy
            block_pos[:, 2] = (self.peg_length1[env_idx] + 0.04) / 2

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
    def vertical_tip_pose(self):
        return self.peg.pose * self.vertical_tip_offsets

    @property
    def horizontal_tip_pose(self):
        return self.peg.pose * self.horizontal_tip_offsets

    @property
    def l_joint_pose(self):
        return self.peg.pose * self.l_joint_offsets

    @property
    def hole_entry_pose(self):
        return self.block.pose * self.hole_entry_offsets

    def has_peg_inserted(self):
        """Check if L-shaped peg is properly inserted into the hole."""
        # Transform key points to hole reference frame
        vertical_tip_at_hole = (self.hole_entry_pose.inv() * self.vertical_tip_pose).p
        horizontal_tip_at_hole = (
            self.hole_entry_pose.inv() * self.horizontal_tip_pose
        ).p
        l_joint_at_hole = (self.hole_entry_pose.inv() * self.l_joint_pose).p

        # Check vertical segment insertion
        vertical_depth = -vertical_tip_at_hole[:, 2]  # Negative z is down
        vertical_inserted = vertical_depth >= (self.peg_length1 - 0.01)

        # Check vertical segment lateral alignment
        vertical_lateral_dist = torch.linalg.norm(vertical_tip_at_hole[:, :2], dim=1)
        vertical_aligned = vertical_lateral_dist <= self.hole_radii

        # Check horizontal segment position
        horizontal_height = horizontal_tip_at_hole[:, 2]
        horizontal_at_top = torch.abs(horizontal_height) <= 0.01  # Near top of block

        # Check horizontal segment lateral alignment
        horizontal_lateral_dist = torch.linalg.norm(
            horizontal_tip_at_hole[:, :2], dim=1
        )
        horizontal_aligned = horizontal_lateral_dist <= self.hole_radii

        # Check L-joint position
        l_joint_height = l_joint_at_hole[:, 2]
        l_joint_at_top = torch.abs(l_joint_height) <= 0.01

        success = (
            vertical_inserted
            & vertical_aligned
            & horizontal_at_top
            & horizontal_aligned
            & l_joint_at_top
        )

        return success, {
            "vertical_tip": vertical_tip_at_hole,
            "horizontal_tip": horizontal_tip_at_hole,
            "l_joint": l_joint_at_hole,
        }

    def evaluate(self):
        success, positions = self.has_peg_inserted()
        return dict(success=success, **positions)

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self.obs_mode_struct.use_state:
            obs.update(
                peg_pose=self.peg.pose.raw_pose,
                peg_length1=self.peg_length1,
                peg_length2=self.peg_length2,
                peg_radius=self.peg_radii,
                hole_entry_pose=self.hole_entry_pose.raw_pose,
                hole_radius=self.hole_radii,
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

        # Stage 2: Lift and orient the peg correctly
        peg_height = self.peg.pose.p[:, 2]
        lifted = peg_height > 0.15

        # Check if peg is upright (z-axis of peg pointing up)
        peg_z_axis = self.peg.pose.to_transformation_matrix()[:, :3, 2]
        upright_alignment = torch.sum(
            peg_z_axis * torch.tensor([0, 0, 1], device=self.device), dim=1
        )
        upright_reward = 2 * torch.clamp(upright_alignment, 0, 1)

        reward += upright_reward * is_grasped * lifted

        # Stage 3: Align peg with hole entry
        vertical_tip_at_hole = (self.hole_entry_pose.inv() * self.vertical_tip_pose).p
        alignment_dist = torch.linalg.norm(vertical_tip_at_hole[:, :2], dim=1)

        alignment_reward = 3 * (1 - torch.tanh(8.0 * alignment_dist))
        well_oriented = upright_alignment > 0.8
        reward += alignment_reward * is_grasped * well_oriented

        # Stage 4: Insert vertical segment
        vertical_depth = torch.clamp(-vertical_tip_at_hole[:, 2], 0, self.peg_length1)
        vertical_insertion_reward = 4 * (vertical_depth / self.peg_length1)

        well_aligned = alignment_dist <= self.hole_radii
        reward += vertical_insertion_reward * is_grasped * well_oriented * well_aligned

        # Stage 5: Position horizontal segment at top
        horizontal_tip_at_hole = (
            self.hole_entry_pose.inv() * self.horizontal_tip_pose
        ).p
        horizontal_height_error = torch.abs(horizontal_tip_at_hole[:, 2])
        horizontal_position_reward = 2 * (
            1 - torch.tanh(10.0 * horizontal_height_error)
        )

        vertical_mostly_inserted = vertical_depth > (self.peg_length1 * 0.8)
        reward += horizontal_position_reward * is_grasped * vertical_mostly_inserted

        # Success bonus
        reward[info["success"]] = 25

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs, action, info) / 25
