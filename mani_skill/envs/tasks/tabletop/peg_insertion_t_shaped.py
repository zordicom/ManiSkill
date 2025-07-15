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


def _build_t_shaped_peg(
    scene: ManiSkillScene, stem_length: float, cross_length: float, radius: float
):
    """Build a T-shaped peg with a vertical stem and horizontal crossbar."""
    builder = scene.create_actor_builder()

    # Vertical stem
    stem_pose = sapien.Pose(p=[0, 0, stem_length / 2])
    builder.add_cylinder_collision(
        pose=stem_pose, radius=radius, half_length=stem_length / 2
    )

    # Horizontal crossbar
    crossbar_pose = sapien.Pose(p=[0, 0, stem_length], q=euler2quat(0, np.pi / 2, 0))
    builder.add_cylinder_collision(
        pose=crossbar_pose, radius=radius, half_length=cross_length / 2
    )

    # Visual materials
    stem_mat = sapien.render.RenderMaterial(
        base_color=sapien_utils.hex2rgba("#E74C3C"), roughness=0.4, specular=0.6
    )
    crossbar_mat = sapien.render.RenderMaterial(
        base_color=sapien_utils.hex2rgba("#2ECC71"), roughness=0.4, specular=0.6
    )

    # Add visual components
    builder.add_cylinder_visual(
        pose=stem_pose, radius=radius, half_length=stem_length / 2, material=stem_mat
    )
    builder.add_cylinder_visual(
        pose=crossbar_pose,
        radius=radius,
        half_length=cross_length / 2,
        material=crossbar_mat,
    )

    return builder


def _build_t_shaped_hole_block(
    scene: ManiSkillScene, stem_length: float, cross_length: float, hole_radius: float
):
    """Build a block with a T-shaped hole."""
    builder = scene.create_actor_builder()

    # Main block dimensions
    block_width = cross_length + 0.06  # Extra margin
    block_depth = 0.06  # Depth of block
    block_height = stem_length + 0.04  # Height to accommodate stem

    # Create the main block material
    main_mat = sapien.render.RenderMaterial(
        base_color=sapien_utils.hex2rgba("#8B4513"), roughness=0.6, specular=0.3
    )

    # Build block using multiple segments to approximate the T-shaped hole
    segment_thickness = 0.02

    # Bottom part of the block (base)
    bottom_pose = sapien.Pose(p=[0, 0, -block_height / 2 + segment_thickness / 2])
    builder.add_box_collision(
        pose=bottom_pose,
        half_size=[block_width / 2, block_depth / 2, segment_thickness / 2],
    )
    builder.add_box_visual(
        pose=bottom_pose,
        half_size=[block_width / 2, block_depth / 2, segment_thickness / 2],
        material=main_mat,
    )

    # Create walls around the T-shaped hole
    # Vertical walls around stem
    stem_wall_height = stem_length / 2
    stem_wall_poses = [
        sapien.Pose(p=[hole_radius + segment_thickness / 2, 0, stem_wall_height / 2]),
        sapien.Pose(p=[-hole_radius - segment_thickness / 2, 0, stem_wall_height / 2]),
        sapien.Pose(p=[0, hole_radius + segment_thickness / 2, stem_wall_height / 2]),
        sapien.Pose(p=[0, -hole_radius - segment_thickness / 2, stem_wall_height / 2]),
    ]

    for pose in stem_wall_poses:
        builder.add_box_collision(
            pose=pose,
            half_size=[
                segment_thickness / 2,
                segment_thickness / 2,
                stem_wall_height / 2,
            ],
        )
        builder.add_box_visual(
            pose=pose,
            half_size=[
                segment_thickness / 2,
                segment_thickness / 2,
                stem_wall_height / 2,
            ],
            material=main_mat,
        )

    # Horizontal walls around crossbar
    crossbar_wall_height = segment_thickness
    crossbar_wall_z = stem_length - crossbar_wall_height / 2

    # Left and right walls of crossbar
    crossbar_wall_poses = [
        sapien.Pose(p=[cross_length / 2 + segment_thickness / 2, 0, crossbar_wall_z]),
        sapien.Pose(p=[-cross_length / 2 - segment_thickness / 2, 0, crossbar_wall_z]),
    ]

    for pose in crossbar_wall_poses:
        builder.add_box_collision(
            pose=pose,
            half_size=[
                segment_thickness / 2,
                hole_radius + segment_thickness / 2,
                crossbar_wall_height / 2,
            ],
        )
        builder.add_box_visual(
            pose=pose,
            half_size=[
                segment_thickness / 2,
                hole_radius + segment_thickness / 2,
                crossbar_wall_height / 2,
            ],
            material=main_mat,
        )

    # Top walls around crossbar
    crossbar_top_poses = [
        sapien.Pose(p=[0, hole_radius + segment_thickness / 2, crossbar_wall_z]),
        sapien.Pose(p=[0, -hole_radius - segment_thickness / 2, crossbar_wall_z]),
    ]

    for pose in crossbar_top_poses:
        builder.add_box_collision(
            pose=pose,
            half_size=[
                cross_length / 2,
                segment_thickness / 2,
                crossbar_wall_height / 2,
            ],
        )
        builder.add_box_visual(
            pose=pose,
            half_size=[
                cross_length / 2,
                segment_thickness / 2,
                crossbar_wall_height / 2,
            ],
            material=main_mat,
        )

    return builder


@register_env("PegInsertionTShaped-v1", max_episode_steps=180)
class PegInsertionTShapedEnv(BaseEnv):
    """
    **Task Description:**
    Pick up a T-shaped peg and insert it into a matching T-shaped hole in a wooden block.
    The peg has a vertical stem and horizontal crossbar forming a T-shape.

    **Randomizations:**
    - Stem length is randomized between 0.08 and 0.12 meters
    - Crossbar length is randomized between 0.06 and 0.10 meters
    - Peg radius is randomized between 0.008 and 0.012 meters
    - Hole radius is peg radius + 0.002m clearance
    - Peg and block positions are randomized on the table
    - Block orientation is randomized around z-axis

    **Success Conditions:**
    - The stem is fully inserted into the vertical hole
    - The crossbar is properly positioned in the horizontal slot
    - The peg is fully seated with proper alignment
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PegInsertionTShaped-v1_rt.mp4"
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
        pose = sapien_utils.look_at([0, -0.6, 0.5], [0, 0, 0.15])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.9, -0.9, 1.3], [0.05, -0.1, 0.4])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        with torch.device(self.device):
            self.table_scene = TableSceneBuilder(self)
            self.table_scene.build()

            # Randomize T-shaped peg dimensions
            stem_lengths = self._batched_episode_rng.uniform(0.08, 0.12)
            cross_lengths = self._batched_episode_rng.uniform(0.06, 0.10)
            peg_radii = self._batched_episode_rng.uniform(0.008, 0.012)
            hole_radii = peg_radii + self._clearance

            # Store dimensions
            self.stem_lengths = common.to_tensor(stem_lengths)
            self.cross_lengths = common.to_tensor(cross_lengths)
            self.peg_radii = common.to_tensor(peg_radii)
            self.hole_radii = common.to_tensor(hole_radii)

            # Create key point offsets
            # Stem tip (bottom of stem)
            stem_tip_offsets = torch.zeros((self.num_envs, 3))
            stem_tip_offsets[:, 2] = 0  # Bottom of stem
            self.stem_tip_offsets = Pose.create_from_pq(p=stem_tip_offsets)

            # Crossbar center (center of crossbar)
            crossbar_center_offsets = torch.zeros((self.num_envs, 3))
            crossbar_center_offsets[:, 2] = self.stem_lengths
            self.crossbar_center_offsets = Pose.create_from_pq(
                p=crossbar_center_offsets
            )

            # Crossbar tips (ends of crossbar)
            crossbar_left_tip_offsets = torch.zeros((self.num_envs, 3))
            crossbar_left_tip_offsets[:, 0] = -self.cross_lengths / 2
            crossbar_left_tip_offsets[:, 2] = self.stem_lengths
            self.crossbar_left_tip_offsets = Pose.create_from_pq(
                p=crossbar_left_tip_offsets
            )

            crossbar_right_tip_offsets = torch.zeros((self.num_envs, 3))
            crossbar_right_tip_offsets[:, 0] = self.cross_lengths / 2
            crossbar_right_tip_offsets[:, 2] = self.stem_lengths
            self.crossbar_right_tip_offsets = Pose.create_from_pq(
                p=crossbar_right_tip_offsets
            )

            # Hole entry point (top of block)
            hole_entry_offsets = torch.zeros((self.num_envs, 3))
            hole_entry_offsets[:, 2] = (self.stem_lengths + 0.04) / 2
            self.hole_entry_offsets = Pose.create_from_pq(p=hole_entry_offsets)

            # Build pegs and blocks for each environment
            pegs = []
            blocks = []

            for i in range(self.num_envs):
                scene_idxs = [i]
                stem_length = stem_lengths[i]
                cross_length = cross_lengths[i]
                peg_radius = peg_radii[i]
                hole_radius = hole_radii[i]

                # Create T-shaped peg
                peg_builder = _build_t_shaped_peg(
                    self.scene, stem_length, cross_length, peg_radius
                )
                peg_builder.initial_pose = sapien.Pose(p=[0, 0, 0.15])
                peg_builder.set_scene_idxs(scene_idxs)
                peg = peg_builder.build(f"peg_{i}")
                self.remove_from_state_dict_registry(peg)

                # Create block with T-shaped hole
                block_builder = _build_t_shaped_hole_block(
                    self.scene, stem_length, cross_length, hole_radius
                )
                block_builder.initial_pose = sapien.Pose(
                    p=[0, 0.45, (stem_length + 0.04) / 2]
                )
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
                low=torch.tensor([-0.25, -0.35]),
                high=torch.tensor([0.25, -0.05]),
                size=(b, 2),
            )
            peg_pos = torch.zeros((b, 3))
            peg_pos[:, :2] = peg_xy
            peg_pos[:, 2] = 0.06  # Above table surface

            # Random orientation - lying on side
            peg_quat = randomization.random_quaternions(
                b, self.device, lock_x=False, lock_y=False, bounds=(0, 2 * np.pi)
            )
            # Rotate to lie on side
            side_quat = euler2quat(np.pi / 2, 0, 0)
            peg_quat = sapien_utils.quat_multiply(peg_quat, side_quat)
            self.peg.set_pose(Pose.create_from_pq(peg_pos, peg_quat))

            # Initialize block position
            block_xy = randomization.uniform(
                low=torch.tensor([-0.1, 0.35]),
                high=torch.tensor([0.1, 0.55]),
                size=(b, 2),
            )
            block_pos = torch.zeros((b, 3))
            block_pos[:, :2] = block_xy
            block_pos[:, 2] = (self.stem_lengths[env_idx] + 0.04) / 2

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
    def stem_tip_pose(self):
        return self.peg.pose * self.stem_tip_offsets

    @property
    def crossbar_center_pose(self):
        return self.peg.pose * self.crossbar_center_offsets

    @property
    def crossbar_left_tip_pose(self):
        return self.peg.pose * self.crossbar_left_tip_offsets

    @property
    def crossbar_right_tip_pose(self):
        return self.peg.pose * self.crossbar_right_tip_offsets

    @property
    def hole_entry_pose(self):
        return self.block.pose * self.hole_entry_offsets

    def has_peg_inserted(self):
        """Check if T-shaped peg is properly inserted into the hole."""
        # Transform key points to hole reference frame
        stem_tip_at_hole = (self.hole_entry_pose.inv() * self.stem_tip_pose).p
        crossbar_center_at_hole = (
            self.hole_entry_pose.inv() * self.crossbar_center_pose
        ).p
        crossbar_left_at_hole = (
            self.hole_entry_pose.inv() * self.crossbar_left_tip_pose
        ).p
        crossbar_right_at_hole = (
            self.hole_entry_pose.inv() * self.crossbar_right_tip_pose
        ).p

        # Check stem insertion
        stem_depth = -stem_tip_at_hole[:, 2]  # Negative z is down
        stem_inserted = stem_depth >= (self.stem_lengths - 0.01)

        # Check stem lateral alignment
        stem_lateral_dist = torch.linalg.norm(stem_tip_at_hole[:, :2], dim=1)
        stem_aligned = stem_lateral_dist <= self.hole_radii

        # Check crossbar position at top of hole
        crossbar_height = crossbar_center_at_hole[:, 2]
        crossbar_at_top = torch.abs(crossbar_height) <= 0.01

        # Check crossbar lateral alignment
        crossbar_center_lateral = torch.linalg.norm(
            crossbar_center_at_hole[:, :2], dim=1
        )
        crossbar_center_aligned = crossbar_center_lateral <= self.hole_radii

        # Check crossbar tips are within bounds
        crossbar_left_lateral = torch.linalg.norm(crossbar_left_at_hole[:, :2], dim=1)
        crossbar_right_lateral = torch.linalg.norm(crossbar_right_at_hole[:, :2], dim=1)
        crossbar_tips_bounded = (
            crossbar_left_lateral <= self.cross_lengths / 2 + 0.01
        ) & (crossbar_right_lateral <= self.cross_lengths / 2 + 0.01)

        success = (
            stem_inserted
            & stem_aligned
            & crossbar_at_top
            & crossbar_center_aligned
            & crossbar_tips_bounded
        )

        return success, {
            "stem_tip": stem_tip_at_hole,
            "crossbar_center": crossbar_center_at_hole,
            "crossbar_left": crossbar_left_at_hole,
            "crossbar_right": crossbar_right_at_hole,
        }

    def evaluate(self):
        success, positions = self.has_peg_inserted()
        return dict(success=success, **positions)

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self.obs_mode_struct.use_state:
            obs.update(
                peg_pose=self.peg.pose.raw_pose,
                stem_length=self.stem_lengths,
                cross_length=self.cross_lengths,
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

        # Stage 2: Lift and orient the peg correctly (upright)
        peg_height = self.peg.pose.p[:, 2]
        lifted = peg_height > 0.18

        # Check if peg is upright (z-axis pointing up)
        peg_z_axis = self.peg.pose.to_transformation_matrix()[:, :3, 2]
        upright_alignment = torch.sum(
            peg_z_axis * torch.tensor([0, 0, 1], device=self.device), dim=1
        )
        upright_reward = 3 * torch.clamp(upright_alignment, 0, 1)

        reward += upright_reward * is_grasped * lifted

        # Stage 3: Align peg with hole entry
        stem_tip_at_hole = (self.hole_entry_pose.inv() * self.stem_tip_pose).p
        alignment_dist = torch.linalg.norm(stem_tip_at_hole[:, :2], dim=1)

        alignment_reward = 4 * (1 - torch.tanh(10.0 * alignment_dist))
        well_oriented = upright_alignment > 0.9
        reward += alignment_reward * is_grasped * well_oriented

        # Stage 4: Insert stem into hole
        stem_depth = torch.clamp(-stem_tip_at_hole[:, 2], 0, self.stem_lengths)
        stem_insertion_reward = 5 * (stem_depth / self.stem_lengths)

        well_aligned = alignment_dist <= self.hole_radii
        reward += stem_insertion_reward * is_grasped * well_oriented * well_aligned

        # Stage 5: Position crossbar at top of hole
        crossbar_center_at_hole = (
            self.hole_entry_pose.inv() * self.crossbar_center_pose
        ).p
        crossbar_height_error = torch.abs(crossbar_center_at_hole[:, 2])
        crossbar_position_reward = 3 * (1 - torch.tanh(15.0 * crossbar_height_error))

        stem_mostly_inserted = stem_depth > (self.stem_lengths * 0.8)
        reward += crossbar_position_reward * is_grasped * stem_mostly_inserted

        # Stage 6: Align crossbar laterally
        crossbar_lateral_dist = torch.linalg.norm(crossbar_center_at_hole[:, :2], dim=1)
        crossbar_alignment_reward = 2 * (1 - torch.tanh(8.0 * crossbar_lateral_dist))

        crossbar_near_top = crossbar_height_error < 0.02
        reward += crossbar_alignment_reward * is_grasped * crossbar_near_top

        # Success bonus
        reward[info["success"]] = 30

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs, action, info) / 30
