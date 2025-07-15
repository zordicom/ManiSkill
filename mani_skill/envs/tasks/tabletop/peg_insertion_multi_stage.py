"""
Copyright 2025 Zordi, Inc. All rights reserved.
"""

from typing import Any, Dict, Union

import numpy as np
import sapien
import torch

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


def _build_multi_stage_peg(
    scene: ManiSkillScene, segment_lengths: list, segment_radii: list
):
    """Build a multi-stage peg with different sized segments."""
    builder = scene.create_actor_builder()

    # Colors for different segments
    colors = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6"]

    z_offset = 0
    for i, (length, radius) in enumerate(zip(segment_lengths, segment_radii)):
        # Position segment
        segment_pose = sapien.Pose(p=[0, 0, z_offset + length / 2])

        # Add collision
        builder.add_cylinder_collision(
            pose=segment_pose, radius=radius, half_length=length / 2
        )

        # Add visual
        mat = sapien.render.RenderMaterial(
            base_color=sapien_utils.hex2rgba(colors[i % len(colors)]),
            roughness=0.4,
            specular=0.6,
        )
        builder.add_cylinder_visual(
            pose=segment_pose, radius=radius, half_length=length / 2, material=mat
        )

        z_offset += length

    return builder


def _build_multi_stage_hole_block(
    scene: ManiSkillScene, hole_lengths: list, hole_radii: list
):
    """Build a block with multi-stage hole (different sized segments)."""
    builder = scene.create_actor_builder()

    # Block dimensions
    max_radius = max(hole_radii)
    block_size = max_radius * 2.5
    total_hole_depth = sum(hole_lengths)
    block_height = total_hole_depth + 0.04

    # Material
    mat = sapien.render.RenderMaterial(
        base_color=sapien_utils.hex2rgba("#8B4513"), roughness=0.6, specular=0.3
    )

    # Create the block structure around the multi-stage hole
    wall_thickness = 0.02

    # Bottom of block
    bottom_pose = sapien.Pose(p=[0, 0, -block_height / 2 + wall_thickness / 2])
    builder.add_box_collision(
        pose=bottom_pose, half_size=[block_size / 2, block_size / 2, wall_thickness / 2]
    )
    builder.add_box_visual(
        pose=bottom_pose,
        half_size=[block_size / 2, block_size / 2, wall_thickness / 2],
        material=mat,
    )

    # Create walls for each stage of the hole
    z_offset = block_height / 2
    for i, (length, radius) in enumerate(zip(hole_lengths, hole_radii)):
        stage_z = z_offset - length / 2

        # Create cylindrical wall segments around this stage
        num_segments = 16
        for j in range(num_segments):
            angle = j * 2 * np.pi / num_segments

            # Inner wall (around hole)
            inner_wall_radius = radius + wall_thickness / 2
            inner_x = inner_wall_radius * np.cos(angle)
            inner_y = inner_wall_radius * np.sin(angle)

            inner_pose = sapien.Pose(p=[inner_x, inner_y, stage_z])
            builder.add_box_collision(
                pose=inner_pose,
                half_size=[wall_thickness / 2, wall_thickness / 2, length / 2],
            )
            builder.add_box_visual(
                pose=inner_pose,
                half_size=[wall_thickness / 2, wall_thickness / 2, length / 2],
                material=mat,
            )

            # Outer wall (if this is the largest segment)
            if i == 0:  # Only for the first (largest) segment
                outer_wall_radius = block_size / 2 - wall_thickness / 2
                outer_x = outer_wall_radius * np.cos(angle)
                outer_y = outer_wall_radius * np.sin(angle)

                outer_pose = sapien.Pose(p=[outer_x, outer_y, stage_z])
                builder.add_box_collision(
                    pose=outer_pose,
                    half_size=[wall_thickness / 2, wall_thickness / 2, length / 2],
                )
                builder.add_box_visual(
                    pose=outer_pose,
                    half_size=[wall_thickness / 2, wall_thickness / 2, length / 2],
                    material=mat,
                )

        z_offset -= length

    return builder


@register_env("PegInsertionMultiStage-v1", max_episode_steps=250)
class PegInsertionMultiStageEnv(BaseEnv):
    """
    **Task Description:**
    Pick up a multi-stage peg and insert it into a matching multi-stage hole.
    The peg has multiple segments of decreasing size that must be inserted in sequence.

    **Randomizations:**
    - Number of segments is randomized between 2 and 4
    - Each segment length is randomized between 0.03 and 0.06 meters
    - Each segment radius decreases by 0.005-0.010m from the previous segment
    - Base segment radius is randomized between 0.020 and 0.030 meters
    - Peg and block positions are randomized on the table
    - Block orientation is randomized around z-axis

    **Success Conditions:**
    - All segments are properly inserted into their corresponding holes
    - The peg is fully seated with the smallest segment at the bottom
    - Each segment is within its designated hole boundaries
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PegInsertionMultiStage-v1_rt.mp4"
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
        pose = sapien_utils.look_at([0, -0.5, 0.4], [0, 0, 0.15])
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

            # Randomize multi-stage peg dimensions
            num_segments = self._batched_episode_rng.integers(2, 5)  # 2-4 segments
            base_radii = self._batched_episode_rng.uniform(0.020, 0.030)

            # Store segment information for each environment
            self.num_segments = num_segments
            self.segment_lengths = []
            self.segment_radii = []
            self.hole_lengths = []
            self.hole_radii = []

            # Generate segment dimensions for each environment
            for i in range(self.num_envs):
                n_seg = (
                    num_segments[i]
                    if isinstance(num_segments, np.ndarray)
                    else num_segments
                )
                base_radius = (
                    base_radii[i] if isinstance(base_radii, np.ndarray) else base_radii
                )

                # Generate decreasing radii for segments
                seg_radii = []
                for j in range(n_seg):
                    radius_reduction = j * self._batched_episode_rng.uniform(
                        0.005, 0.010
                    )
                    seg_radii.append(base_radius - radius_reduction)

                # Generate lengths for each segment
                seg_lengths = self._batched_episode_rng.uniform(0.03, 0.06, size=n_seg)

                # Store for this environment
                self.segment_lengths.append(seg_lengths)
                self.segment_radii.append(seg_radii)
                self.hole_lengths.append(seg_lengths)
                self.hole_radii.append([r + self._clearance for r in seg_radii])

            # Convert to tensors where possible
            max_segments = (
                max(num_segments)
                if isinstance(num_segments, np.ndarray)
                else num_segments
            )
            self.max_segments = max_segments

            # Create segment tip offsets for each segment
            self.segment_tip_offsets = []
            for seg_idx in range(max_segments):
                offsets = torch.zeros((self.num_envs, 3))
                for env_idx in range(self.num_envs):
                    if seg_idx < len(self.segment_lengths[env_idx]):
                        # Calculate cumulative length up to this segment
                        cumulative_length = sum(
                            self.segment_lengths[env_idx][: seg_idx + 1]
                        )
                        offsets[env_idx, 2] = cumulative_length
                self.segment_tip_offsets.append(Pose.create_from_pq(p=offsets))

            # Create hole entry offset
            hole_entry_offsets = torch.zeros((self.num_envs, 3))
            for env_idx in range(self.num_envs):
                total_depth = sum(self.hole_lengths[env_idx])
                hole_entry_offsets[env_idx, 2] = (total_depth + 0.04) / 2
            self.hole_entry_offsets = Pose.create_from_pq(p=hole_entry_offsets)

            # Build pegs and blocks for each environment
            pegs = []
            blocks = []

            for i in range(self.num_envs):
                scene_idxs = [i]

                # Create multi-stage peg
                peg_builder = _build_multi_stage_peg(
                    self.scene, self.segment_lengths[i], self.segment_radii[i]
                )
                peg_builder.initial_pose = sapien.Pose(p=[0, 0, 0.15])
                peg_builder.set_scene_idxs(scene_idxs)
                peg = peg_builder.build(f"peg_{i}")
                self.remove_from_state_dict_registry(peg)

                # Create block with multi-stage hole
                block_builder = _build_multi_stage_hole_block(
                    self.scene, self.hole_lengths[i], self.hole_radii[i]
                )
                total_depth = sum(self.hole_lengths[i])
                block_builder.initial_pose = sapien.Pose(
                    p=[0, 0.4, (total_depth + 0.04) / 2]
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

            # Initialize peg position
            peg_xy = randomization.uniform(
                low=torch.tensor([-0.2, -0.3]),
                high=torch.tensor([0.2, -0.1]),
                size=(b, 2),
            )
            peg_pos = torch.zeros((b, 3))
            peg_pos[:, :2] = peg_xy

            # Set peg height based on total length
            for i, env_i in enumerate(env_idx):
                total_length = sum(self.segment_lengths[env_i])
                peg_pos[i, 2] = total_length / 2 + 0.01

            # Random orientation around z-axis
            peg_quat = randomization.random_quaternions(
                b, self.device, lock_x=True, lock_y=True, bounds=(0, 2 * np.pi)
            )
            self.peg.set_pose(Pose.create_from_pq(peg_pos, peg_quat))

            # Initialize block position
            block_xy = randomization.uniform(
                low=torch.tensor([-0.1, 0.3]),
                high=torch.tensor([0.1, 0.5]),
                size=(b, 2),
            )
            block_pos = torch.zeros((b, 3))
            block_pos[:, :2] = block_xy

            # Set block height based on total hole depth
            for i, env_i in enumerate(env_idx):
                total_depth = sum(self.hole_lengths[env_i])
                block_pos[i, 2] = (total_depth + 0.04) / 2

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
    def hole_entry_pose(self):
        return self.block.pose * self.hole_entry_offsets

    def get_segment_tip_poses(self, seg_idx: int):
        """Get the tip poses for a specific segment."""
        if seg_idx < len(self.segment_tip_offsets):
            return self.peg.pose * self.segment_tip_offsets[seg_idx]
        return None

    def has_peg_inserted(self):
        """Check if multi-stage peg is properly inserted."""
        success_flags = torch.ones(self.num_envs, dtype=torch.bool)
        segment_positions = {}

        for env_idx in range(self.num_envs):
            n_segments = len(self.segment_lengths[env_idx])

            # Check each segment
            for seg_idx in range(n_segments):
                segment_tip_pose = self.get_segment_tip_poses(seg_idx)
                if segment_tip_pose is None:
                    continue

                # Transform segment tip to hole reference frame
                segment_tip_at_hole = (self.hole_entry_pose.inv() * segment_tip_pose).p[
                    env_idx
                ]

                # Calculate expected depth for this segment
                expected_depth = sum(self.segment_lengths[env_idx][: seg_idx + 1])

                # Check if segment is inserted deep enough
                depth_flag = segment_tip_at_hole[2] <= (-expected_depth + 0.01)

                # Check if segment is within radius bounds
                lateral_dist = torch.linalg.norm(segment_tip_at_hole[:2])
                radius_flag = lateral_dist <= self.hole_radii[env_idx][seg_idx]

                if not (depth_flag and radius_flag):
                    success_flags[env_idx] = False

                segment_positions[f"segment_{seg_idx}"] = segment_tip_at_hole

        return success_flags, segment_positions

    def evaluate(self):
        success, segment_positions = self.has_peg_inserted()
        return dict(success=success, **segment_positions)

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self.obs_mode_struct.use_state:
            obs.update(
                peg_pose=self.peg.pose.raw_pose,
                hole_entry_pose=self.hole_entry_pose.raw_pose,
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
        target_height = 0.15
        lifted = peg_height > target_height
        lift_reward = 3 * lifted.float()
        reward += lift_reward * is_grasped

        # Stage 3: Align peg with hole entry
        # Use the first segment (largest) for initial alignment
        first_segment_pose = self.get_segment_tip_poses(0)
        if first_segment_pose is not None:
            segment_at_hole = (self.hole_entry_pose.inv() * first_segment_pose).p
            xy_alignment = torch.linalg.norm(segment_at_hole[:, :2], dim=1)

            alignment_reward = 4 * (1 - torch.tanh(10.0 * xy_alignment))
            reward += alignment_reward * is_grasped * lifted

        # Stage 4: Progressive insertion rewards for each segment
        total_insertion_reward = 0
        for env_idx in range(self.num_envs):
            n_segments = len(self.segment_lengths[env_idx])

            for seg_idx in range(n_segments):
                segment_tip_pose = self.get_segment_tip_poses(seg_idx)
                if segment_tip_pose is None:
                    continue

                segment_at_hole = (self.hole_entry_pose.inv() * segment_tip_pose).p[
                    env_idx
                ]

                # Expected depth for this segment
                expected_depth = sum(self.segment_lengths[env_idx][: seg_idx + 1])

                # Calculate insertion progress
                insertion_depth = torch.clamp(-segment_at_hole[2], 0, expected_depth)
                segment_insertion_reward = (insertion_depth / expected_depth) * 3

                # Check if segment is aligned
                lateral_dist = torch.linalg.norm(segment_at_hole[:2])
                segment_aligned = lateral_dist <= self.hole_radii[env_idx][seg_idx]

                if segment_aligned:
                    total_insertion_reward += segment_insertion_reward

        # Average insertion reward across environments
        avg_insertion_reward = total_insertion_reward / self.num_envs
        well_positioned = (
            xy_alignment < 0.01 if first_segment_pose is not None else True
        )
        reward += avg_insertion_reward * is_grasped * well_positioned

        # Success bonus
        reward[info["success"]] = 40

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs, action, info) / 40
