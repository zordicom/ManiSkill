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


def _build_cylinder_with_hole(
    scene: ManiSkillScene, inner_radius: float, outer_radius: float, height: float
):
    """Build a cylindrical block with a circular hole through the center."""
    builder = scene.create_actor_builder()

    # Create hollow cylinder using multiple collision shapes
    # We'll approximate the hollow cylinder with multiple boxes arranged in a circle
    num_segments = 16
    segment_angle = 2 * np.pi / num_segments

    # Calculate segment dimensions
    segment_width = 2 * outer_radius * np.sin(segment_angle / 2)
    segment_depth = outer_radius - inner_radius
    segment_center_radius = (outer_radius + inner_radius) / 2

    mat = sapien.render.RenderMaterial(
        base_color=sapien_utils.hex2rgba("#8B4513"), roughness=0.6, specular=0.3
    )

    for i in range(num_segments):
        angle = i * segment_angle
        # Position segments around the circle
        x = segment_center_radius * np.cos(angle)
        y = segment_center_radius * np.sin(angle)

        # Create pose for this segment
        pose = sapien.Pose(p=[x, y, 0], q=euler2quat(0, 0, angle))

        half_size = [segment_depth / 2, segment_width / 2, height / 2]
        builder.add_box_collision(pose, half_size)
        builder.add_box_visual(pose, half_size, material=mat)

    return builder


@register_env("PegInsertionTop-v1", max_episode_steps=100)
class PegInsertionTopEnv(BaseEnv):
    """
    **Task Description:**
    Pick up a cylindrical peg and insert it top-down into a circular hole in a wooden block.

    **Randomizations:**
    - Peg height is randomized between 0.08 and 0.12 meters
    - Peg radius is randomized between 0.012 and 0.022 meters
    - Hole radius is peg radius + 0.002m clearance
    - Peg and block positions are randomized on the table
    - Peg and block orientations are randomized around z-axis

    **Success Conditions:**
    - The peg tip is inserted at least 0.06m into the hole
    - The peg is properly aligned with the hole (within clearance)
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PegInsertionTop-v1_rt.mp4"
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
        pose = sapien_utils.look_at([0, -0.4, 0.3], [0, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, -0.6, 0.9], [0.05, -0.1, 0.4])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        with torch.device(self.device):
            self.table_scene = TableSceneBuilder(self)
            self.table_scene.build()

            # Randomize peg and hole dimensions
            peg_heights = self._batched_episode_rng.uniform(0.08, 0.12)
            peg_radii = self._batched_episode_rng.uniform(0.012, 0.022)
            hole_radii = peg_radii + self._clearance

            # Store dimensions for use in reward computation
            self.peg_heights = common.to_tensor(peg_heights)
            self.peg_radii = common.to_tensor(peg_radii)
            self.hole_radii = common.to_tensor(hole_radii)

            # Create peg tip offset (bottom of peg)
            peg_tip_offsets = torch.zeros((self.num_envs, 3))
            peg_tip_offsets[:, 2] = -self.peg_heights / 2
            self.peg_tip_offsets = Pose.create_from_pq(p=peg_tip_offsets)

            # Create hole center offset
            hole_center_offsets = torch.zeros((self.num_envs, 3))
            self.hole_center_offsets = Pose.create_from_pq(p=hole_center_offsets)

            # Build pegs and blocks for each environment
            pegs = []
            blocks = []

            for i in range(self.num_envs):
                scene_idxs = [i]
                peg_height = peg_heights[i]
                peg_radius = peg_radii[i]
                hole_radius = hole_radii[i]

                # Create cylindrical peg
                peg_builder = self.scene.create_actor_builder()
                peg_builder.add_cylinder_collision(
                    radius=peg_radius, half_length=peg_height / 2
                )

                # Peg head (top) - blue
                head_mat = sapien.render.RenderMaterial(
                    base_color=sapien_utils.hex2rgba("#3498DB"),
                    roughness=0.4,
                    specular=0.6,
                )
                peg_builder.add_cylinder_visual(
                    pose=sapien.Pose([0, 0, peg_height / 4]),
                    radius=peg_radius,
                    half_length=peg_height / 4,
                    material=head_mat,
                )

                # Peg tail (bottom) - red
                tail_mat = sapien.render.RenderMaterial(
                    base_color=sapien_utils.hex2rgba("#E74C3C"),
                    roughness=0.4,
                    specular=0.6,
                )
                peg_builder.add_cylinder_visual(
                    pose=sapien.Pose([0, 0, -peg_height / 4]),
                    radius=peg_radius,
                    half_length=peg_height / 4,
                    material=tail_mat,
                )

                peg_builder.initial_pose = sapien.Pose(p=[0, 0, 0.15])
                peg_builder.set_scene_idxs(scene_idxs)
                peg = peg_builder.build(f"peg_{i}")
                self.remove_from_state_dict_registry(peg)

                # Create block with hole
                block_height = 0.05
                block_outer_radius = 0.08
                hole_builder = _build_cylinder_with_hole(
                    self.scene, hole_radius, block_outer_radius, block_height
                )
                hole_builder.initial_pose = sapien.Pose(p=[0, 0.3, block_height / 2])
                hole_builder.set_scene_idxs(scene_idxs)
                block = hole_builder.build_kinematic(f"block_{i}")
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
                low=torch.tensor([-0.15, -0.3]),
                high=torch.tensor([0.15, -0.1]),
                size=(b, 2),
            )
            peg_pos = torch.zeros((b, 3))
            peg_pos[:, :2] = peg_xy
            peg_pos[:, 2] = self.peg_heights[env_idx] / 2 + 0.01  # On table surface

            peg_quat = randomization.random_quaternions(
                b, self.device, lock_x=True, lock_y=True, bounds=(0, 2 * np.pi)
            )
            self.peg.set_pose(Pose.create_from_pq(peg_pos, peg_quat))

            # Initialize block position
            block_xy = randomization.uniform(
                low=torch.tensor([-0.1, 0.2]),
                high=torch.tensor([0.1, 0.4]),
                size=(b, 2),
            )
            block_pos = torch.zeros((b, 3))
            block_pos[:, :2] = block_xy
            block_pos[:, 2] = 0.025  # Half height of block

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
                    0.0,  # 6 arm joints
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,  # 6 gripper joints (open)
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
    def goal_pose(self):
        # Target pose for peg insertion
        goal_offset = torch.zeros((self.num_envs, 3))
        goal_offset[:, 2] = -0.06  # Insert 6cm into hole
        return self.hole_center_pose * Pose.create_from_pq(p=goal_offset)

    def has_peg_inserted(self):
        """Check if peg is properly inserted into hole."""
        peg_tip_at_hole = (self.hole_center_pose.inv() * self.peg_tip_pose).p

        # Check if peg is inserted deep enough (z-axis is up)
        depth_flag = peg_tip_at_hole[:, 2] <= -0.06

        # Check if peg is within hole radius in xy plane
        xy_dist = torch.linalg.norm(peg_tip_at_hole[:, :2], dim=1)
        radius_flag = xy_dist <= self.hole_radii

        return depth_flag & radius_flag, peg_tip_at_hole

    def evaluate(self):
        success, peg_tip_at_hole = self.has_peg_inserted()
        return dict(success=success, peg_tip_at_hole=peg_tip_at_hole)

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self.obs_mode_struct.use_state:
            obs.update(
                peg_pose=self.peg.pose.raw_pose,
                peg_height=self.peg_heights,
                peg_radius=self.peg_radii,
                hole_center_pose=self.hole_center_pose.raw_pose,
                hole_radius=self.hole_radii,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # Stage 1: Reach and grasp the peg
        gripper_pos = self.agent.tcp.pose.p
        peg_grasp_pos = self.peg.pose.p + torch.tensor(
            [
                0,
                0,
                0.02,
            ],
            device=self.device,
        )  # Grasp slightly above center

        reach_dist = torch.linalg.norm(gripper_pos - peg_grasp_pos, dim=1)
        reach_reward = 1 - torch.tanh(3.0 * reach_dist)

        is_grasped = self.agent.is_grasping(self.peg, max_angle=30)
        reward = reach_reward + is_grasped

        # Stage 2: Lift the peg
        peg_height = self.peg.pose.p[:, 2]
        lifted = peg_height > 0.12
        lift_reward = 2 * lifted.float()
        reward += lift_reward * is_grasped

        # Stage 3: Align peg with hole
        peg_tip_at_hole = (self.hole_center_pose.inv() * self.peg_tip_pose).p
        xy_alignment = torch.linalg.norm(peg_tip_at_hole[:, :2], dim=1)

        alignment_reward = 3 * (1 - torch.tanh(10.0 * xy_alignment))
        reward += alignment_reward * is_grasped * lifted

        # Stage 4: Insert peg into hole
        insertion_depth = torch.clamp(-peg_tip_at_hole[:, 2], 0, 0.06)
        insertion_reward = 5 * (insertion_depth / 0.06)

        well_aligned = xy_alignment < self.hole_radii
        reward += insertion_reward * is_grasped * well_aligned

        # Success bonus
        reward[info["success"]] = 15

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs, action, info) / 15
