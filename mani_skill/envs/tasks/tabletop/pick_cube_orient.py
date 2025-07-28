"""
Copyright 2025 Zordi, Inc. All rights reserved.

PickCube environment with orientation reward for keeping gripper downward-facing.
"""

from typing import Any, Dict, Union

import torch

from mani_skill.agents.robots import (
    SO100,
    A1Galaxea,
    Fetch,
    Panda,
    WidowXAI,
    XArm6Robotiq,
)
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.utils.registration import register_env

PICK_CUBE_ORIENT_DOC_STRING = """**Task Description:**
A modified version of PickCube-v1 that adds an orientation reward to encourage the robot gripper
to maintain a downward-facing orientation (perpendicular to the table surface) throughout the task.
This helps prevent the robot from twisting its arm in random directions when moving the cube to
the goal position.

**Additional Reward Component:**
- **Orientation reward**: Encourages the gripper's palm direction (local -z axis) to align with
  the world -z direction (downward). The reward is computed as:

  orient_reward = orient_coef * (1.0 - tanh(orient_lambda * (1.0 - alignment) / 2.0))

  where alignment is the cosine similarity between palm direction and world down direction.

**Randomizations:**
- Same as PickCube-v1: cube xy position, z-axis rotation, and goal position randomization

**Success Conditions:**
- Same as PickCube-v1: cube within goal threshold and robot static
"""


@register_env("PickCubeOrient-v1", max_episode_steps=50)
class PickCubeOrientEnv(PickCubeEnv):
    """
    PickCube with extra reward encouraging the gripper z-axis to align with –world-z
    (i.e. palm facing the table).
    """

    agent: Union[Panda, Fetch, XArm6Robotiq, SO100, WidowXAI, A1Galaxea]

    # Orientation reward parameters - tune these for desired behavior
    orient_coef = 0.2  # Weight of orientation reward (contributes up to +0.5)
    orient_lambda = 5.0  # Controls tanh steepness (higher = more strict)

    def __init__(self, *args, **kwargs):
        """Initialize PickCubeOrient environment."""
        super().__init__(*args, **kwargs)

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """
        Compute dense reward including original PickCube rewards plus orientation reward.

        Args:
            obs: Environment observations
            action: Action taken by the agent
            info: Information dictionary containing evaluation metrics

        Returns:
            torch.Tensor: Total reward including orientation component
        """
        # Get the original PickCube reward components
        reward = super().compute_dense_reward(obs, action, info)

        # ------------------------------------------------------------------
        # Add orientation reward to encourage downward-facing gripper
        # ------------------------------------------------------------------
        # Get TCP rotation matrix: shape (B, 3, 3)
        tcp_transform = self.agent.tcp_pose.to_transformation_matrix()  # (B, 4, 4)
        R = tcp_transform[..., :3, :3]  # Extract rotation matrix (B, 3, 3)

        # Gripper's local -z axis (palm direction) in world coordinates
        # The third column of the rotation matrix is the local z-axis,
        # so -R[..., :, 2] gives us the palm direction
        palm_dir = -R[..., :, 2]  # (B, 3)

        # World down direction
        world_down = torch.tensor([0.0, 0.0, -1.0], device=palm_dir.device)

        # Compute cosine similarity (dot product since both are unit vectors)
        # This gives us a value in [-1, 1] where 1 = perfect alignment
        alignment = (palm_dir * world_down).sum(-1)  # (B,)

        # Convert alignment to reward using tanh to provide smooth gradients
        # When alignment = 1 (perfect): (1.0 - alignment) / 2.0 = 0, tanh(0) = 0, reward = 1.0
        # When alignment = -1 (opposite): (1.0 - alignment) / 2.0 = 1, tanh(lambda) ≈ 1, reward ≈ 0
        orient_reward = 1.0 - torch.tanh(self.orient_lambda * (1.0 - alignment) / 2.0)  # (B,)

        # Add weighted orientation reward to total reward
        reward += self.orient_coef * orient_reward

        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """
        Compute normalized dense reward.

        The original PickCube reward has a max of 5, and we add up to orient_coef,
        so the new max is approximately (5 + orient_coef).
        """
        max_reward = 5.0 + self.orient_coef
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward


# Update docstring with robot-specific information
PickCubeOrientEnv.__doc__ = PICK_CUBE_ORIENT_DOC_STRING.format(robot_id="Panda")


@register_env("PickCubeOrientSO100-v1", max_episode_steps=50)
class PickCubeOrientSO100Env(PickCubeOrientEnv):
    """PickCubeOrient variant for SO100 robot."""

    _sample_video_link = (
        "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PickCubeSO100-v1_rt.mp4"
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, robot_uids="so100", **kwargs)


PickCubeOrientSO100Env.__doc__ = PICK_CUBE_ORIENT_DOC_STRING.format(robot_id="SO100")


@register_env("PickCubeOrientWidowXAI-v1", max_episode_steps=50)
class PickCubeOrientWidowXAIEnv(PickCubeOrientEnv):
    """PickCubeOrient variant for WidowXAI robot."""

    _sample_video_link = (
        "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PickCubeWidowXAI-v1_rt.mp4"
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, robot_uids="widowxai", **kwargs)


PickCubeOrientWidowXAIEnv.__doc__ = PICK_CUBE_ORIENT_DOC_STRING.format(robot_id="WidowXAI")
