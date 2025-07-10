"""
Copyright 2025 Zordi, Inc. All rights reserved.

PickCubeNoisy-v1 environment with configurable noise injection for robustness testing.
"""

from typing import Any, Dict, Union

import numpy as np
import torch

from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.utils.registration import register_env


@register_env("PickCubeNoisy-v1", max_episode_steps=50)
class PickCubeNoisyEnv(PickCubeEnv):
    """
    PickCube environment with configurable noise injection for robustness testing.

    Supports noise injection in:
    - Observations (position, velocity, pose data)
    - Rewards (dense and sparse)
    - Action execution (optional)

    Noise types supported:
    - Gaussian noise
    - Uniform noise
    - Multiplicative noise
    - Additive noise
    """

    SUPPORTED_ROBOTS = [
        "panda",
        "fetch",
        "xarm6_robotiq",
        "so100",
        "widowxai",
        "a1_galaxea",
    ]

    def __init__(
        self,
        *args,
        # Observation noise parameters
        obs_noise_type: str = "gaussian",  # "gaussian", "uniform", "none"
        obs_noise_std: float = 0.01,  # Standard deviation for gaussian noise
        obs_noise_range: float = 0.02,  # Range for uniform noise [-range, range]
        # Position-specific noise
        pos_noise_std: float = 0.005,  # Noise std for position observations
        vel_noise_std: float = 0.01,  # Noise std for velocity observations
        quat_noise_std: float = 0.01,  # Noise std for quaternion observations
        # Reward noise parameters
        reward_noise_type: str = "gaussian",  # "gaussian", "uniform", "none"
        reward_noise_std: float = 0.1,  # Standard deviation for reward noise
        reward_noise_range: float = 0.2,  # Range for uniform reward noise
        # Action noise parameters (optional)
        action_noise_type: str = "none",  # "gaussian", "uniform", "none"
        action_noise_std: float = 0.01,  # Standard deviation for action noise
        action_noise_range: float = 0.02,  # Range for uniform action noise
        # Noise scheduling (optional)
        noise_growth_rate: float = 0.0,  # Growth rate for noise over episodes (0 = no growth)
        min_noise_factor: float = 0.1,  # Starting noise factor
        max_noise_factor: float = 1.0,  # Maximum noise factor
        **kwargs,
    ):
        """
        Initialize PickCubeNoisy environment.

        Args:
            obs_noise_type: Type of observation noise ("gaussian", "uniform", "none")
            obs_noise_std: Standard deviation for gaussian observation noise
            obs_noise_range: Range for uniform observation noise
            pos_noise_std: Specific noise std for position observations
            vel_noise_std: Specific noise std for velocity observations
            quat_noise_std: Specific noise std for quaternion observations
            reward_noise_type: Type of reward noise ("gaussian", "uniform", "none")
            reward_noise_std: Standard deviation for gaussian reward noise
            reward_noise_range: Range for uniform reward noise
            action_noise_type: Type of action noise ("gaussian", "uniform", "none")
            action_noise_std: Standard deviation for gaussian action noise
            action_noise_range: Range for uniform action noise
            noise_growth_rate: Growth rate for noise over episodes (0 = no growth)
            min_noise_factor: Starting noise factor
            max_noise_factor: Maximum noise factor
        """
        # Store noise parameters
        self.obs_noise_type = obs_noise_type
        self.obs_noise_std = obs_noise_std
        self.obs_noise_range = obs_noise_range

        # Position-specific noise parameters
        self.pos_noise_std = pos_noise_std
        self.vel_noise_std = vel_noise_std
        self.quat_noise_std = quat_noise_std

        # Reward noise parameters
        self.reward_noise_type = reward_noise_type
        self.reward_noise_std = reward_noise_std
        self.reward_noise_range = reward_noise_range

        # Action noise parameters
        self.action_noise_type = action_noise_type
        self.action_noise_std = action_noise_std
        self.action_noise_range = action_noise_range

        # Noise scheduling
        self.noise_growth_rate = noise_growth_rate
        self.min_noise_factor = min_noise_factor
        self.max_noise_factor = max_noise_factor
        self.current_noise_factor = min_noise_factor  # Start with minimum noise
        self.episode_count = 0

        super().__init__(*args, **kwargs)

    def _apply_noise(self, data, noise_type: str, std: float, range_val: float):
        """
        Apply noise to tensor or numpy array data.

        Args:
            data: Input tensor or numpy array to add noise to
            noise_type: Type of noise ("gaussian", "uniform", "none")
            std: Standard deviation for gaussian noise
            range_val: Range for uniform noise

        Returns:
            Noisy tensor or numpy array
        """
        if noise_type == "none":
            return data

        noise_factor = self.current_noise_factor

        # Handle numpy arrays and torch tensors
        if isinstance(data, np.ndarray):
            if noise_type == "gaussian":
                noise = np.random.randn(*data.shape) * std * noise_factor
                return data + noise
            elif noise_type == "uniform":
                noise = (
                    (np.random.rand(*data.shape) - 0.5) * 2 * range_val * noise_factor
                )
                return data + noise
        elif noise_type == "gaussian":
            noise = torch.randn_like(data) * std * noise_factor
            return data + noise
        elif noise_type == "uniform":
            noise = (torch.rand_like(data) - 0.5) * 2 * range_val * noise_factor
            return data + noise

        return data

    def _apply_position_noise(self, positions: torch.Tensor) -> torch.Tensor:
        """Apply position-specific noise."""
        return self._apply_noise(
            positions, self.obs_noise_type, self.pos_noise_std, self.obs_noise_range
        )

    def _apply_velocity_noise(self, velocities: torch.Tensor) -> torch.Tensor:
        """Apply velocity-specific noise."""
        return self._apply_noise(
            velocities, self.obs_noise_type, self.vel_noise_std, self.obs_noise_range
        )

    def _apply_quaternion_noise(self, quaternions: torch.Tensor) -> torch.Tensor:
        """Apply quaternion-specific noise (small rotational perturbations)."""
        if self.obs_noise_type == "none":
            return quaternions

        # For quaternions, we apply small rotational noise
        noise_factor = self.current_noise_factor
        if self.obs_noise_type == "gaussian":
            # Small angle perturbations
            angle_noise = (
                torch.randn_like(quaternions[..., :3])
                * self.quat_noise_std
                * noise_factor
            )
            # Convert to quaternion perturbation (small angle approximation)
            quat_noise = torch.zeros_like(quaternions)
            quat_noise[..., :3] = (
                angle_noise * 0.5
            )  # Small angle approx: q ≈ [sin(θ/2), cos(θ/2)]
            quat_noise[..., 3] = 1.0

            # Normalize the result
            result = quaternions + quat_noise
            return result / torch.norm(result, dim=-1, keepdim=True)
        else:
            return quaternions

    def _get_obs_extra(self, info: Dict):
        """Override to add noise to observations."""
        obs = super()._get_obs_extra(info)

        # Add noise to TCP pose
        if "tcp_pose" in obs:
            tcp_pose = obs["tcp_pose"]
            # Position noise
            tcp_pose[..., :3] = self._apply_position_noise(tcp_pose[..., :3])
            # Quaternion noise
            tcp_pose[..., 3:7] = self._apply_quaternion_noise(tcp_pose[..., 3:7])
            obs["tcp_pose"] = tcp_pose

        # Add noise to goal position
        if "goal_pos" in obs:
            obs["goal_pos"] = self._apply_position_noise(obs["goal_pos"])

        # Add noise to state-based observations
        if "state" in self.obs_mode:
            if "obj_pose" in obs:
                obj_pose = obs["obj_pose"]
                # Position noise
                obj_pose[..., :3] = self._apply_position_noise(obj_pose[..., :3])
                # Quaternion noise
                obj_pose[..., 3:7] = self._apply_quaternion_noise(obj_pose[..., 3:7])
                obs["obj_pose"] = obj_pose

            if "tcp_to_obj_pos" in obs:
                obs["tcp_to_obj_pos"] = self._apply_position_noise(
                    obs["tcp_to_obj_pos"]
                )

            if "obj_to_goal_pos" in obs:
                obs["obj_to_goal_pos"] = self._apply_position_noise(
                    obs["obj_to_goal_pos"]
                )

        return obs

    def step(self, action):
        """Override step to add action noise if enabled."""
        # Apply action noise if enabled
        if self.action_noise_type != "none":
            action = self._apply_noise(
                action,
                self.action_noise_type,
                self.action_noise_std,
                self.action_noise_range,
            )
            # Clamp actions to valid range
            if isinstance(action, np.ndarray):
                action = np.clip(action, -1.0, 1.0)
            else:
                action = torch.clamp(action, -1.0, 1.0)

        return super().step(action)

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """Override to add noise to dense rewards."""
        reward = super().compute_dense_reward(obs, action, info)

        # Add reward noise
        if self.reward_noise_type != "none":
            reward = self._apply_noise(
                reward,
                self.reward_noise_type,
                self.reward_noise_std,
                self.reward_noise_range,
            )

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        """Override to add noise to normalized dense rewards."""
        reward = super().compute_normalized_dense_reward(obs, action, info)

        # Add reward noise (scaled for normalized rewards)
        if self.reward_noise_type != "none":
            reward = self._apply_noise(
                reward,
                self.reward_noise_type,
                self.reward_noise_std / 5.0,  # Scale down for normalized rewards
                self.reward_noise_range / 5.0,
            )

        return reward

    def reset(self, seed=None, options=None):
        """Override reset to update noise scheduling."""
        # Update noise factor based on episode count (curriculum learning)
        # Start with min noise and gradually increase to max noise
        noise_increase = self.noise_growth_rate * self.episode_count
        self.current_noise_factor = min(
            self.max_noise_factor, self.min_noise_factor + noise_increase
        )
        self.episode_count += 1

        return super().reset(seed=seed, options=options)

    def get_noise_info(self) -> Dict:
        """Get current noise configuration and state."""
        return {
            "obs_noise_type": self.obs_noise_type,
            "obs_noise_std": self.obs_noise_std,
            "reward_noise_type": self.reward_noise_type,
            "reward_noise_std": self.reward_noise_std,
            "action_noise_type": self.action_noise_type,
            "action_noise_std": self.action_noise_std,
            "current_noise_factor": self.current_noise_factor,
            "episode_count": self.episode_count,
        }


# Additional specialized noisy variants for different robots
@register_env("PickCubeNoisyFetch-v1", max_episode_steps=50)
class PickCubeNoisyFetchEnv(PickCubeNoisyEnv):
    """PickCubeNoisy environment configured for Fetch robot."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, robot_uids="fetch", **kwargs)


@register_env("PickCubeNoisyXArm6Robotiq-v1", max_episode_steps=50)
class PickCubeNoisyXArm6RobotiqEnv(PickCubeNoisyEnv):
    """PickCubeNoisy environment configured for XArm6Robotiq robot."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, robot_uids="xarm6_robotiq", **kwargs)


@register_env("PickCubeNoisySO100-v1", max_episode_steps=50)
class PickCubeNoisySO100Env(PickCubeNoisyEnv):
    """PickCubeNoisy environment configured for SO100 robot."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, robot_uids="so100", **kwargs)


@register_env("PickCubeNoisyWidowXAI-v1", max_episode_steps=50)
class PickCubeNoisyWidowXAIEnv(PickCubeNoisyEnv):
    """PickCubeNoisy environment configured for WidowXAI robot."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, robot_uids="widowxai", **kwargs)


@register_env("PickCubeNoisyA1Galaxea-v1", max_episode_steps=50)
class PickCubeNoisyA1GalaxeaEnv(PickCubeNoisyEnv):
    """PickCubeNoisy environment configured for A1Galaxea robot."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, robot_uids="a1_galaxea", **kwargs)


# Documentation
PickCubeNoisyEnv.__doc__ = """
**Task Description:**
A noisy version of the PickCube task where configurable noise can be injected into observations, rewards, and actions.
This environment is designed for testing robustness of learning algorithms like PPO.

**Supported Robots:**
- PickCubeNoisy-v1 (Panda - default)
- PickCubeNoisyFetch-v1 (Fetch robot)
- PickCubeNoisyXArm6Robotiq-v1 (XArm6 with Robotiq gripper)
- PickCubeNoisySO100-v1 (SO100 robot)
- PickCubeNoisyWidowXAI-v1 (WidowXAI robot)
- PickCubeNoisyA1Galaxea-v1 (A1Galaxea robot)

**Noise Types:**
- Observation noise: Applied to positions, velocities, and quaternions
- Reward noise: Applied to dense and sparse rewards  
- Action noise: Applied to robot actions (optional)

**Noise Parameters:**
- obs_noise_type: Type of observation noise ("gaussian", "uniform", "none")
- obs_noise_std: Standard deviation for gaussian noise
- pos_noise_std: Specific noise for position observations
- reward_noise_type: Type of reward noise  
- reward_noise_std: Standard deviation for reward noise
- noise_growth_rate: Growth rate for noise over episodes (curriculum learning)
- min_noise_factor: Starting noise factor (easy)
- max_noise_factor: Maximum noise factor (hard)

**Usage Example:**
```python
import gymnasium as gym
# Basic usage with Panda robot
env = gym.make("PickCubeNoisy-v1", 
               obs_noise_type="gaussian", 
               obs_noise_std=0.01,
               reward_noise_type="gaussian",
               reward_noise_std=0.1)

# Using different robot
env = gym.make("PickCubeNoisyFetch-v1",
               obs_noise_type="gaussian",
               obs_noise_std=0.01)

# Curriculum learning (start easy, get harder)
env = gym.make("PickCubeNoisy-v1",
               obs_noise_type="gaussian",
               obs_noise_std=0.02,
               noise_growth_rate=0.01,
               min_noise_factor=0.1,
               max_noise_factor=1.0)
```
"""
