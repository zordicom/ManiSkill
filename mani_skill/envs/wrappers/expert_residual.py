"""
Copyright 2025 Zordi, Inc. All rights reserved.

Expert+Residual action decomposition wrapper for ManiSkill environments.
Enables hybrid control where expert policy provides base actions and learned policy provides residual corrections.
"""

import logging
import math
import os
from collections.abc import Callable
from typing import Any, Dict, Union

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from torch import nn
from torch.distributions.normal import Normal

from mani_skill.utils import common


# PPO Model loading support
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initialize layer weights for PPO models."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PPOAgent(nn.Module):
    """PPO Agent for state-based observations (from ppo.py)."""

    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(
                nn.Linear(256, np.prod(envs.single_action_space.shape)),
                std=0.01 * np.sqrt(2),
            ),
        )
        self.actor_logstd = nn.Parameter(
            torch.ones(1, np.prod(envs.single_action_space.shape)) * -0.5
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action(self, x, deterministic=False):
        action_mean = self.actor_mean(x)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )


class NatureCNN(nn.Module):
    """Optimized CNN for processing RGB observations (from ppo_rgb_fast.py)."""

    def __init__(self, sample_obs, device=None):
        super().__init__()

        extractors = {}
        self.out_features = 0
        feature_size = 256

        # Handle RGB observations
        if "rgb" in sample_obs:
            in_channels = sample_obs["rgb"].shape[-1]
            image_size = (sample_obs["rgb"].shape[1], sample_obs["rgb"].shape[2])

            extractors["rgb"] = nn.Sequential(
                layer_init(
                    nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, device=device)
                ),
                nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2, device=device)),
                nn.ReLU(),
                layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1, device=device)),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(
                    nn.Linear(
                        self._get_conv_output_size(image_size, in_channels),
                        feature_size,
                        device=device,
                    )
                ),
                nn.ReLU(),
            )
            self.out_features += feature_size

        # Handle additional RGB cameras (e.g., wrist camera)
        for key in sample_obs.keys():
            if key.endswith("_rgb") or (key.startswith("rgb") and key != "rgb"):
                in_channels = sample_obs[key].shape[-1]
                image_size = (sample_obs[key].shape[1], sample_obs[key].shape[2])

                extractors[key] = nn.Sequential(
                    layer_init(
                        nn.Conv2d(
                            in_channels, 32, kernel_size=8, stride=4, device=device
                        )
                    ),
                    nn.ReLU(),
                    layer_init(
                        nn.Conv2d(32, 64, kernel_size=4, stride=2, device=device)
                    ),
                    nn.ReLU(),
                    layer_init(
                        nn.Conv2d(64, 64, kernel_size=3, stride=1, device=device)
                    ),
                    nn.ReLU(),
                    nn.Flatten(),
                    layer_init(
                        nn.Linear(
                            self._get_conv_output_size(image_size, in_channels),
                            feature_size,
                            device=device,
                        )
                    ),
                    nn.ReLU(),
                )
                self.out_features += feature_size

        # Handle state observations
        if "state" in sample_obs:
            state_size = sample_obs["state"].shape[-1]
            extractors["state"] = nn.Sequential(
                layer_init(nn.Linear(state_size, 256, device=device)),
                nn.ReLU(),
                layer_init(nn.Linear(256, 256, device=device)),
                nn.ReLU(),
            )
            self.out_features += 256

        self.extractors = nn.ModuleDict(extractors)

    def _get_conv_output_size(self, image_size, in_channels):
        """Calculate the output size of convolutional layers."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, *image_size)
            dummy_output = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.Flatten(),
            )(dummy_input)
            return dummy_output.shape[1]

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            if key in observations:
                # Handle RGB observations (need channel permutation)
                if "rgb" in key:
                    # Convert from (B, H, W, C) to (B, C, H, W)
                    obs = observations[key].permute(0, 3, 1, 2) / 255.0
                else:
                    obs = observations[key]

                encoded_tensor_list.append(extractor(obs))

        return torch.cat(encoded_tensor_list, dim=1)


class PPORGBAgent(nn.Module):
    """PPO Agent with RGB observations (from ppo_rgb_fast.py)."""

    def __init__(self, n_act, sample_obs, device=None):
        super().__init__()

        # CNN feature extractor
        self.cnn = NatureCNN(sample_obs, device=device)

        # Policy and value networks
        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.cnn.out_features, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1, device=device)),
        )

        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(self.cnn.out_features, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, n_act, device=device), std=0.01 * np.sqrt(2)),
        )

        self.actor_logstd = nn.Parameter(torch.zeros(1, n_act, device=device))

    def get_features(self, x):
        return self.cnn(x)

    def get_value(self, x):
        features = self.get_features(x)
        return self.critic(features)

    def get_action(self, x, deterministic=False):
        features = self.get_features(x)
        action_mean = self.actor_mean(features)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()

    def get_action_and_value(self, obs, action=None):
        features = self.get_features(obs)
        action_mean = self.actor_mean(features)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            action = action_mean + action_std * torch.randn_like(action_mean)

        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(features),
        )


def load_ppo_expert(
    model_path: str,
    env_id: str,
    obs_mode: str = "state",
    deterministic: bool = True,
    device: str = "cuda",
    **env_kwargs,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Load a trained PPO model as an expert policy.

    Args:
        model_path: Path to the saved model checkpoint
        env_id: Environment ID used for training
        obs_mode: Observation mode ("state" or "rgb")
        deterministic: Whether to use deterministic actions
        device: Device to load the model on
        **env_kwargs: Additional environment arguments

    Returns:
        Expert policy function that takes observations and returns actions
    """
    # Create a dummy environment to get observation/action space info
    dummy_env = gym.make(env_id, num_envs=1, obs_mode=obs_mode, **env_kwargs)

    # Apply the same wrappers that would be used during training
    if obs_mode == "rgb":
        from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

        dummy_env = FlattenRGBDObservationWrapper(dummy_env, rgb=True, state=True)

    if isinstance(dummy_env.action_space, gym.spaces.Dict):
        from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper

        dummy_env = FlattenActionSpaceWrapper(dummy_env)

    # Create the appropriate agent
    if obs_mode == "state":
        agent = PPOAgent(dummy_env)
    else:  # RGB mode
        n_act = math.prod(dummy_env.single_action_space.shape)
        sample_obs = dummy_env.reset()[0]
        agent = PPORGBAgent(n_act, sample_obs, device=device)

    # Load the trained weights
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.to(device)
    agent.eval()

    dummy_env.close()

    def expert_policy(obs: torch.Tensor) -> torch.Tensor:
        """Expert policy function that takes observations and returns actions."""
        with torch.no_grad():
            # Ensure obs is a tensor and on the correct device
            if isinstance(obs, dict):
                # Convert numpy arrays to tensors if needed
                obs_processed = {}
                for k, v in obs.items():
                    if isinstance(v, np.ndarray):
                        obs_processed[k] = torch.from_numpy(v).to(device).float()
                    else:
                        obs_processed[k] = v.to(device).float()
                obs = obs_processed
            elif isinstance(obs, np.ndarray):
                obs = torch.from_numpy(obs).to(device).float()
            else:
                obs = obs.to(device).float()

            # Handle batch dimension
            if obs_mode == "state":
                # For state observations, flatten if needed
                if isinstance(obs, dict):
                    # Flatten dictionary observations
                    obs_parts = []
                    for k in sorted(obs.keys()):  # Sort for consistent ordering
                        if k == "rgb":
                            # Flatten RGB: (B, H, W, C) -> (B, H*W*C)
                            obs_parts.append(obs[k].flatten(start_dim=1))
                        else:
                            # Flatten other observations
                            obs_parts.append(obs[k].flatten(start_dim=1))
                    obs_flat = torch.cat(obs_parts, dim=-1)
                else:
                    obs_flat = obs.flatten(start_dim=1) if obs.dim() > 1 else obs
                action = agent.get_action(obs_flat, deterministic=deterministic)
            else:
                # For RGB observations, pass dict directly
                if not isinstance(obs, dict):
                    raise ValueError(
                        f"RGB mode expects dict observations, got {type(obs)}"
                    )
                action = agent.get_action(obs, deterministic=deterministic)

            return action.float()

    return expert_policy


def create_ppo_expert_policy(
    expert_type: str,
    model_path: str,
    env_id: str,
    device: str = "cuda",
    deterministic: bool = True,
    **kwargs,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Create a PPO expert policy based on the expert type.

    Args:
        expert_type: Type of expert ("ppo" or "ppo_rgb")
        model_path: Path to the saved model checkpoint
        env_id: Environment ID
        device: Device to load the model on
        deterministic: Whether to use deterministic actions
        **kwargs: Additional arguments

    Returns:
        Expert policy function
    """
    if expert_type == "ppo":
        return load_ppo_expert(
            model_path=model_path,
            env_id=env_id,
            obs_mode="state",
            device=device,
            deterministic=deterministic,
            **kwargs,
        )
    elif expert_type == "ppo_rgb":
        return load_ppo_expert(
            model_path=model_path,
            env_id=env_id,
            obs_mode="rgb",
            device=device,
            deterministic=deterministic,
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported expert type: {expert_type}")


class ExpertResidualWrapper(gym.Wrapper):
    """
    ManiSkill wrapper for expert+residual action decomposition.

    This wrapper transforms any ManiSkill environment to support expert+residual control:
    1. Expert policy provides base actions (e.g., IK solver, pre-trained model)
    2. Learned policy provides residual corrections
    3. Final action = expert_action + residual_action

    The wrapper:
    - Creates environments from env_id and lets ManiSkill handle vectorization efficiently
    - Automatically handles any number of parallel environments efficiently
    - Extends observation space to include expert actions: [base_obs, expert_action]
    - Receives residual actions from the agent
    - Combines expert + residual actions before sending to environment
    - Uses torch tensors throughout for ManiSkill3 compatibility
    - Supports loading trained PPO models as expert policies

    Args:
        env_id: ManiSkill environment ID (e.g., "PickBox-v1")
        expert_policy_fn: Callable that takes observation tensor and returns expert action tensor.
            The expert policy MUST return torch.Tensor on the same device as the input observation.
            If None and expert_type is "ppo" or "ppo_rgb", will load from model_path.
        num_envs: Number of parallel environments (default: 1)
        residual_scale: Scale factor for residual actions (default: 1.0)
        clip_final_action: Whether to clip final action to action space bounds (default: True)
        expert_action_noise: Gaussian noise std to add to expert actions (default: 0.0)
        log_actions: Whether to log action decomposition for debugging (default: False)
        track_action_stats: Whether to track expert/residual action statistics (default: False)
        device: Device to place environments on (default: "cuda")
        expert_type: Type of expert policy ("none", "ppo", "ppo_rgb") (default: "none")
        model_path: Path to saved PPO model checkpoint (required for "ppo" and "ppo_rgb" types)
        deterministic_expert: Whether to use deterministic actions from expert policy (default: True)
        **env_kwargs: Additional arguments for environment creation

    Usage Examples:
        # Single environment with custom expert policy
        >>> wrapper = ExpertResidualWrapper(
        ...     "PickBox-v1", expert_policy_fn=custom_expert
        ... )

        # Load PPO state expert from checkpoint
        >>> wrapper = ExpertResidualWrapper(
        ...     "PickCube-v1",
        ...     expert_type="ppo",
        ...     model_path="runs/model/final_ckpt.pt",
        ...     num_envs=100,
        ... )

        # Load PPO RGB expert from checkpoint
        >>> wrapper = ExpertResidualWrapper(
        ...     "PickCube-v1",
        ...     expert_type="ppo_rgb",
        ...     model_path="runs/rgb_model/final_ckpt.pt",
        ...     num_envs=100,
        ... )

        # 100 parallel environments - ManiSkill handles vectorization
        >>> wrapper = ExpertResidualWrapper("PickBox-v1", expert_policy, num_envs=100)

        # 2048 parallel environments - Still efficient! ManiSkill handles it
        >>> wrapper = ExpertResidualWrapper("PickBox-v1", expert_policy, num_envs=2048)

        # All cases use the same interface
        >>> obs, info = wrapper.reset()
        >>> residual_actions = torch.randn(
        ...     wrapper.num_envs, wrapper.action_space.shape[-1]
        ... )
        >>> next_obs, reward, terminated, truncated, info = wrapper.step(
        ...     residual_actions
        ... )
    """

    def __init__(
        self,
        env_id: str,
        expert_policy_fn: Callable[[torch.Tensor], torch.Tensor] = None,
        num_envs: int = 1,
        residual_scale: float = 1.0,
        clip_final_action: bool = True,
        expert_action_noise: float = 0.0,
        log_actions: bool = False,
        track_action_stats: bool = False,
        device: str = "cuda",
        control_mode: str = "pd_joint_delta_pos",
        # PPO expert parameters
        expert_type: str = "none",
        model_path: str = None,
        deterministic_expert: bool = True,
        **env_kwargs,
    ):
        """
        Initialize expert+residual wrapper.

        Args:
            env_id: ManiSkill environment ID (e.g., "PickBox-v1")
            expert_policy_fn: Function that takes observation tensor and returns expert action tensor.
                MUST return torch.Tensor. Input and output should be on the same device.
                If None and expert_type is "ppo" or "ppo_rgb", will load from model_path.
            num_envs: Number of parallel environments
            residual_scale: Scale factor for residual actions
            clip_final_action: Whether to clip final action to action space bounds
            expert_action_noise: Gaussian noise std to add to expert actions
            log_actions: Whether to log action decomposition for debugging
            track_action_stats: Whether to track expert/residual action statistics for monitoring
            device: Device to place environments on
            control_mode: Control mode for the environment.
                IMPORTANT: Use "pd_ee_delta_pos" for IK expert policies.
            expert_type: Type of expert policy ("none", "ppo", "ppo_rgb")
            model_path: Path to saved PPO model checkpoint (required for "ppo" and "ppo_rgb" types)
            deterministic_expert: Whether to use deterministic actions from expert policy
            **env_kwargs: Additional environment creation arguments
        """
        # Initialize logger first
        self.logger = logging.getLogger(__name__)

        # Set control mode in env_kwargs
        env_kwargs["control_mode"] = control_mode

        # Handle expert policy creation
        if expert_policy_fn is None:
            if expert_type in ["ppo", "ppo_rgb"]:
                if model_path is None:
                    raise ValueError(
                        f"model_path is required for expert_type='{expert_type}'"
                    )

                # Create expert policy from trained PPO model
                expert_policy_fn = create_ppo_expert_policy(
                    expert_type=expert_type,
                    model_path=model_path,
                    env_id=env_id,
                    device=device,
                    deterministic=deterministic_expert,
                    **env_kwargs,
                )
                self.logger.info(
                    f"Created {expert_type} expert policy from {model_path}"
                )
            else:
                raise ValueError(
                    f"expert_policy_fn is required for expert_type='{expert_type}'"
                )

        # Let ManiSkill handle vectorization efficiently
        env = gym.make(env_id, num_envs=num_envs, **env_kwargs)

        # Apply FlattenRGBDObservationWrapper only if environment has RGB observations
        if self._has_rgb_observations(env):
            from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

            # Apply flattening wrapper to get consistent observation format
            env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
            self._expects_flattened_obs = True
        else:
            self._expects_flattened_obs = False

        super().__init__(env)

        # Use environment's device if available, otherwise use specified device
        if hasattr(env.unwrapped, "device"):
            self.device = env.unwrapped.device
        else:
            self.device = device

        self.env_id = env_id
        self.expert_policy_fn = expert_policy_fn
        self.num_envs = num_envs
        self.residual_scale = residual_scale
        self.clip_final_action = clip_final_action
        self.expert_action_noise = expert_action_noise
        self.log_actions = log_actions
        self.track_action_stats = track_action_stats
        self.control_mode = control_mode
        self.expert_type = expert_type
        self.model_path = model_path
        self.deterministic_expert = deterministic_expert

        # Set up wrapper
        self._setup_wrapper()

    def _has_rgb_observations(self, env) -> bool:
        """Check if environment has RGB observations that need flattening."""
        if not hasattr(env, "observation_space"):
            return False

        obs_space = env.observation_space

        # Check if it's a Dict observation space (potential RGB environment)
        if not isinstance(obs_space, spaces.Dict):
            return False

        # Check if it has sensor_data with RGB cameras
        if "sensor_data" in obs_space.spaces:
            sensor_data_space = obs_space.spaces["sensor_data"]
            if isinstance(sensor_data_space, spaces.Dict):
                # Look for RGB cameras in sensor_data
                for camera_name, camera_space in sensor_data_space.spaces.items():
                    if (
                        isinstance(camera_space, spaces.Dict)
                        and "rgb" in camera_space.spaces
                    ):
                        return True
        return False

    def _calculate_rgb_channels(self) -> int:
        """Calculate the number of RGB channels that will be provided by FlattenRGBDObservationWrapper."""
        total_channels = 0
        if hasattr(self, "env") and hasattr(self.env, "observation_space"):  # noqa: PLR1702
            obs_space = self.env.observation_space
            if isinstance(obs_space, spaces.Dict) and "sensor_data" in obs_space.spaces:
                sensor_data_space = obs_space.spaces["sensor_data"]
                if isinstance(sensor_data_space, spaces.Dict):
                    for camera_name, camera_space in sensor_data_space.spaces.items():
                        if (
                            isinstance(camera_space, spaces.Dict)
                            and "rgb" in camera_space.spaces
                        ):
                            rgb_space = camera_space.spaces["rgb"]
                            if len(rgb_space.shape) >= 3:
                                _, _, c = rgb_space.shape[-3:]
                                total_channels += c

            # Also check if env has 'rgb' key directly (from FlattenRGBDObservationWrapper)
            if isinstance(obs_space, spaces.Dict) and "rgb" in obs_space.spaces:
                rgb_space = obs_space.spaces["rgb"]
                if len(rgb_space.shape) >= 3:
                    _, _, c = rgb_space.shape[-3:]
                    total_channels = c  # Use the flattened RGB channels

        result = total_channels if total_channels > 0 else 3
        return result

    def _setup_wrapper(self):
        """Set up wrapper for any number of environments."""
        # Store original spaces
        self.base_observation_space = self.env.observation_space
        self.base_action_space = self.env.action_space

        # Validate action space
        if not isinstance(self.base_action_space, spaces.Box):
            raise ValueError(
                f"Expert+Residual wrapper requires Box action space, got {type(self.base_action_space)}"
            )

        # Cache action space bounds as tensors
        if len(self.base_action_space.shape) == 1:
            # Single environment
            self._action_low = torch.tensor(
                self.base_action_space.low, device=self.device, dtype=torch.float32
            )
            self._action_high = torch.tensor(
                self.base_action_space.high, device=self.device, dtype=torch.float32
            )
        else:
            # Vectorized environment - use bounds for first environment
            self._action_low = torch.tensor(
                self.base_action_space.low[0], device=self.device, dtype=torch.float32
            )
            self._action_high = torch.tensor(
                self.base_action_space.high[0], device=self.device, dtype=torch.float32
            )

        # Action space remains the same (residual actions only)
        self.action_space = self.base_action_space

        # Always extend observation space to include expert actions
        if isinstance(self.base_observation_space, spaces.Box):
            # Handle Box observation space (state observations)
            self._setup_box_observation_space()
        elif isinstance(self.base_observation_space, spaces.Dict):
            # Handle Dict observation space (RGB/visual observations)
            self._setup_dict_observation_space()
        else:
            raise ValueError(
                f"Expert+Residual wrapper requires Box or Dict observation space, got {type(self.base_observation_space)}"
            )

        # Initialize statistics
        self._setup_statistics()

        # Store last base observation for expert policy
        self.last_base_obs = None

    def _setup_box_observation_space(self):
        """Setup wrapper for Box observation spaces (state observations)."""
        # Handle vectorized vs single environment observation spaces
        import numpy as np

        # For ManiSkill, even single environments have batched observation spaces
        # obs_space: (1, obs_dim) for single env, (num_envs, obs_dim) for vectorized
        # action_space: (action_dim,) for single env, (num_envs, action_dim) for vectorized

        if len(self.base_observation_space.shape) == 1:
            # Rare case: truly single dimension observation
            base_obs_dim = self.base_observation_space.shape[0]
            obs_shape = self.base_observation_space.shape
            is_vectorized = False
        elif len(self.base_observation_space.shape) == 2:
            # Common case: batched observations
            base_obs_dim = self.base_observation_space.shape[1]  # Use last dimension
            obs_shape = (self.base_observation_space.shape[1],)  # Per-env obs shape
            is_vectorized = self.base_observation_space.shape[0] > 1
        else:
            raise ValueError(
                f"Unsupported observation space shape: {self.base_observation_space.shape}"
            )

        # Handle action dimension - check if it's batched or not
        if len(self.base_action_space.shape) == 1:
            action_dim = self.base_action_space.shape[0]
        else:
            action_dim = self.base_action_space.shape[1]  # Use last dimension

        # New observation space: [base_obs, expert_action] per environment
        extended_obs_dim = base_obs_dim + action_dim

        # Create bounds for single environment observation space
        if len(self.base_observation_space.shape) == 2:
            # Batched observation bounds - use first environment
            base_obs_low = self.base_observation_space.low[0]
            base_obs_high = self.base_observation_space.high[0]
        else:
            # Unbatched observation bounds
            base_obs_low = self.base_observation_space.low
            base_obs_high = self.base_observation_space.high

        if len(self.base_action_space.shape) == 2:
            # Batched action bounds - use first environment
            action_low = self.base_action_space.low[0]
            action_high = self.base_action_space.high[0]
        else:
            # Unbatched action bounds
            action_low = self.base_action_space.low
            action_high = self.base_action_space.high

        # Ensure all arrays have same number of dimensions for concatenation
        base_obs_low = np.atleast_1d(base_obs_low)
        base_obs_high = np.atleast_1d(base_obs_high)
        action_low = np.atleast_1d(action_low)
        action_high = np.atleast_1d(action_high)

        # Create extended bounds
        extended_low = np.concatenate([base_obs_low, action_low])
        extended_high = np.concatenate([base_obs_high, action_high])

        # Create extended observation space (per-environment)
        self.observation_space = spaces.Box(
            low=extended_low,
            high=extended_high,
            shape=(extended_obs_dim,),
            dtype=np.float32,  # Always use float32 for consistent dtype
        )

    def _setup_dict_observation_space(self):
        """Setup wrapper for Dict observation spaces (RGB/visual observations)."""
        # Create flattened observation space compatible with FlattenRGBDObservationWrapper
        # Keep RGB in image format, flatten only state+expert_action

        # Get action dimension
        if len(self.base_action_space.shape) == 1:
            action_dim = self.base_action_space.shape[0]
        else:
            # Vectorized - use per-environment dimensions
            action_dim = self.base_action_space.shape[1]

        new_spaces = {}

        # Keep RGB space in image format (concatenate cameras along channel dimension)
        total_channels = 0
        image_height = None
        image_width = None

        # Check if RGB is already in the base observation space (from FlattenRGBDObservationWrapper)
        if "rgb" in self.base_observation_space.spaces:  # noqa: PLR1702
            rgb_space = self.base_observation_space.spaces["rgb"]
            if len(rgb_space.shape) >= 3:
                # RGB space should be (H, W, C) or (batch, H, W, C)
                if len(rgb_space.shape) == 4:
                    # Batched: (batch, H, W, C)
                    _, h, w, c = rgb_space.shape
                else:
                    # Single: (H, W, C)
                    h, w, c = rgb_space.shape
                total_channels = c
                image_height = h
                image_width = w

        # If not found in base observation space, check sensor_data (original environment)
        elif "sensor_data" in self.base_observation_space.spaces:
            sensor_data_space = self.base_observation_space.spaces["sensor_data"]

            if isinstance(sensor_data_space, spaces.Dict):
                for camera_name, camera_space in sensor_data_space.spaces.items():
                    if (
                        isinstance(camera_space, spaces.Dict)
                        and "rgb" in camera_space.spaces
                    ):
                        rgb_space = camera_space.spaces["rgb"]
                        # RGB space should be (H, W, C)
                        if len(rgb_space.shape) >= 3:
                            h, w, c = rgb_space.shape[-3:]
                            total_channels += c
                            if image_height is None:
                                image_height = h
                                image_width = w

        # For expert residual wrapper, always include RGB space to maintain consistency
        # with PPO RGB mode expectations (even if creating dummy RGB observations)

        if total_channels > 0 and image_height is not None:
            new_spaces["rgb"] = spaces.Box(
                low=0,
                high=255,
                shape=(image_height, image_width, total_channels),
                dtype=np.uint8,
            )
        else:
            # Create a default RGB space for dummy observations
            # Check if the environment has RGB observations that we need to account for
            rgb_channels = 3  # Default to 3 channels
            if hasattr(self, "env") and self._has_rgb_observations(self.env):
                # If environment has RGB observations, calculate the actual number of channels
                # that will be provided by FlattenRGBDObservationWrapper
                rgb_channels = self._calculate_rgb_channels()

            new_spaces["rgb"] = spaces.Box(
                low=0,
                high=255,
                shape=(224, 224, rgb_channels),
                dtype=np.uint8,
            )

        # Calculate flattened state space (including expert_action)
        # If the environment is already wrapped with FlattenRGBDObservationWrapper,
        # use the existing state dimensions + expert_action dimensions
        if "state" in self.base_observation_space.spaces:  # noqa: PLR1702
            base_state_space = self.base_observation_space.spaces["state"]

            # Get the per-environment state dimensions
            if len(base_state_space.shape) > 1:
                # Remove batch dimension: (num_envs, ...) -> (...)
                per_env_state_dims = int(np.prod(base_state_space.shape[1:]))
            else:
                # Single environment: (state_dim,)
                per_env_state_dims = int(np.prod(base_state_space.shape))

            # Add expert action dimensions
            total_state_dim = per_env_state_dims + action_dim

        else:
            # If no state space exists, calculate from agent/extra components
            total_state_dim = action_dim  # Start with expert_action dimension

            # Add agent components (qpos, qvel, etc.)
            if "agent" in self.base_observation_space.spaces:
                agent_space = self.base_observation_space.spaces["agent"]
                if isinstance(agent_space, spaces.Dict):
                    for key, space in agent_space.spaces.items():
                        if isinstance(space, spaces.Box):
                            if len(space.shape) > 1:
                                # Remove batch dimension: (num_envs, ...) -> (...)
                                per_env_shape = space.shape[1:]
                                total_state_dim += int(np.prod(per_env_shape))
                            else:
                                # Single dimension per environment
                                total_state_dim += 1

            if "extra" in self.base_observation_space.spaces:
                extra_space = self.base_observation_space.spaces["extra"]
                if isinstance(extra_space, spaces.Box):
                    # Add flattened state dimensions (per-environment)
                    if len(extra_space.shape) > 1:
                        # Remove batch dimension: (num_envs, ...) -> (...)
                        per_env_shape = extra_space.shape[1:]
                        total_state_dim += int(np.prod(per_env_shape))
                    else:
                        # Single dimension per environment
                        total_state_dim += 1
                elif isinstance(extra_space, spaces.Dict):
                    # Add all state components (per-environment)
                    for key, space in extra_space.spaces.items():
                        if isinstance(space, spaces.Box):
                            if len(space.shape) > 1:
                                # Remove batch dimension: (num_envs, ...) -> (...)
                                per_env_shape = space.shape[1:]
                                total_state_dim += int(np.prod(per_env_shape))
                            else:
                                # Single dimension per environment
                                total_state_dim += 1

        if total_state_dim > 0:
            new_spaces["state"] = spaces.Box(
                low=-np.inf, high=np.inf, shape=(total_state_dim,), dtype=np.float32
            )

        self.observation_space = spaces.Dict(new_spaces)

    @property
    def single_observation_space(self):
        """Return the observation space for a single environment."""
        return self.observation_space

    def _setup_statistics(self):
        """Initialize action statistics tracking."""
        # Always track basic counters
        self.episode_count = 0
        self.step_count = 0

        # Only setup detailed action statistics if requested
        if self.track_action_stats:
            # Use per-environment action dimensions for statistics
            if len(self.base_action_space.shape) == 1:
                action_dim = self.base_action_space.shape[0]
            else:
                action_dim = self.base_action_space.shape[
                    1
                ]  # Per-environment action dim

            self.expert_action_stats = {
                "mean": torch.zeros(
                    action_dim, device=self.device, dtype=torch.float32
                ),
                "std": torch.zeros(action_dim, device=self.device, dtype=torch.float32),
                "min": torch.full(
                    (action_dim,), float("inf"), device=self.device, dtype=torch.float32
                ),
                "max": torch.full(
                    (action_dim,),
                    float("-inf"),
                    device=self.device,
                    dtype=torch.float32,
                ),
            }
            self.residual_action_stats = {
                "mean": torch.zeros(
                    action_dim, device=self.device, dtype=torch.float32
                ),
                "std": torch.zeros(action_dim, device=self.device, dtype=torch.float32),
                "min": torch.full(
                    (action_dim,), float("inf"), device=self.device, dtype=torch.float32
                ),
                "max": torch.full(
                    (action_dim,),
                    float("-inf"),
                    device=self.device,
                    dtype=torch.float32,
                ),
            }
        else:
            self.expert_action_stats = None
            self.residual_action_stats = None

    def reset(self, **kwargs):
        """Reset environment and return extended observation."""
        base_obs, info = self.env.reset(**kwargs)
        self.last_base_obs = base_obs
        expert_action = self._get_expert_action(base_obs)
        extended_obs = self._create_extended_obs(base_obs, expert_action)
        self.episode_count += 1
        return extended_obs, info

    def step(self, residual_action: torch.Tensor):
        """Take environment step with expert+residual action."""
        # Convert action to tensor
        residual_action = common.to_tensor(residual_action, device=self.device)

        # Get expert action for current observation
        expert_action = self._get_expert_action(self.last_base_obs)

        # Scale residual action
        scaled_residual = residual_action * self.residual_scale

        # Combine expert and residual actions
        final_action = 0.2 * expert_action + scaled_residual

        # Clip final action if requested
        if self.clip_final_action:
            # Handle dimension mismatch between final action and action bounds
            if final_action.dim() == 1:
                # Single environment case
                action_low = self._action_low[: final_action.shape[0]]
                action_high = self._action_high[: final_action.shape[0]]
                final_action = torch.clamp(final_action, action_low, action_high)
            else:
                # Batch case - apply bounds to each environment
                action_low = self._action_low[: final_action.shape[1]]
                action_high = self._action_high[: final_action.shape[1]]
                final_action = torch.clamp(final_action, action_low, action_high)

        # Update statistics if tracking is enabled
        if self.track_action_stats:
            self._update_action_stats(expert_action, scaled_residual)

        # Log action decomposition if requested
        self.logger.info(f"Action decomposition (step {self.step_count}):")
        self.logger.info(f"  Expert:   {expert_action}")
        self.logger.info(f"  Residual: {scaled_residual}")
        self.logger.info(f"  Final:    {final_action}")

        # Take environment step
        next_base_obs, reward, terminated, truncated, info = self.env.step(final_action)

        # Store base observation for next step
        self.last_base_obs = next_base_obs

        # Get expert action for next observation
        next_expert_action = self._get_expert_action(next_base_obs)

        # Create extended next observation
        next_extended_obs = self._create_extended_obs(next_base_obs, next_expert_action)

        # Add action decomposition to info
        info["expert_action"] = expert_action
        info["residual_action"] = scaled_residual
        info["final_action"] = final_action

        self.step_count += 1

        return next_extended_obs, reward, terminated, truncated, info

    def _get_expert_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Get expert action for given observation."""
        # Handle different observation types
        if isinstance(obs, dict):
            # Check expert type for proper observation handling
            if self.expert_type == "ppo":
                # PPO state expert expects flattened observations
                if self._expects_flattened_obs:
                    obs_input = self._flatten_dict_obs(obs)
                else:
                    obs_input = obs
            elif self.expert_type == "ppo_rgb":
                # PPO RGB expert expects dict observations with RGB and state
                obs_input = obs
            elif self.expert_type == "model":
                # Model expert type - check if it's RGB-based by looking at observation structure
                if "rgb" in obs:
                    # This is likely an RGB model, pass dict observations
                    obs_input = obs
                # This is likely a state model, flatten the observations
                elif self._expects_flattened_obs:
                    obs_input = self._flatten_dict_obs(obs)
                else:
                    obs_input = obs
            else:
                # Check if this is an ACT expert policy by looking at the function name
                expert_func_name = getattr(self.expert_policy_fn, "__name__", "unknown")
                is_act_expert = (
                    isinstance(expert_func_name, str)
                    and "act_expert" in expert_func_name
                )

                if is_act_expert:
                    # For ACT expert policies, pass dict observations directly
                    # ACT expert policies can handle both flattened and raw observations
                    obs_input = obs
                elif self._expects_flattened_obs:
                    # For other expert policies with flattened observations, flatten the dict
                    obs_input = self._flatten_dict_obs(obs)
                else:
                    # For other expert policies with raw dict observations
                    # This shouldn't happen with current setup, but handle it
                    obs_input = self._flatten_dict_obs(obs)
        else:
            # Box observations (state-only environments)
            obs_input = common.to_tensor(obs, device=self.device)

        expert_action = self.expert_policy_fn(obs_input)

        # Expert policy should return torch.Tensor - validate this expectation
        if not isinstance(expert_action, torch.Tensor):
            raise TypeError(
                f"Expert policy must return torch.Tensor, got {type(expert_action)}. "
                f"Expert policies should be designed to work with torch tensors directly."
            )

        # Ensure expert action is on the correct device
        expert_action = expert_action.to(self.device)

        # Add noise if specified
        if self.expert_action_noise > 0:
            noise = torch.randn_like(expert_action) * self.expert_action_noise

        # Ensure expert action is within bounds
        # Handle dimension mismatch between expert action and action bounds
        if expert_action.dim() == 1:
            # Single environment case
            action_low = self._action_low[: expert_action.shape[0]]
            action_high = self._action_high[: expert_action.shape[0]]
            expert_action = torch.clamp(expert_action, action_low, action_high)
        else:
            # Batch case - apply bounds to each environment
            action_low = self._action_low[: expert_action.shape[1]]
            action_high = self._action_high[: expert_action.shape[1]]
            expert_action = torch.clamp(expert_action, action_low, action_high)

        return expert_action.float()

    def _flatten_dict_obs(self, obs_dict: dict) -> torch.Tensor:
        """Flatten dictionary observations into a single tensor (for expert policy input)."""
        obs_parts = []
        batch_size = 1

        # For RGB observations, flatten the image dimensions but preserve batch
        if "rgb" in obs_dict:
            rgb_value = obs_dict["rgb"]
            if isinstance(rgb_value, torch.Tensor):
                batch_size = rgb_value.shape[0] if rgb_value.dim() > 0 else 1
                # Ensure tensor is on correct device
                rgb_value = rgb_value.to(self.device).float()
                if rgb_value.dim() == 1:
                    obs_parts.append(rgb_value)
                else:
                    # Flatten image dimensions: (B, H, W, C) -> (B, H*W*C)
                    obs_parts.append(rgb_value.flatten(start_dim=1))
            elif isinstance(rgb_value, np.ndarray):
                tensor_val = torch.from_numpy(rgb_value).to(self.device).float()
                batch_size = tensor_val.shape[0] if tensor_val.dim() > 0 else 1
                if tensor_val.dim() == 1:
                    obs_parts.append(tensor_val)
                else:
                    obs_parts.append(tensor_val.flatten(start_dim=1))

        # For state observations, use as-is (already flattened)
        if "state" in obs_dict:
            state_value = obs_dict["state"]
            if isinstance(state_value, torch.Tensor):
                batch_size = state_value.shape[0] if state_value.dim() > 0 else 1
                # Ensure tensor is on correct device
                state_value = state_value.to(self.device).float()
                if state_value.dim() == 1:
                    obs_parts.append(state_value)
                else:
                    obs_parts.append(state_value.flatten(start_dim=1))
            elif isinstance(state_value, np.ndarray):
                tensor_val = torch.from_numpy(state_value).to(self.device).float()
                batch_size = tensor_val.shape[0] if tensor_val.dim() > 0 else 1
                if tensor_val.dim() == 1:
                    obs_parts.append(tensor_val)
                else:
                    obs_parts.append(tensor_val.flatten(start_dim=1))

        if obs_parts:
            flattened = torch.cat(obs_parts, dim=-1).float()
            return flattened
        else:
            # Fallback to small tensor if no valid observations found
            # This is likely called during initialization - return proper batch size
            return torch.zeros(batch_size, 1, device=self.device, dtype=torch.float32)

    def _create_extended_obs(
        self, base_obs: torch.Tensor, expert_action: torch.Tensor
    ) -> torch.Tensor:
        """Create extended observation including expert action."""
        base_obs = common.to_tensor(base_obs, device=self.device)
        expert_action = common.to_tensor(expert_action, device=self.device)

        if isinstance(self.base_observation_space, spaces.Box):
            # Handle Box observation space (state observations)
            return self._create_extended_box_obs(base_obs, expert_action)
        elif isinstance(self.base_observation_space, spaces.Dict):
            # Handle Dict observation space
            if self._expects_flattened_obs:
                # RGB observations that have been flattened
                return self._create_extended_dict_obs(base_obs, expert_action)
            else:
                # Raw dict observations - shouldn't happen with current setup
                raise ValueError(
                    "Raw dict observations not supported. "
                    "ExpertResidualWrapper expected flattened observations."
                )
        else:
            raise ValueError(
                f"Unsupported observation space type: {type(self.base_observation_space)}"
            )

    def _create_extended_box_obs(
        self, base_obs: torch.Tensor, expert_action: torch.Tensor
    ) -> torch.Tensor:
        """Create extended observation for Box observation spaces."""
        # Handle observation concatenation for both single and vectorized environments
        if base_obs.dim() == 1:
            # Single environment observation: (obs_dim,)
            # Expert action should be: (action_dim,)
            return torch.cat([base_obs, expert_action], dim=-1).float()
        elif base_obs.dim() == 2:
            # Vectorized environment observation: (num_envs, obs_dim)
            # Expert action should be: (num_envs, action_dim)
            if expert_action.dim() == 1:
                # Expand expert action to match batch dimension
                expert_action = expert_action.unsqueeze(0).expand(base_obs.shape[0], -1)
            return torch.cat([base_obs, expert_action], dim=-1).float()
        # Handle higher-dimensional observations by flattening
        elif base_obs.shape[0] == 1:
            # Single environment case - flatten everything
            flattened_obs = base_obs.flatten()
            flattened_expert = expert_action.flatten()
            return torch.cat([flattened_obs, flattened_expert], dim=-1).float()
        else:
            # Batched case - flatten each observation but preserve batch dimension
            flattened_obs = base_obs.flatten(start_dim=1)
            if expert_action.dim() == 1:
                expert_action = expert_action.unsqueeze(0).expand(base_obs.shape[0], -1)
            return torch.cat([flattened_obs, expert_action], dim=-1).float()

    def _create_extended_dict_obs(
        self, base_obs: dict, expert_action: torch.Tensor
    ) -> dict:
        """Create extended observation for Dict observation spaces with flattened structure.

        Expects base_obs to be in FlattenRGBDObservationWrapper format:
        - base_obs["state"]: Flattened state tensor
        - base_obs["rgb"]: RGB image tensor
        """
        # Expect FlattenRGBDObservationWrapper format
        if "state" not in base_obs:
            raise ValueError(
                "ExpertResidualWrapper expects FlattenRGBDObservationWrapper to be applied first. "
                "Expected 'state' key in base_obs, but got keys: "
                + str(list(base_obs.keys()))
            )

        extended_obs = {}

        # Handle RGB observations for PPO RGB mode
        if "rgb" in base_obs:
            extended_obs["rgb"] = base_obs["rgb"]
        # For PPO RGB expert policy, RGB data is required
        elif self.expert_type == "ppo_rgb":
            raise ValueError(
                "PPO RGB expert policy requires RGB data but none found in environment observation. "
                "Make sure the environment is configured with RGB cameras and FlattenRGBDObservationWrapper is applied."
            )
        # For ACT expert policy, RGB data is required
        elif self.expert_type == "act":
            raise ValueError(
                "ACT expert policy requires RGB data but none found in environment observation. "
                "Make sure the environment is configured with RGB cameras and FlattenRGBDObservationWrapper is applied."
            )

        # For other expert types in PPO RGB mode, create dummy RGB observation
        # This ensures consistency with PPO RGB mode expectations
        else:
            batch_size = 1
            if "state" in base_obs:
                state_tensor = base_obs["state"]
                if isinstance(state_tensor, torch.Tensor) and state_tensor.dim() > 1:
                    batch_size = state_tensor.shape[0]

            # Create a dummy RGB tensor (batch_size, H, W, C)
            dummy_rgb = torch.full(
                (batch_size, 224, 224, 3), 128, dtype=torch.uint8, device=self.device
            )
            extended_obs["rgb"] = dummy_rgb

        # Extend state with expert action
        base_state = base_obs["state"]
        if isinstance(base_state, torch.Tensor):
            base_state = base_state.float()
        else:
            base_state = torch.tensor(
                base_state, dtype=torch.float32, device=self.device
            )

        # Ensure expert action matches batch size
        if base_state.dim() == 1:
            # Single environment
            if expert_action.dim() == 1:
                extended_state = torch.cat([base_state, expert_action], dim=-1)
            else:
                extended_state = torch.cat(
                    [base_state, expert_action.squeeze(0)], dim=-1
                )
        else:
            # Batch environment
            batch_size = base_state.shape[0]
            if expert_action.dim() == 1:
                expert_action = expert_action.unsqueeze(0).expand(batch_size, -1)
            elif expert_action.shape[0] != batch_size:
                if expert_action.shape[0] == 1:
                    expert_action = expert_action.expand(batch_size, -1)
            extended_state = torch.cat([base_state, expert_action], dim=-1)

        extended_obs["state"] = extended_state

        return extended_obs

    def _update_action_stats(
        self, expert_action: torch.Tensor, residual_action: torch.Tensor
    ) -> None:
        """Update action statistics for monitoring."""
        # Handle both single and batched actions
        if expert_action.dim() == 1:
            expert_mean = expert_action
            residual_mean = residual_action
            expert_min = expert_action
            expert_max = expert_action
            residual_min = residual_action
            residual_max = residual_action
        else:
            expert_mean = expert_action.mean(dim=0)
            residual_mean = residual_action.mean(dim=0)
            expert_min = expert_action.min(dim=0).values
            expert_max = expert_action.max(dim=0).values
            residual_min = residual_action.min(dim=0).values
            residual_max = residual_action.max(dim=0).values

        # Update expert action stats with exponential moving average
        self.expert_action_stats["mean"] = (
            0.99 * self.expert_action_stats["mean"] + 0.01 * expert_mean
        )
        self.expert_action_stats["min"] = torch.min(
            self.expert_action_stats["min"], expert_min
        )
        self.expert_action_stats["max"] = torch.max(
            self.expert_action_stats["max"], expert_max
        )

        # Update residual action stats
        self.residual_action_stats["mean"] = (
            0.99 * self.residual_action_stats["mean"] + 0.01 * residual_mean
        )
        self.residual_action_stats["min"] = torch.min(
            self.residual_action_stats["min"], residual_min
        )
        self.residual_action_stats["max"] = torch.max(
            self.residual_action_stats["max"], residual_max
        )

    def get_action_stats(self) -> Dict[str, Union[Dict[str, torch.Tensor], int, str]]:
        """Get action statistics for monitoring."""
        stats = {
            "episode_count": self.episode_count,
            "step_count": self.step_count,
            "num_envs": self.num_envs,
            "track_action_stats": self.track_action_stats,
        }

        if self.track_action_stats and self.expert_action_stats is not None:
            stats.update({
                "expert_action": {
                    k: v.clone() for k, v in self.expert_action_stats.items()
                },
                "residual_action": {
                    k: v.clone() for k, v in self.residual_action_stats.items()
                },
            })
        else:
            stats.update({
                "expert_action": "tracking_disabled",
                "residual_action": "tracking_disabled",
            })

        return stats

    def get_wrapper_info(self) -> Dict[str, Any]:
        """Get wrapper configuration information."""
        info = {
            "env_id": self.env_id,
            "expert_type": self.expert_type,
            "model_path": self.model_path,
            "deterministic_expert": self.deterministic_expert,
            "residual_scale": self.residual_scale,
            "clip_final_action": self.clip_final_action,
            "expert_action_noise": self.expert_action_noise,
            "track_action_stats": self.track_action_stats,
            "device": str(self.device),
            "num_envs": self.num_envs,
            "control_mode": self.control_mode,
        }

        # Add observation and action space information
        if hasattr(self, "base_observation_space"):
            if hasattr(self.base_observation_space, "shape"):
                info["base_obs_dim"] = self.base_observation_space.shape[0]
            else:
                info["base_obs_space_type"] = type(self.base_observation_space).__name__

        if hasattr(self, "base_action_space"):
            if hasattr(self.base_action_space, "shape"):
                info["action_dim"] = self.base_action_space.shape[0]
            else:
                info["action_space_type"] = type(self.base_action_space).__name__

        if hasattr(self, "observation_space"):
            if hasattr(self.observation_space, "shape"):
                info["extended_obs_dim"] = self.observation_space.shape[0]
            else:
                info["extended_obs_space_type"] = type(self.observation_space).__name__

        return info

    def close(self):
        """Close environment."""
        self.env.close()


# Example usage:
"""
# Example 1: Load PPO state expert for PickCube-v1
from mani_skill.envs.wrappers.expert_residual import ExpertResidualWrapper

# Create wrapper with PPO state expert
wrapper = ExpertResidualWrapper(
    env_id="PickCube-v1",
    expert_type="ppo",
    model_path="runs/ppo_state_model/final_ckpt.pt",
    num_envs=256,
    residual_scale=0.1,
    control_mode="pd_joint_delta_pos",
    robot_uids="panda"
)

# Use with standard training loop
obs, info = wrapper.reset()
for step in range(50):
    # Agent provides residual actions
    residual_actions = agent.predict(obs)
    obs, reward, terminated, truncated, info = wrapper.step(residual_actions)
    
    # Expert action info is in info["expert_action"]
    # Final action is in info["final_action"]

# Example 2: Load PPO RGB expert for PickCube-v1 with wrist camera
wrapper = ExpertResidualWrapper(
    env_id="PickCube-v1",
    expert_type="ppo_rgb",
    model_path="runs/ppo_rgb_model/final_ckpt.pt",
    num_envs=64,
    residual_scale=0.05,
    control_mode="pd_joint_delta_pos",
    robot_uids="panda_wristcam"
)

# Training loop is the same
obs, info = wrapper.reset()
for step in range(50):
    residual_actions = agent.predict(obs)
    obs, reward, terminated, truncated, info = wrapper.step(residual_actions)
"""
