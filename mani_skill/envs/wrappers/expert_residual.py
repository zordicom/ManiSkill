"""
Copyright 2025 Zordi, Inc. All rights reserved.

Expert+Residual action decomposition wrapper for ManiSkill environments.
Enables hybrid control where expert policy provides base actions and learned policy provides residual corrections.
"""

import logging
from collections.abc import Callable
from typing import Any, Dict, Union

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from mani_skill.utils import common


def create_zero_expert_policy(action_dim: int) -> callable:
    """
    Create zero expert policy (no expert knowledge).

    Args:
        action_dim: Action space dimension

    Returns:
        Expert policy function that returns torch.Tensor
    """

    def zero_expert_policy(obs: torch.Tensor) -> torch.Tensor:
        """Zero expert policy that returns zero actions and is never trainable."""
        with torch.no_grad():
            if obs.dim() == 1:
                return torch.zeros(action_dim, device=obs.device, dtype=torch.float32)
            else:
                batch_size = obs.shape[0]
                return torch.zeros(
                    batch_size, action_dim, device=obs.device, dtype=torch.float32
                )

    return zero_expert_policy


def create_ik_expert_policy(action_dim: int, gain: float = 2.0) -> callable:
    """
    Create IK-style expert policy for manipulation environments.

    This expert policy extracts target position from observation and uses
    proportional control to move the end-effector towards the target.

    Args:
        action_dim: Action space dimension
        gain: Proportional control gain

    Returns:
        Expert policy function that returns torch.Tensor
    """

    def ik_expert_policy(obs: torch.Tensor) -> torch.Tensor:
        """IK-style expert policy for manipulation environments."""
        # Handle both single and batched observations
        if obs.dim() == 1:
            batch_size = 1
            obs_batch = obs.unsqueeze(0)
        else:
            batch_size = obs.shape[0]
            obs_batch = obs

        # Extract relevant information from observation
        # This is a simplified IK policy - in practice you'd want to extract
        # actual TCP and target positions from the observation

        # For demonstration, we'll use a simple proportional controller
        # that moves towards the first 3 elements of the observation
        if obs_batch.shape[1] >= 6:
            # Assume first 3 elements are current position, next 3 are target
            current_pos = obs_batch[:, :3]
            target_pos = obs_batch[:, 3:6]

            # Compute proportional control action
            position_error = target_pos - current_pos
            action = torch.clamp(position_error * gain, -1.0, 1.0)

            # Pad to full action space
            if action.shape[1] < action_dim:
                padding = torch.zeros(
                    batch_size, action_dim - action.shape[1], device=obs.device
                )
                action = torch.cat([action, padding], dim=1)
        else:
            # Fallback to zero action if observation is too small
            action = torch.zeros(batch_size, action_dim, device=obs.device)

        # Handle single environment case
        if obs.dim() == 1:
            action = action.squeeze(0)

        return action.float()

    return ik_expert_policy


def create_model_expert_policy(
    model_path: str, action_dim: int, device: str = "cuda"
) -> callable:
    """
    Create expert policy from pre-trained model.

    Args:
        model_path: Path to pre-trained model
        action_dim: Action space dimension
        device: Device to run model on

    Returns:
        Expert policy function that returns torch.Tensor
    """
    # This would load a pre-trained model
    # Implementation depends on your model format

    def model_expert_policy(obs: torch.Tensor) -> torch.Tensor:
        """Expert policy from pre-trained model."""
        # Ensure we return tensor on same device as input
        device = obs.device

        # Load model and get action
        # This is a placeholder implementation
        if obs.dim() == 1:
            return torch.zeros(action_dim, device=device, dtype=torch.float32)
        else:
            batch_size = obs.shape[0]
            return torch.zeros(
                batch_size, action_dim, device=device, dtype=torch.float32
            )

    return model_expert_policy


def create_expert_policy(expert_type: str, action_dim: int, **kwargs) -> callable:
    """
    Create expert policy based on type.

    Args:
        expert_type: Type of expert policy ('zero', 'ik', 'model')
        action_dim: Action space dimension
        **kwargs: Additional arguments for specific expert types

    Returns:
        Expert policy function
    """
    if expert_type == "zero":
        return create_zero_expert_policy(action_dim)
    elif expert_type == "ik":
        gain = kwargs.get("gain", 2.0)
        return create_ik_expert_policy(action_dim, gain=gain)
    elif expert_type == "model":
        model_path = kwargs.get("model_path", "dummy_path")
        device = kwargs.get("device", "cuda")
        return create_model_expert_policy(model_path, action_dim, device=device)
    else:
        raise ValueError(
            f"Unknown expert type: {expert_type}. Available: 'zero', 'ik', 'model'"
        )


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

    Args:
        env_id: ManiSkill environment ID (e.g., "PickBox-v1")
        expert_policy_fn: Callable that takes observation tensor and returns expert action tensor.
            The expert policy MUST return torch.Tensor on the same device as the input observation.
        num_envs: Number of parallel environments (default: 1)
        residual_scale: Scale factor for residual actions (default: 1.0)
        clip_final_action: Whether to clip final action to action space bounds (default: True)
        expert_action_noise: Gaussian noise std to add to expert actions (default: 0.0)
        log_actions: Whether to log action decomposition for debugging (default: False)
        track_action_stats: Whether to track expert/residual action statistics (default: False)
        device: Device to place environments on (default: "cuda")
        **env_kwargs: Additional arguments for environment creation

    Usage Examples:
        # Single environment
        >>> wrapper = ExpertResidualWrapper("PickBox-v1", expert_policy)

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
        expert_policy_fn: Callable[[torch.Tensor], torch.Tensor],
        num_envs: int = 1,
        residual_scale: float = 1.0,
        clip_final_action: bool = True,
        expert_action_noise: float = 0.0,
        log_actions: bool = False,
        track_action_stats: bool = False,
        device: str = "cuda",
        **env_kwargs,
    ):
        """
        Initialize expert+residual wrapper.

        Args:
            env_id: ManiSkill environment ID (e.g., "PickBox-v1")
            expert_policy_fn: Function that takes observation tensor and returns expert action tensor.
                MUST return torch.Tensor. Input and output should be on the same device.
            num_envs: Number of parallel environments
            residual_scale: Scale factor for residual actions
            clip_final_action: Whether to clip final action to action space bounds
            expert_action_noise: Gaussian noise std to add to expert actions
            log_actions: Whether to log action decomposition for debugging
            track_action_stats: Whether to track expert/residual action statistics for monitoring
            device: Device to place environments on
            **env_kwargs: Additional environment creation arguments
        """
        # Let ManiSkill handle vectorization efficiently
        env = gym.make(env_id, num_envs=num_envs, **env_kwargs)
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

        self.logger = logging.getLogger(__name__)

        # Set up wrapper
        self._setup_wrapper()

        self.logger.info("✅ Expert+Residual wrapper initialized:")
        self.logger.info(f"   Environment: {env_id}")
        self.logger.info(f"   Parallel environments: {self.num_envs}")
        self.logger.info(f"   Residual scale: {self.residual_scale}")
        self.logger.info(f"   Track action stats: {self.track_action_stats}")
        self.logger.info(f"   Device: {self.device}")

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
        if "sensor_data" in self.base_observation_space.spaces:
            total_channels = 0
            image_height = None
            image_width = None
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

            if total_channels > 0 and image_height is not None:
                new_spaces["rgb"] = spaces.Box(
                    low=0,
                    high=255,
                    shape=(image_height, image_width, total_channels),
                    dtype=np.uint8,
                )

        # Calculate flattened state space (including expert_action)
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
        final_action = expert_action + scaled_residual

        # Clip final action if requested
        if self.clip_final_action:
            final_action = torch.clamp(
                final_action, self._action_low, self._action_high
            )

        # Update statistics if tracking is enabled
        if self.track_action_stats:
            self._update_action_stats(expert_action, scaled_residual)

        # Log action decomposition if requested
        if self.log_actions and self.step_count % 100 == 0:
            self.logger.debug(f"Action decomposition (step {self.step_count}):")
            self.logger.debug(f"  Expert:   {expert_action}")
            self.logger.debug(f"  Residual: {scaled_residual}")
            self.logger.debug(f"  Final:    {final_action}")

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
        try:
            # Handle different observation types
            if isinstance(obs, dict):
                # For Dict observations, try to flatten first
                # Expert policies typically expect flattened state
                obs_input = self._flatten_dict_obs(obs)
            else:
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
                expert_action = expert_action + noise

            # Ensure expert action is within bounds
            expert_action = torch.clamp(
                expert_action, self._action_low, self._action_high
            )

            return expert_action.float()

        except Exception as e:
            self.logger.error(f"Expert policy failed: {e}")
            if isinstance(obs, dict):
                self.logger.error(f"Dict observation keys: {list(obs.keys())}")
                for key, value in obs.items():
                    if hasattr(value, "shape"):
                        self.logger.error(f"  {key} shape: {value.shape}")
            else:
                self.logger.error(f"Observation shape: {obs.shape}")
                self.logger.error(f"Observation device: {obs.device}")
                if hasattr(obs, "min") and hasattr(obs, "max"):
                    self.logger.error(
                        f"Observation range: [{obs.min():.3f}, {obs.max():.3f}]"
                    )

            # Raise error instead of falling back to zero action
            raise RuntimeError(
                f"Expert policy failed and no fallback is available. "
                f"Original error: {e}. "
                f"Please check that your expert policy function is correctly implemented "
                f"and can handle observations of the given type and shape."
            ) from e

    def _flatten_dict_obs(self, obs_dict: dict) -> torch.Tensor:
        """Flatten dictionary observations into a single tensor (for expert policy input)."""
        obs_parts = []
        batch_size = 1

        # For RGB observations, flatten the image dimensions but preserve batch
        if "rgb" in obs_dict:
            rgb_value = obs_dict["rgb"]
            if isinstance(rgb_value, torch.Tensor):
                batch_size = rgb_value.shape[0] if rgb_value.dim() > 0 else 1
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
            # Handle Dict observation space (RGB/visual observations)
            return self._create_extended_dict_obs(base_obs, expert_action)
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
        """Create extended observation for Dict observation spaces with flattened structure."""
        # Create flattened observation structure compatible with FlattenRGBDObservationWrapper
        # Keep RGB in image format, flatten only state+expert_action
        flattened_obs = {}

        # Find the correct batch size by looking at actual observation tensors
        batch_size = 1
        if "sensor_data" in base_obs:
            for camera_name, camera_data in base_obs["sensor_data"].items():
                if "rgb" in camera_data:
                    rgb_data = camera_data["rgb"]
                    if rgb_data.dim() >= 1:
                        batch_size = rgb_data.shape[0]
                        break
        elif "extra" in base_obs:
            extra_obs = base_obs["extra"]
            if isinstance(extra_obs, dict):
                for key, value in extra_obs.items():
                    if isinstance(value, torch.Tensor) and value.dim() >= 1:
                        batch_size = value.shape[0]
                        break
            elif isinstance(extra_obs, torch.Tensor) and extra_obs.dim() >= 1:
                batch_size = extra_obs.shape[0]

        # Ensure expert action matches the batch size
        if expert_action.dim() == 1:
            # Single action for all envs, expand to [batch_size, action_dim]
            if batch_size > 1:
                expert_action = expert_action.unsqueeze(0).expand(batch_size, -1)
            else:
                expert_action = expert_action.unsqueeze(0)
        elif expert_action.shape[0] != batch_size:
            # Batch size mismatch, this shouldn't happen but let's handle it
            if expert_action.shape[0] == 1 and batch_size > 1:
                # Expand single action to match batch size
                expert_action = expert_action.expand(batch_size, -1)

        # Keep RGB observations in image format (concatenate cameras if multiple)
        if "sensor_data" in base_obs:
            rgb_tensors = []
            for camera_name, camera_data in base_obs["sensor_data"].items():
                if "rgb" in camera_data:
                    rgb_data = camera_data["rgb"]
                    rgb_tensors.append(rgb_data)

            if len(rgb_tensors) == 1:
                # Single camera - keep original format
                flattened_obs["rgb"] = rgb_tensors[0]
            elif len(rgb_tensors) > 1:
                # Multiple cameras - concatenate along channel dimension
                # Assuming RGB data is (B, H, W, C), concatenate along C
                flattened_obs["rgb"] = torch.cat(rgb_tensors, dim=-1)

        # Flatten state observations from "agent" and "extra" and concatenate with expert_action
        state_tensors = []

        # Add agent components (qpos, qvel, etc.)
        if "agent" in base_obs:
            agent_obs = base_obs["agent"]
            if isinstance(agent_obs, dict):
                # Process agent components in sorted order for consistency
                for key in sorted(agent_obs.keys()):
                    value = agent_obs[key]
                    if isinstance(value, torch.Tensor):
                        # Convert to float32 to handle any dtype issues
                        value = value.float()
                        if value.dim() > 1:
                            state_tensors.append(value.flatten(start_dim=1))
                        else:
                            state_tensors.append(
                                value.unsqueeze(1) if value.dim() == 1 else value
                            )
            elif isinstance(agent_obs, torch.Tensor):
                # Direct tensor - convert to float32 first
                agent_obs = agent_obs.float()
                if agent_obs.dim() > 1:
                    state_tensors.append(agent_obs.flatten(start_dim=1))
                else:
                    state_tensors.append(agent_obs.unsqueeze(1))

        if "extra" in base_obs:
            extra_obs = base_obs["extra"]
            if isinstance(extra_obs, dict):
                # Flatten all state data from extra
                for key in sorted(extra_obs.keys()):
                    value = extra_obs[key]
                    if isinstance(value, torch.Tensor):
                        # Convert to float32 to handle boolean and other dtypes
                        value = value.float()
                        if value.dim() > 1:
                            state_tensors.append(value.flatten(start_dim=1))
                        else:
                            state_tensors.append(
                                value.unsqueeze(1) if value.dim() == 1 else value
                            )
            elif isinstance(extra_obs, torch.Tensor):
                # Direct tensor - convert to float32 first
                extra_obs = extra_obs.float()
                if extra_obs.dim() > 1:
                    state_tensors.append(extra_obs.flatten(start_dim=1))
                else:
                    state_tensors.append(extra_obs.unsqueeze(1))

        # Add expert_action to state (should now have correct batch size)
        if expert_action.dim() == 1:
            state_tensors.append(expert_action.unsqueeze(1))
        else:
            state_tensors.append(expert_action)

        # Concatenate all state data including expert_action
        if state_tensors:
            flattened_obs["state"] = torch.cat(state_tensors, dim=-1)

        return flattened_obs

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
        return {
            "env_id": self.env_id,
            "residual_scale": self.residual_scale,
            "clip_final_action": self.clip_final_action,
            "expert_action_noise": self.expert_action_noise,
            "track_action_stats": self.track_action_stats,
            "base_obs_dim": self.base_observation_space.shape[0],
            "action_dim": self.base_action_space.shape[0],
            "extended_obs_dim": self.observation_space.shape[0],
            "device": str(self.device),
            "num_envs": self.num_envs,
        }

    def close(self):
        """Close environment."""
        self.env.close()
