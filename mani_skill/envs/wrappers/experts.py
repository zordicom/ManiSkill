"""
Expert policy implementations for ManiSkill environments.

This module contains various expert policy implementations including:
- Zero expert policy (returns zeros)
- IK expert policy (inverse kinematics-based)
- Model expert policy (pre-trained model-based)
- ACT expert policy (Action Chunking Transformer)
"""

import logging
from typing import Any, Dict, Union

import gymnasium as gym
import numpy as np
import torch


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

    IMPORTANT: This policy is designed for end-effector position control
    and should be used with control_mode="pd_ee_delta_pos".

    Args:
        action_dim: Action space dimension (should be 4 for pd_ee_delta_pos: 3 pos + 1 gripper)
        gain: Proportional control gain

    Returns:
        Expert policy function that returns torch.Tensor
    """
    # Validate action dimension for end-effector position control
    if action_dim != 4:
        import warnings

        warnings.warn(
            f"IK expert policy expects action_dim=4 for pd_ee_delta_pos control "
            f"(3 position + 1 gripper), but got action_dim={action_dim}. "
            f"This may cause issues. Consider using pd_ee_delta_pos control mode."
        )

    def ik_expert_policy(obs: torch.Tensor) -> torch.Tensor:
        """IK-style expert policy for end-effector position control."""
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

            # Compute proportional control action for position (first 3 dimensions)
            position_error = target_pos - current_pos
            position_action = torch.clamp(position_error * gain, -1.0, 1.0)

            # Create action tensor
            action = torch.zeros(batch_size, action_dim, device=obs.device)
            action[:, :3] = position_action  # Position delta

            # Set gripper action based on distance to target
            if action_dim >= 4:
                distance = torch.norm(position_error, dim=1, keepdim=True)
                # Close gripper when close to target, open when far
                gripper_action = torch.where(
                    distance < 0.05,
                    torch.ones_like(distance),
                    -torch.ones_like(distance),
                )
                action[:, 3:4] = gripper_action  # Gripper action

        else:
            # Fallback to zero action if observation is too small
            action = torch.zeros(batch_size, action_dim, device=obs.device)

        # Handle single environment case
        if obs.dim() == 1:
            action = action.squeeze(0)

        return action.float() * 0.05

    return ik_expert_policy


def create_model_expert_policy(
    model_path: str, action_dim: int, device: str = "cuda"
) -> callable:
    """
    Create expert policy from pre-trained model.

    Supports both PPO state models and PPO RGB models.
    Automatically detects model type from checkpoint structure.

    Args:
        model_path: Path to pre-trained model
        action_dim: Action space dimension
        device: Device to run model on

    Returns:
        Expert policy function that returns torch.Tensor
    """
    import os
    import sys
    from pathlib import Path

    import torch
    from torch import nn
    from torch.distributions.normal import Normal

    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    # Load checkpoint to inspect its structure
    checkpoint = torch.load(model_path, map_location=device)

    # Detect model type based on checkpoint keys
    checkpoint_keys = list(checkpoint.keys())
    is_rgb_model = any("cnn" in key for key in checkpoint_keys)

    # Detect action dimension from checkpoint for RGB models
    if is_rgb_model:
        # Get action dimension from the actor_mean output layer
        if "actor_mean.4.weight" in checkpoint:
            actual_action_dim = checkpoint["actor_mean.4.weight"].shape[0]
        elif "actor_mean.6.weight" in checkpoint:
            actual_action_dim = checkpoint["actor_mean.6.weight"].shape[0]
        else:
            # Fallback to provided action_dim
            actual_action_dim = action_dim

        if actual_action_dim != action_dim:
            print(
                f"🔧 Detected action dimension mismatch: checkpoint has {actual_action_dim}, environment expects {action_dim}"
            )
            print(f"   Using checkpoint action dimension: {actual_action_dim}")
            action_dim = actual_action_dim

    print(f"🔍 Detected model type: {'RGB' if is_rgb_model else 'State'}")
    print(f"📊 Action dimension: {action_dim}")

    if is_rgb_model:
        # Load PPO RGB model using the actual RGBFastAgent like in dataset generator
        # Add path to import from ppo_rgb_fast
        ppo_path = (
            Path(__file__).parent.parent.parent.parent
            / "examples"
            / "baselines"
            / "ppo"
        )
        sys.path.append(str(ppo_path))

        try:
            from ppo_rgb_fast import Agent as RGBFastAgent
        except ImportError as e:
            raise ImportError(
                f"Could not import RGBFastAgent from ppo_rgb_fast. "
                f"Make sure the path is correct: {ppo_path}. Error: {e}"
            )

        # Create environment to get sample observation - match dataset generator approach
        from mani_skill.utils.wrappers.flatten import (
            FlattenActionSpaceWrapper,
            FlattenRGBDObservationWrapper,
        )

        try:
            # Create environment exactly like the dataset generator
            env_config = {
                "id": "PickCube-v1",
                "robot_uids": "panda_wristcam",
                "control_mode": "pd_ee_delta_pos",
                "obs_mode": "rgb",
                "render_mode": "all",
                "sim_backend": "physx_cuda",
            }
            temp_env = gym.make(**env_config)

            # Apply wrappers exactly like dataset generator
            temp_env = FlattenRGBDObservationWrapper(temp_env, rgb=True, state=True)
            if isinstance(temp_env.action_space, gym.spaces.Dict):
                temp_env = FlattenActionSpaceWrapper(temp_env)

            sample_obs = temp_env.reset()[0]
            temp_env.close()
        except Exception as e:
            print(f"Warning: Could not create sample observation from environment: {e}")
            # Fallback to hardcoded observation structure
            sample_obs = {
                "rgb": torch.zeros(128, 128, 6, dtype=torch.uint8),  # Common RGB shape
                "state": torch.zeros(29, dtype=torch.float32),  # Common state shape
            }

        # Create agent exactly like the dataset generator
        import math

        n_act = math.prod((action_dim,))  # Handle single dimension case
        agent = RGBFastAgent(n_act, sample_obs, device=device)

        def model_expert_policy(obs: torch.Tensor) -> torch.Tensor:
            """Expert policy from pre-trained RGB model."""
            with torch.no_grad():
                # RGB model expects dictionary observations
                if not isinstance(obs, dict):
                    raise ValueError(
                        f"RGB model expects dictionary observation, got {type(obs)}"
                    )

                # Prepare observation for agent - match dataset generator exactly
                obs_device = {}
                single_env = False

                for key, value in obs.items():
                    if isinstance(value, torch.Tensor):
                        obs_device[key] = value.to(device)
                    else:
                        obs_device[key] = torch.from_numpy(value).to(device)

                    # Convert RGB to float - match dataset generator exactly
                    if "rgb" in key:
                        obs_device[key] = obs_device[key].float()

                    # Handle single environment case
                    if obs_device[key].dim() == 1 and key == "state":
                        obs_device[key] = obs_device[key].unsqueeze(0)
                        single_env = True
                    elif obs_device[key].dim() == 3 and key == "rgb":
                        obs_device[key] = obs_device[key].unsqueeze(0)
                        single_env = True

                # Get action from agent - match dataset generator exactly
                features = agent.get_features(obs_device)
                action = agent.actor_mean(features)  # Deterministic action

                # Remove batch dimension if we added it
                if single_env:
                    action = action.squeeze(0)

                return action.float()

    else:
        # Load regular PPO state model
        # Create a dummy environment to get observation space info
        class DummyEnv:
            def __init__(self):
                self.single_observation_space = gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(42,),
                    dtype=np.float32,  # Common state dim
                )
                self.single_action_space = gym.spaces.Box(
                    low=-1, high=1, shape=(action_dim,), dtype=np.float32
                )

        # Import the regular PPO Agent
        ppo_path = (
            Path(__file__).parent.parent.parent.parent
            / "examples"
            / "baselines"
            / "ppo"
        )
        sys.path.append(str(ppo_path))

        try:
            from ppo import Agent as PPOAgent
        except ImportError as e:
            raise ImportError(
                f"Could not import PPOAgent from ppo. "
                f"Make sure the path is correct: {ppo_path}. Error: {e}"
            )

        dummy_env = DummyEnv()
        agent = PPOAgent(dummy_env).to(device)

        def model_expert_policy(obs: torch.Tensor) -> torch.Tensor:
            """Expert policy from pre-trained state model."""
            with torch.no_grad():
                # Ensure obs is on the correct device
                obs = obs.to(device).float()

                # Handle both single and batched observations
                if obs.dim() == 1:
                    obs = obs.unsqueeze(0)
                    single_env = True
                else:
                    single_env = False

                # For state observations, flatten if needed
                if isinstance(obs, dict):
                    # Flatten dictionary observations
                    obs_parts = []
                    for k in sorted(obs.keys()):
                        v = obs[k]
                        if isinstance(v, torch.Tensor):
                            v = v.to(device).float()
                        else:
                            v = torch.tensor(v, device=device).float()
                        obs_parts.append(v.flatten(start_dim=1))
                    obs = torch.cat(obs_parts, dim=-1)
                else:
                    obs = obs.flatten(start_dim=1) if obs.dim() > 1 else obs

                # Get deterministic action from trained agent
                action = agent.get_action(obs, deterministic=True)

                # Return to original shape if needed
                if single_env:
                    action = action.squeeze(0)

                return action.float()

    # Load the trained weights
    try:
        agent.load_state_dict(checkpoint, strict=True)
        agent.eval()
        print(
            f"✅ Successfully loaded {'RGB' if is_rgb_model else 'State'} PPO model from {model_path}"
        )
    except Exception as e:
        print(f"❌ Error loading PPO model: {e}")
        raise

    return model_expert_policy


def create_act_expert_policy(
    config_path: str,
    checkpoint_path: str,
    action_dim: int,
    device: str = "cuda",
    action_offset: int = 0,
    action_clamp: float = 0.5,
    control_mode: str = "pd_joint_delta_pos",
) -> callable:
    """
    Create ACT expert policy from pre-trained ACT model.

    ACT model has fixed output format: [6 joints, 1 gripper, 7 ee_pose(3pos+4quat)] = 14 dims
    This function converts ACT output to the environment's expected action format.

    Args:
        config_path: Path to ACT model configuration
        checkpoint_path: Path to ACT model checkpoint
        action_dim: Environment action space dimension
        device: Device to run model on
        action_offset: Offset for selecting action from action chunk
        action_clamp: Clamp range for actions
        control_mode: Environment control mode for proper format conversion

    Returns:
        Expert policy function that returns torch.Tensor in environment format
    """
    # Import ACT-related dependencies
    from zordi_vla.common.normalizers import (
        DepthNormalizer,
        ImageNormalizer,
        MaskNormalizer,
        MeanStdNormalizer,
        MinMaxNormalizer,
    )
    from zordi_vla.configs.configs import ImageObsConfig
    from zordi_vla.models.policies.unified_policy_model import UnifiedPolicyModel
    from zordi_vla.utils.io_utils import load_modular_config

    # Load ACT model
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    act_config = load_modular_config(config_path)
    act_model = UnifiedPolicyModel(act_config).to(device_obj)

    # Load checkpoint
    checkpoint = torch.load(
        checkpoint_path, map_location=device_obj, weights_only=False
    )
    raw_sd = checkpoint.get(
        "model_state_dict", checkpoint.get("state_dict", checkpoint)
    )
    if not isinstance(raw_sd, dict):
        raise RuntimeError(f"Unexpected checkpoint format: {type(raw_sd)}")

    # Strip torch.compile prefixes if present
    state_dict_clean = {k.removeprefix("_orig_mod."): v for k, v in raw_sd.items()}
    act_model.load_state_dict(state_dict_clean)
    act_model.eval()

    # Initialize normalizers from checkpoint
    def _initialize_normalizers_from_checkpoint(checkpoint: Dict[str, Any]) -> None:
        stats = checkpoint.get("dataset_stats", {})

        # State normalizer
        state_norm_cfg = stats.get("state_normalizer")
        state_normalizer = None
        if isinstance(state_norm_cfg, dict):
            if state_norm_cfg.get("type") == "meanstd":
                data = {
                    "mean": np.array(state_norm_cfg["mean"], dtype=np.float32),
                    "std": np.array(state_norm_cfg["std"], dtype=np.float32),
                }
                state_normalizer = MeanStdNormalizer(data)
            else:
                data = {
                    "min": np.array(state_norm_cfg["min"], dtype=np.float32),
                    "max": np.array(state_norm_cfg["max"], dtype=np.float32),
                }
                state_normalizer = MinMaxNormalizer(data)

        # Action normalizer
        action_norm_cfg = stats.get("action_normalizer")
        action_normalizer = None
        if isinstance(action_norm_cfg, dict):
            if action_norm_cfg.get("type") == "meanstd":
                data = {
                    "mean": np.array(action_norm_cfg["mean"], dtype=np.float32),
                    "std": np.array(action_norm_cfg["std"], dtype=np.float32),
                }
                action_normalizer = MeanStdNormalizer(data)
            else:
                data = {
                    "min": np.array(action_norm_cfg["min"], dtype=np.float32),
                    "max": np.array(action_norm_cfg["max"], dtype=np.float32),
                }
                action_normalizer = MinMaxNormalizer(data)

        # Image normalizers
        rgb_normalizer = ImageNormalizer()

        dataset_cfg = getattr(act_config, "dataset", None)
        mask_num_classes = (
            getattr(dataset_cfg, "mask_num_classes", 256) if dataset_cfg else 256
        )
        mask_normalizer = MaskNormalizer(num_actual_classes=mask_num_classes)

        depth_clipping_range = (
            getattr(dataset_cfg, "depth_clipping_range", [0.1, 10.0])
            if dataset_cfg
            else [0.1, 10.0]
        )
        depth_unit_scale = (
            getattr(dataset_cfg, "depth_unit_scale", 1000.0) if dataset_cfg else 1000.0
        )
        depth_normalizer = DepthNormalizer(
            min_depth_m=depth_clipping_range[0],
            max_depth_m=depth_clipping_range[1],
            depth_unit_scale=depth_unit_scale,
        )

        # Attach normalizers to the model
        act_model.set_normalizers(
            state_normalizer=state_normalizer,
            action_normalizer=action_normalizer,
            rgb_normalizer=rgb_normalizer,
            mask_normalizer=mask_normalizer,
            depth_normalizer=depth_normalizer,
        )

    _initialize_normalizers_from_checkpoint(checkpoint)

    def _convert_ppo_rgb_obs_to_semantic_format(obs: Dict[str, Any]) -> torch.Tensor:
        """Convert PPO RGB mode observations to semantic state format.

        Extracts the 28D semantic state from PPO RGB mode observation:
        - obs["state"]: Flattened state tensor with 28+ dimensions

        Returns:
            torch.Tensor: Semantic state tensor [batch_size, 28] or [28]
        """
        if "state" not in obs:
            raise ValueError("Expected 'state' key in observation for PPO RGB mode")

        state = obs["state"]
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        elif isinstance(state, torch.Tensor):
            state = state.float()
        else:
            raise ValueError(f"Unexpected state type: {type(state)}")

        # Handle extended state from ExpertResidualWrapper
        # State format: [original_state (28D), expert_action (7D)] = 35D total
        # But we only want the original 28D state
        if state.shape[-1] >= 28:
            # Extract only the first 28 dimensions (original state)
            state = state[..., :28]
        else:
            raise ValueError(
                f"Expected state to have at least 28 dimensions, got {state.shape[-1]}"
            )

        return state

    def _convert_act_output_to_env_format(
        act_actions: torch.Tensor, obs_batch: torch.Tensor
    ) -> torch.Tensor:
        """Convert ACT model output to environment-expected format."""
        batch_size = obs_batch.shape[0]

        # ACT outputs absolute joint targets: 6 arm joints + 1 gripper action = 7 total
        act_joint_targets = act_actions[:, :7]  # Use first 7 values from ACT output

        if control_mode == "pd_joint_delta_pos":
            current_joints = obs_batch[:, :action_dim]
            expert_delta = act_joint_targets[:, :7] - current_joints[:, :7]
            result = torch.zeros(batch_size, action_dim, device=obs_batch.device)
            result[:, :7] = expert_delta.clamp(-action_clamp, action_clamp)
        elif control_mode == "pd_joint_pos":
            # Environment expects absolute joint positions
            result = torch.zeros(batch_size, action_dim, device=obs_batch.device)
            result[:, :7] = act_joint_targets[:, :7]
        else:
            raise ValueError(f"Unsupported control mode: {control_mode}")

        return result

    def act_expert_policy(obs: Union[torch.Tensor, Dict[str, Any]]) -> torch.Tensor:
        """ACT expert policy that handles format conversion and history management."""
        # Initialize history buffers if not exists (per environment)
        if not hasattr(act_expert_policy, "history_buffers"):
            act_expert_policy.history_buffers = {}
            act_expert_policy.step_counts = {}

            # Convert PPO RGB mode observations to semantic format for history management
        if isinstance(obs, dict) and "state" in obs:
            # Convert PPO RGB mode observations to semantic format
            semantic_state = _convert_ppo_rgb_obs_to_semantic_format(obs)
            current_state = semantic_state.cpu().numpy()

            # Extract batch size
            batch_size = 1
            if current_state.ndim == 1:
                current_state = current_state.reshape(1, -1)
            else:
                batch_size = current_state.shape[0]
        else:
            # Determine batch size and handle different observation types
            batch_size = 1
        if isinstance(obs, dict):  # noqa: PLR1702
            # For dict observations, extract batch size and semantic components
            if "agent" in obs and isinstance(obs["agent"], dict):
                if "qpos" in obs["agent"]:
                    qpos = obs["agent"]["qpos"]
                    if isinstance(qpos, torch.Tensor) and qpos.dim() > 1:
                        batch_size = qpos.shape[0]
                else:
                    # Multi-agent case
                    for key, value in obs["agent"].items():
                        if isinstance(value, dict) and "qpos" in value:
                            qpos = value["qpos"]
                            if isinstance(qpos, torch.Tensor) and qpos.dim() > 1:
                                batch_size = qpos.shape[0]
                            break
            elif "extra" in obs and isinstance(obs["extra"], dict):
                # Try to get batch size from extra data
                for key, value in obs["extra"].items():
                    if isinstance(value, torch.Tensor) and value.dim() > 1:
                        batch_size = value.shape[0]
                        break

            # Extract semantic components for history - same logic as convert_maniskill_obs_to_act_format
            semantic_state = torch.zeros(batch_size, 28, dtype=torch.float32)

            # 1. right_arm_joints [0:7]: 6 arm joints + 1 gripper from agent.qpos
            if "agent" in obs:
                agent_data = obs["agent"]
                if isinstance(agent_data, dict):
                    # Handle different agent structures
                    if "qpos" in agent_data:
                        # Single arm case
                        qpos = agent_data["qpos"]
                        if isinstance(qpos, torch.Tensor):
                            if qpos.dim() == 1:
                                qpos = qpos.unsqueeze(0)
                            # Take first 6 joints (arm) + 1 gripper joint (index 6)
                            semantic_state[:, 0:6] = qpos[:, :6]  # 6 arm joints
                            semantic_state[:, 6:7] = qpos[:, 6:7]  # 1 gripper joint
                        else:
                            qpos = torch.tensor(qpos, dtype=torch.float32)
                            if qpos.dim() == 1:
                                qpos = qpos.unsqueeze(0)
                            semantic_state[:, 0:6] = qpos[:, :6]  # 6 arm joints
                            semantic_state[:, 6:7] = qpos[:, 6:7]  # 1 gripper joint
                    else:
                        # Multi-agent case - look for right arm
                        right_arm_key = None
                        for key in agent_data.keys():
                            if isinstance(key, str):
                                if "1" in key or "right" in key.lower():
                                    right_arm_key = key
                                    break

                        if right_arm_key and isinstance(
                            agent_data[right_arm_key], dict
                        ):
                            right_agent = agent_data[right_arm_key]
                            if "qpos" in right_agent:
                                qpos = right_agent["qpos"]
                                if isinstance(qpos, torch.Tensor):
                                    if qpos.dim() == 1:
                                        qpos = qpos.unsqueeze(0)
                                    semantic_state[:, 0:6] = qpos[:, :6]  # 6 arm joints
                                    semantic_state[:, 6:7] = qpos[
                                        :, 6:7
                                    ]  # 1 gripper joint
                                else:
                                    qpos = torch.tensor(qpos, dtype=torch.float32)
                                    if qpos.dim() == 1:
                                        qpos = qpos.unsqueeze(0)
                                    semantic_state[:, 0:6] = qpos[:, :6]  # 6 arm joints
                                    semantic_state[:, 6:7] = qpos[
                                        :, 6:7
                                    ]  # 1 gripper joint

            # 2. right_arm_tool_pose [7:14]: TCP pose from extra
            if "extra" in obs:
                extra_data = obs["extra"]
                if isinstance(extra_data, dict):
                    # Use right_tcp_pose if available (bimanual), otherwise tcp_pose
                    tcp_pose_key = None
                    if "right_tcp_pose" in extra_data:
                        tcp_pose_key = "right_tcp_pose"
                    elif "tcp_pose" in extra_data:
                        tcp_pose_key = "tcp_pose"

                    if tcp_pose_key and tcp_pose_key in extra_data:
                        tcp_pose = extra_data[tcp_pose_key]
                        if isinstance(tcp_pose, torch.Tensor):
                            if tcp_pose.dim() == 1:
                                tcp_pose = tcp_pose.unsqueeze(0)
                            semantic_state[:, 7:14] = tcp_pose[:, :7]
                        else:
                            tcp_pose = torch.tensor(tcp_pose, dtype=torch.float32)
                            if tcp_pose.dim() == 1:
                                tcp_pose = tcp_pose.unsqueeze(0)
                            semantic_state[:, 7:14] = tcp_pose[:, :7]

                    # 3. pick_target_pose [14:21]: Object pose from extra
                    if "obj_pose" in extra_data:
                        obj_pose = extra_data["obj_pose"]
                        if isinstance(obj_pose, torch.Tensor):
                            if obj_pose.dim() == 1:
                                obj_pose = obj_pose.unsqueeze(0)
                            semantic_state[:, 14:21] = obj_pose[:, :7]
                        else:
                            obj_pose = torch.tensor(obj_pose, dtype=torch.float32)
                            if obj_pose.dim() == 1:
                                obj_pose = obj_pose.unsqueeze(0)
                            semantic_state[:, 14:21] = obj_pose[:, :7]

                    # 4. place_target_pose [21:28]: Goal pose constructed from goal_pos
                    if "goal_pos" in extra_data:
                        goal_pos = extra_data["goal_pos"]
                        if isinstance(goal_pos, torch.Tensor):
                            if goal_pos.dim() == 1:
                                goal_pos = goal_pos.unsqueeze(0)
                            # Construct 7D pose from 3D position (add identity quaternion)
                            semantic_state[:, 21:24] = goal_pos[:, :3]  # Position
                            semantic_state[:, 24:28] = (
                                torch.tensor([0, 0, 0, 1], dtype=torch.float32)
                                .unsqueeze(0)
                                .expand(batch_size, -1)
                            )  # Identity quaternion [x,y,z,w]
                        else:
                            goal_pos = torch.tensor(goal_pos, dtype=torch.float32)
                            if goal_pos.dim() == 1:
                                goal_pos = goal_pos.unsqueeze(0)
                            semantic_state[:, 21:24] = goal_pos[:, :3]  # Position
                            semantic_state[:, 24:28] = (
                                torch.tensor([0, 0, 0, 1], dtype=torch.float32)
                                .unsqueeze(0)
                                .expand(batch_size, -1)
                            )  # Identity quaternion [x,y,z,w]

            current_state = semantic_state.cpu().numpy()

        elif isinstance(obs, torch.Tensor):
            # For tensor observations, pad/trim to 28 dimensions
            current_state = obs.cpu().numpy()
            if current_state.ndim == 1:
                current_state = current_state.reshape(1, -1)
                batch_size = 1
            else:
                batch_size = current_state.shape[0]

            # Ensure state is at least 28 dimensions (pad with zeros if necessary)
            if current_state.shape[-1] < 28:
                padding_size = 28 - current_state.shape[-1]
                current_state = np.pad(
                    current_state,
                    ((0, 0), (0, padding_size)),
                    mode="constant",
                    constant_values=0,
                )

            # Take only first 28 dimensions if larger
            if current_state.shape[-1] > 28:
                current_state = current_state[..., :28]
        else:
            raise ValueError(f"Unexpected observation type: {type(obs)}")

        # Manage per-environment history buffers
        actions = []
        for env_idx in range(batch_size):
            # Initialize history buffer for this environment if needed
            if env_idx not in act_expert_policy.history_buffers:
                act_expert_policy.history_buffers[env_idx] = []
                act_expert_policy.step_counts[env_idx] = 0

            # Add current state to this environment's history buffer
            act_expert_policy.history_buffers[env_idx].append(current_state[env_idx])
            act_expert_policy.step_counts[env_idx] += 1

            # Keep only the last n_obs_steps states
            if len(act_expert_policy.history_buffers[env_idx]) > act_config.n_obs_steps:
                act_expert_policy.history_buffers[env_idx].pop(0)

            # For the first 8 timesteps, return current joint positions as action
            if act_expert_policy.step_counts[env_idx] <= 8:
                # Extract current joint positions from semantic state (first 7 dimensions)
                current_joints = (
                    torch.from_numpy(current_state[env_idx][:7]).float().to(device_obj)
                )

                # Convert to appropriate action format based on control mode
                if control_mode == "pd_joint_delta_pos":
                    # For delta position control, return zero delta (stay at current position)
                    action = torch.zeros(
                        action_dim, device=device_obj, dtype=torch.float32
                    )
                elif control_mode == "pd_joint_pos":
                    # For absolute position control, return current joint positions
                    action = torch.zeros(
                        action_dim, device=device_obj, dtype=torch.float32
                    )
                    action[:7] = current_joints[:7]  # Set joint positions
                else:
                    raise ValueError(f"Unsupported control mode: {control_mode}")

                actions.append(action)
                continue

            # If we don't have enough history yet, return zero action
            if len(act_expert_policy.history_buffers[env_idx]) < act_config.n_obs_steps:
                action = torch.zeros(action_dim, device=device_obj, dtype=torch.float32)
                actions.append(action)
                continue

            # We have enough history, generate action using ACT model
            with torch.no_grad():
                # Create ACT observation with this environment's history
                act_obs = {}

                # Stack this environment's history
                env_history = np.stack(
                    act_expert_policy.history_buffers[env_idx], axis=0
                )  # [n_obs_steps, 28]
                state_history = torch.from_numpy(
                    env_history
                ).float()  # [n_obs_steps, 28]
                act_obs["state"] = state_history

                # Add RGB images if present (required for ACT model)
                if "rgb" in obs:
                    rgb_data = obs["rgb"]
                    if isinstance(rgb_data, torch.Tensor):
                        if rgb_data.dim() > 3:  # Batched
                            rgb_tensor = rgb_data[env_idx]
                        else:
                            rgb_tensor = rgb_data
                    elif isinstance(rgb_data, np.ndarray):
                        if rgb_data.ndim > 3:  # Batched
                            rgb_tensor = torch.from_numpy(rgb_data[env_idx])
                        else:
                            rgb_tensor = torch.from_numpy(rgb_data)
                    else:
                        raise ValueError(f"Unexpected RGB data type: {type(rgb_data)}")

                    if rgb_tensor.dim() != 3:
                        raise ValueError(
                            f"RGB tensor must be HWC, got shape {rgb_tensor.shape}"
                        )

                    # Map PPO RGB mode cameras to ACT format
                    H, W, total_channels = rgb_tensor.shape

                    # Extract individual camera images according to PPO RGB mode format
                    if total_channels >= 9:  # All three cameras available
                        end_effector_rgb = rgb_tensor[:, :, 0:3]  # Channels 0-2
                        static_top_rgb = rgb_tensor[:, :, 3:6]  # Channels 3-5
                        eoat_left_rgb = rgb_tensor[:, :, 6:9]  # Channels 6-8
                    elif total_channels >= 6:  # Two cameras available
                        end_effector_rgb = rgb_tensor[:, :, 0:3]  # Channels 0-2
                        static_top_rgb = rgb_tensor[:, :, 3:6]  # Channels 3-5
                        # For missing left camera, use a black image
                        eoat_left_rgb = torch.zeros((H, W, 3), dtype=torch.uint8)
                    elif total_channels >= 3:  # Only one camera available
                        # Assume it's the end_effector_camera
                        end_effector_rgb = rgb_tensor[:, :, 0:3]
                        # Use black images for missing cameras
                        static_top_rgb = torch.zeros((H, W, 3), dtype=torch.uint8)
                        eoat_left_rgb = torch.zeros((H, W, 3), dtype=torch.uint8)
                    else:
                        raise ValueError(
                            f"Expected at least 3 RGB channels, got {total_channels}"
                        )

                    # Map to ACT expected camera names (ManiSkill provides RGB format)
                    act_obs["eoat_right_top_rgb"] = end_effector_rgb.cpu().numpy()
                    act_obs["static_top_rgb"] = static_top_rgb.cpu().numpy()
                    act_obs["eoat_left_top_rgb"] = eoat_left_rgb.cpu().numpy()
                else:
                    # RGB data is required for ACT model
                    raise ValueError(
                        "RGB data is required for ACT model but not found in observation"
                    )

                # Generate action using ACT model
                action_tensor = act_model.generate_actions(act_obs)

                # Extract action from sequence
                if action_tensor.dim() == 3:  # [batch, n_action_steps, action_dim]
                    action_idx = min(action_offset, action_tensor.shape[1] - 1)
                    action = action_tensor[0, action_idx, :].cpu()
                elif action_tensor.dim() == 2:
                    if action_tensor.shape[0] == act_config.n_action_steps:
                        action_idx = min(action_offset, action_tensor.shape[0] - 1)
                        action = action_tensor[action_idx, :].cpu()
                    else:
                        action = action_tensor[0, :].cpu()
                else:
                    action = action_tensor.cpu()

                act_actions = action.unsqueeze(0)  # Add batch dimension

                # Convert ACT output format to environment format
                # Create a dummy state tensor for conversion using the current environment's state
                if env_idx < len(current_state):
                    env_state = current_state[env_idx]
                    state_tensor = (
                        torch.from_numpy(env_state)
                        .float()
                        .unsqueeze(0)
                        .to(act_actions.device)
                    )

                # Ensure state_tensor has at least action_dim elements for conversion
                if state_tensor.shape[1] < action_dim:
                    padding_size = action_dim - state_tensor.shape[1]
                    state_tensor = torch.cat(
                        [
                            state_tensor,
                            torch.zeros(
                                state_tensor.shape[0],
                                padding_size,
                                device=state_tensor.device,
                            ),
                        ],
                        dim=1,
                    )

                result = _convert_act_output_to_env_format(act_actions, state_tensor)

                # Get the action for this environment and ensure it's on the correct device
                action = result.squeeze(0).float().to(device_obj)
                actions.append(action)

        # Stack all actions and return
        if batch_size == 1:
            return actions[0]
        else:
            return torch.stack(actions, dim=0)

    return act_expert_policy


def create_expert_policy(expert_type: str, action_dim: int, **kwargs) -> callable:
    if expert_type == "zero":
        return create_zero_expert_policy(action_dim)
    elif expert_type == "ik":
        # IK expert is designed for end-effector position control
        if action_dim != 4:
            raise ValueError(
                f"IK expert policy requires action_dim=4 for pd_ee_delta_pos control "
                f"(3 position + 1 gripper), but got action_dim={action_dim}. "
                f"Please use ExpertResidualWrapper with control_mode='pd_ee_delta_pos'."
            )
        gain = kwargs.get("gain", 2.0)
        return create_ik_expert_policy(action_dim, gain=gain)
    elif expert_type == "model":
        model_path = kwargs.get("model_path", "dummy_path")
        device = kwargs.get("device", "cuda")
        return create_model_expert_policy(model_path, action_dim, device=device)
    elif expert_type == "act":
        # ACT expert policy with format conversion
        config_path = kwargs.get("config_path")
        if config_path is None:
            config_path = "/home/gilwoo/workspace/zordi_vla/config/galaxea/box_pnp_16hz_14act/galaxea_act_16hz.yaml"

        checkpoint_path = kwargs.get("checkpoint_path")
        if checkpoint_path is None:
            checkpoint_path = "/home/gilwoo/workspace/zordi_vla/outputs/zordi_galaxea_box_pnp/20250709-galaxea_box_pnp_act_16hz_mixed_14act/checkpoints/latest.ckpt"

        device = kwargs.get("device", "cuda")
        action_offset = kwargs.get("action_offset", 0)
        action_clamp = kwargs.get("action_clamp", 0.5)
        control_mode = kwargs.get("control_mode", "pd_joint_delta_pos")

        # Let the ACT model loading provide more informative error messages
        # if the files don't exist instead of failing here
        return create_act_expert_policy(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            action_dim=action_dim,
            device=device,
            action_offset=action_offset,
            action_clamp=action_clamp,
            control_mode=control_mode,
        )

    else:
        raise ValueError(
            f"Unknown expert type: {expert_type}. Available: 'zero', 'ik', 'model', 'act'"
        )
