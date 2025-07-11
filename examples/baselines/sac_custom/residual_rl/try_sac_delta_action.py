#!/usr/bin/env python3
"""
Copyright 2025 Zordi, Inc. All rights reserved.

SAC implementation for delta action learning with multi-modal observations.

This script implements Soft Actor-Critic for learning residual/delta actions
on top of expert demonstrations. Key features:

- **Multi-modal observations** supporting state, images, and expert actions
- **Delta action learning** where the policy outputs corrections to expert actions
- **Dataset-based training** using RLDataset for offline RL data
- **GPU acceleration** with automatic device detection
- **Off-policy learning** with automatic temperature tuning
- **Dual Q-networks** with target networks for stability

The policy network learns to output small delta actions that are added to the expert
actions to improve performance. This approach leverages expert demonstrations while
allowing the agent to learn improvements through SAC's entropy-regularized framework.

Data flow:
1. Dataset provides: state history, expert_action, residual_action (target), rewards
2. Policy observes: [state, expert_action, images] → outputs delta_action
3. Target: delta_action should match the stored residual_action from dataset
4. Final action = expert_action + delta_action

This is an offline RL approach where we use SAC loss on pre-collected data rather
than fresh environment rollouts.

Example usage:
    python playground/rl/residual_rl/try_sac_delta_action.py \
        --config playground/rl/residual_rl/rl_galaxea_sac_box_pnp.yaml
"""

import argparse
import contextlib
import copy
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import cattrs
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import yaml
from rl_configs import NetworkConfig, RLConfig
from rl_dataset import RLDataset
from torch import nn, optim
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader

from zordi_vla.configs.configs import ImageObsConfig, ImageType, StateObsConfig
from zordi_vla.models.components.vision_encoders import build_vision_encoders
from zordi_vla.utils.logging_utils import setup_logger

logger = setup_logger("sac_delta_action")


# Simple TensorBoard logger
class SimpleTBLogger:
    """Lightweight TensorBoard logger."""

    def __init__(self, log_dir: str, enabled: bool = True):
        self.enabled = enabled
        self.writer = None

        if enabled:
            try:
                from torch.utils.tensorboard.writer import SummaryWriter

                self.writer = SummaryWriter(log_dir=log_dir)
                logger.info(f"TensorBoard logging enabled: {log_dir}")
            except ImportError:
                logger.warning("TensorBoard not available, disabling TB logging")
                self.enabled = False

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        if self.enabled and self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def close(self) -> None:
        if self.enabled and self.writer is not None:
            self.writer.close()


# Numerical stability constants
LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0  # SAC standard values


def _custom_json_dumps(obj: dict, max_indent_level: int = 3) -> str:
    """Custom JSON serializer that indents only up to specified level.

    Args:
        obj: Dictionary to serialize
        max_indent_level: Maximum depth to apply indentation (default: 3)

    Returns:
        JSON string with custom indentation
    """

    def _serialize_with_level(obj, level=0, indent_size=2):
        if level >= max_indent_level:
            # Beyond max level, serialize inline without indentation
            return json.dumps(obj, separators=(",", ":"))

        if isinstance(obj, dict):
            if not obj:
                return "{}"

            indent = " " * (level * indent_size)
            next_indent = " " * ((level + 1) * indent_size)

            items = []
            for key, value in obj.items():
                key_str = json.dumps(key)
                if level + 1 >= max_indent_level:
                    # Next level should be inline
                    value_str = json.dumps(value, separators=(",", ":"))
                else:
                    value_str = _serialize_with_level(value, level + 1, indent_size)
                items.append(f"{next_indent}{key_str}: {value_str}")

            return "{\n" + ",\n".join(items) + f"\n{indent}}}"

        elif isinstance(obj, list):
            if not obj:
                return "[]"

            indent = " " * (level * indent_size)
            next_indent = " " * ((level + 1) * indent_size)

            items = []
            for item in obj:
                if level + 1 >= max_indent_level:
                    # Next level should be inline
                    item_str = json.dumps(item, separators=(",", ":"))
                else:
                    item_str = _serialize_with_level(item, level + 1, indent_size)
                items.append(f"{next_indent}{item_str}")

            return "[\n" + ",\n".join(items) + f"\n{indent}]"

        else:
            # Primitive value
            return json.dumps(obj)

    return _serialize_with_level(obj)


def _init_weights(module: nn.Module) -> None:
    """Initialize weights using orthogonal initialization for numerical stability."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=1)
        nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.Conv2d):
        nn.init.orthogonal_(module.weight, gain=1)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


class MultimodalEncoder(nn.Module):
    """Encoder for multimodal observations
    (state + images + expert_action + extra_obs).
    """

    def __init__(
        self,
        shape_meta: Dict[str, Any],
        network_cfg: NetworkConfig,
        device: torch.device,
    ):
        """Initialize multimodal encoder."""
        super().__init__()
        self.device = device
        self.shape_meta = shape_meta
        self.network_cfg = network_cfg

        # State encoder (state comes as [B, history_len, state_dim])
        state_dim = shape_meta["obs"]["state"]["dim"]
        state_layers = []
        prev_dim = state_dim
        for dim in network_cfg.state_encoder_dims:
            state_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
            ])
            prev_dim = dim
        self.state_encoder = nn.Sequential(*state_layers).to(device)

        # Expert action encoder
        action_dim = shape_meta["action"]["shape"][0]
        expert_action_layers = []
        prev_dim = action_dim
        for dim in network_cfg.expert_action_encoder_dims:
            expert_action_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
            ])
            prev_dim = dim
        self.expert_action_encoder = nn.Sequential(*expert_action_layers).to(device)

        # Extra observations encoder (for axis information, etc.)
        self.extra_obs_encoders = nn.ModuleDict()
        extra_obs_feature_dim = 0

        for key, obs_meta in shape_meta["obs"].items():
            if key not in {"state", "expert_action"} and "image_type" not in obs_meta:
                # This is an extra observation field (like ee_x_axis, etc.)
                if "shape" in obs_meta and len(obs_meta["shape"]) == 1:
                    input_dim = obs_meta["shape"][0]
                    # Use configuration values for extra obs encoder dimensions
                    hidden_dim = getattr(network_cfg, "extra_obs_encoder_dim", 32)
                    output_dim = getattr(network_cfg, "extra_obs_output_dim", 16)

                    encoder = nn.Sequential(
                        nn.Linear(input_dim, hidden_dim),
                        nn.LayerNorm(hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, output_dim),
                        nn.LayerNorm(output_dim),
                        nn.ReLU(),
                    ).to(device)
                    self.extra_obs_encoders[key] = encoder
                    extra_obs_feature_dim += output_dim

        # Image encoders (now using build_vision_encoders)
        self.image_encoders = nn.ModuleDict()
        image_feature_dim = 0

        # Structure the obs config for build_vision_encoders
        structured_obs_config: Dict[str, Union[StateObsConfig, ImageObsConfig]] = {}
        for key, meta in self.shape_meta["obs"].items():
            if meta.get("image_type"):
                structured_obs_config[key] = ImageObsConfig(
                    channels=meta["shape"][0],
                    image_size=tuple(meta["shape"][1:]),  # type: ignore
                    image_type=ImageType(meta["image_type"]),
                )
            elif key == "state":
                structured_obs_config[key] = StateObsConfig(dim=meta["dim"])

        # Use the existing vision encoder builder
        image_encoders_dict = build_vision_encoders(
            obs_config=structured_obs_config,
            encoder_cfg=self.network_cfg,  # type: ignore
            output_dim=self.network_cfg.image_fc_dim,
        )
        if not image_encoders_dict:
            raise ValueError("No vision encoders were built.")

        self.image_encoders = image_encoders_dict.to(device)
        image_feature_dim = len(self.image_encoders) * self.network_cfg.image_fc_dim

        # Fusion layer
        total_dim = (
            network_cfg.state_encoder_dims[-1]
            + network_cfg.expert_action_encoder_dims[-1]
            + image_feature_dim
            + extra_obs_feature_dim
        )
        fusion_layers = []
        prev_dim = total_dim
        for dim in network_cfg.fusion_dims:
            fusion_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
            ])
            prev_dim = dim
        self.fusion = nn.Sequential(*fusion_layers).to(device)

        self.output_dim = network_cfg.fusion_dims[-1]

        # Apply orthogonal initialization
        self.apply(_init_weights)

    def log_architecture(self) -> None:
        """Log a concise summary of the multimodal encoder architecture."""
        state_dims = " → ".join(map(str, self.network_cfg.state_encoder_dims))
        action_dims = " → ".join(map(str, self.network_cfg.expert_action_encoder_dims))
        fusion_dims = " → ".join(map(str, self.network_cfg.fusion_dims))

        extra_obs_info = []
        hidden_dim = getattr(self.network_cfg, "extra_obs_encoder_dim", 32)
        output_dim = getattr(self.network_cfg, "extra_obs_output_dim", 16)
        for key in self.extra_obs_encoders.keys():
            extra_obs_info.append(f"{key}: input_dim → {hidden_dim} → {output_dim}")

        logger.info("MultimodalEncoder Architecture:")
        logger.info(
            f"  State Encoder: "
            f"{self.shape_meta['obs']['state']['dim']} → {state_dims} (LayerNorm)"
        )
        logger.info(
            f"  Action Encoder: "
            f"{self.shape_meta['action']['shape'][0]} → {action_dims} (LayerNorm)"
        )
        if self.image_encoders:
            logger.info(
                f"  Image Encoders: {list(self.image_encoders.keys())} (from vision_encoders)"
            )
        for extra_info in extra_obs_info:
            logger.info(f"  Extra Obs Encoder - {extra_info} (LayerNorm)")
        logger.info(f"  Fusion: {fusion_dims} → {self.output_dim} (LayerNorm)")

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through multimodal encoder."""
        features = []

        # Encode state (handle history properly)
        state = obs["state"]  # [B, history_len, state_dim]
        batch_size, history_len, state_dim = state.shape
        state_reshaped = state.view(-1, state_dim)
        state_encoded = self.state_encoder(state_reshaped)
        state_encoded = state_encoded.view(batch_size, history_len, -1)
        state_feat = state_encoded.mean(dim=1)  # Pool over history
        features.append(state_feat)

        # Encode expert action
        expert_action_feat = self.expert_action_encoder(obs["expert_action"])
        features.append(expert_action_feat)

        # Encode extra observations
        for key, encoder in self.extra_obs_encoders.items():
            if key in obs:
                extra_obs_feat = encoder(obs[key])
                features.append(extra_obs_feat)

        # Encode images
        for key, encoder in self.image_encoders.items():
            if key in obs:
                img = obs[key].squeeze(1)  # Remove extra batch dim from dataset
                img_feat = encoder(img)
                features.append(img_feat)

        # Fuse all features
        if not features:
            raise ValueError("No features were encoded, cannot fuse.")
        combined = torch.cat(features, dim=1)
        return self.fusion(combined)


class DeltaSACPolicyNetwork(nn.Module):
    """SAC policy network that outputs delta actions to add to expert actions."""

    def __init__(
        self,
        shape_meta: Dict[str, Any],
        network_cfg: NetworkConfig,
        device: torch.device,
        encoder: "MultimodalEncoder",
        delta_max: float = 0.05,
    ):
        super().__init__()

        self.action_dim = shape_meta["action"]["shape"][0]
        self.device = device
        self.delta_max = delta_max

        # Multimodal encoder
        self.encoder = encoder

        # Policy head that outputs both mean and log_std
        policy_layers = []
        prev_dim = self.encoder.output_dim
        for dim in network_cfg.policy_head_dims:
            policy_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
            ])
            prev_dim = dim
        self.policy_head = nn.Sequential(*policy_layers).to(device)

        # Separate layers for mean and log_std (SAC standard)
        self.mean_layer = nn.Linear(prev_dim, self.action_dim).to(device)
        self.log_std_layer = nn.Linear(prev_dim, self.action_dim).to(device)

        # Initialize policy head
        self.policy_head.apply(_init_weights)
        self.mean_layer.apply(_init_weights)
        self.log_std_layer.apply(_init_weights)

        # Initialize last layers with smaller weights for stability
        with torch.no_grad():
            mean_weight = self.mean_layer.weight
            log_std_weight = self.log_std_layer.weight
            mean_weight.data.mul_(0.01)
            log_std_weight.data.mul_(0.01)

        # Log architecture summary
        self._log_architecture()

    def _log_architecture(self) -> None:
        """Log a concise summary of the delta SAC policy network architecture."""
        policy_dims_list = self.encoder.network_cfg.policy_head_dims
        policy_dims_str = " → ".join(map(str, policy_dims_list))
        logger.info("DeltaSACPolicyNetwork Architecture:")
        logger.info(
            f"  Policy Head: {self.encoder.output_dim} → {policy_dims_str} (LayerNorm)"
        )
        logger.info(
            f"  Mean/LogStd: "
            f"{policy_dims_list[-1] if policy_dims_list else self.encoder.output_dim} → "
            f"{self.action_dim}"
        )
        logger.info(f"  Delta Action Range: ±{self.delta_max}")

    def forward(
        self, obs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning delta action mean and log_std."""
        features = self.encoder(obs)
        policy_features = self.policy_head(features)

        mean = self.mean_layer(policy_features)
        log_std = self.log_std_layer(policy_features)

        # Bound the policy mean with tanh squashing for numerical stability
        mean = torch.tanh(mean) * self.delta_max

        # Clamp log_std for numerical stability
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

        return mean, log_std

    def sample(
        self, obs: Dict[str, torch.Tensor], deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample delta action with log probability for SAC updates."""
        mean, log_std = self.forward(obs)
        std = torch.exp(log_std)

        if deterministic:
            # Use mean action for evaluation
            delta_action = mean
            log_prob = torch.zeros(mean.shape[0], device=self.device)
        else:
            # Sample from Gaussian distribution
            dist = torch.distributions.Normal(mean, std)
            pre_tanh_action = dist.rsample(sample_shape=torch.Size())

            # Compute log probability before tanh squashing
            log_prob = dist.log_prob(pre_tanh_action).sum(dim=-1)

            # Apply tanh squashing
            delta_action = torch.tanh(pre_tanh_action) * self.delta_max

            # Add tanh correction to log probability
            log_prob -= torch.log(
                1 - (delta_action / self.delta_max).pow(2) + 1e-6
            ).sum(dim=-1)

        return delta_action, log_prob

    def get_action(
        self, obs: Dict[str, torch.Tensor], deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get final action by adding delta to expert action."""
        with torch.no_grad():
            delta_action, log_prob = self.sample(obs, deterministic)
            # Add delta to expert action
            final_action = obs["expert_action"] + delta_action
        return final_action, log_prob


class SACQNetwork(nn.Module):
    """Q-network for SAC critic that handles multimodal observations."""

    def __init__(
        self,
        shape_meta: Dict[str, Any],
        network_cfg: NetworkConfig,
        device: torch.device,
        encoder: "MultimodalEncoder",
    ):
        super().__init__()

        self.action_dim = shape_meta["action"]["shape"][0]
        self.device = device

        # Multimodal encoder for observations
        self.encoder = encoder

        # Q-network head: takes encoded observations + delta actions
        q_layers = []
        input_dim = self.encoder.output_dim + self.action_dim
        prev_dim = input_dim

        # Use value head dimensions for Q-network
        for dim in network_cfg.value_head_dims:
            q_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
            ])
            prev_dim = dim

        # Output single Q-value
        q_layers.append(nn.Linear(prev_dim, 1))
        self.q_head = nn.Sequential(*q_layers).to(device)

        # Initialize Q-network
        self.q_head.apply(_init_weights)

        # Initialize last layer with smaller weights for stability
        last_layer = self.q_head[-1]
        if isinstance(last_layer, nn.Linear):
            nn.init.orthogonal_(last_layer.weight, gain=1)  # Use gain=1 for stability
            # Scale weights down for stability
            with torch.no_grad():
                last_layer.weight.data *= 0.01
                # Initialize bias to small negative value to prevent
                # initial Q-value explosion
                if last_layer.bias is not None:
                    last_layer.bias.data.fill_(-0.1)

    def forward(
        self, obs: Dict[str, torch.Tensor], action: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass returning Q-values."""
        obs_features = self.encoder(obs)
        # Concatenate observation features with delta actions
        q_input = torch.cat([obs_features, action], dim=-1)
        return self.q_head(q_input).squeeze(-1)


class SACDeltaAction:
    """SAC algorithm for delta action learning with multi-modal observations."""

    def __init__(
        self,
        shape_meta: Dict[str, Any],
        rl_cfg: RLConfig,
        device: torch.device,
    ):
        """Initialize SAC agent for delta action learning."""
        self.device = device
        self.shape_meta = shape_meta
        self.action_dim = shape_meta["action"]["shape"][0]
        self.sac_cfg = rl_cfg.sac

        # Delta action range configuration
        self.delta_max = self.sac_cfg.delta_action_max_range

        # SAC hyperparameters
        self.gamma = self.sac_cfg.gamma
        self.polyak = self.sac_cfg.polyak

        # Automatic temperature tuning (optimized for delta actions)
        if self.sac_cfg.target_entropy is not None:
            self.target_entropy = float(self.sac_cfg.target_entropy)
        else:
            # For delta actions, use smaller entropy target based on configured range
            self.target_entropy = -0.1 * float(self.action_dim)

        # Delta penalty scheduling
        self.current_epoch = 0
        self.delta_penalty_schedule = self.sac_cfg.delta_penalty_schedule

        # Create shared encoders to avoid multiple DINOv2 loadings
        self.encoder_main = MultimodalEncoder(shape_meta, rl_cfg.network, device)
        self.encoder_main.log_architecture()  # Log architecture once

        self.encoder_target = copy.deepcopy(self.encoder_main)

        self.log_alpha = torch.zeros(1, device=device, requires_grad=True)
        self.alpha_optimizer = optim.Adam(
            [self.log_alpha], lr=self.sac_cfg.alpha_learning_rate
        )

        # Networks - pass delta_max to policy network
        self.policy = DeltaSACPolicyNetwork(
            shape_meta, rl_cfg.network, device, self.encoder_main, self.delta_max
        )
        self.q1 = SACQNetwork(
            shape_meta, rl_cfg.network, device, encoder=self.encoder_main
        )
        self.q2 = SACQNetwork(
            shape_meta, rl_cfg.network, device, encoder=self.encoder_main
        )
        self.q1_target = SACQNetwork(
            shape_meta, rl_cfg.network, device, encoder=self.encoder_target
        )
        self.q2_target = SACQNetwork(
            shape_meta, rl_cfg.network, device, encoder=self.encoder_target
        )

        # Parameters are already copied by deepcopy, so this is not needed
        # self.q1_target.load_state_dict(self.q1.state_dict())

        # Optimizers
        self.policy_optimizer = optim.Adam(
            self.policy.parameters(), lr=self.sac_cfg.policy_learning_rate
        )
        self.q1_optimizer = optim.Adam(
            self.q1.parameters(), lr=self.sac_cfg.critic_learning_rate
        )
        self.q2_optimizer = optim.Adam(
            self.q2.parameters(), lr=self.sac_cfg.critic_learning_rate
        )

        # Value normalization buffers (Milestone 1)
        if self.sac_cfg.use_value_norm:
            self.running_q_mean = torch.zeros(1, device=device)
            self.running_q_var = torch.ones(1, device=device)
        else:
            self.running_q_mean = None
            self.running_q_var = None

        # Adaptive CQL weight (Milestone 3)
        if self.sac_cfg.cql_use_lagrange:
            self.cql_lambda = self.sac_cfg.cql_weight
        else:
            self.cql_lambda = None

        self.rng = np.random.default_rng()

        # AMP scaler for mixed precision training
        self.use_amp = (
            getattr(rl_cfg.training, "use_amp", False) and device.type == "cuda"
        )
        if self.use_amp and GradScaler is not None:
            # Use the appropriate GradScaler constructor
            self.scaler = GradScaler()
            logger.info("✅ Automatic Mixed Precision (AMP) enabled")
        else:
            self.scaler = None
            if getattr(rl_cfg.training, "use_amp", False):
                logger.warning(
                    "⚠️ AMP requested but not available (CPU or missing CUDA)"
                )

        # Store device type for autocast
        self.device_type = "cuda" if device.type == "cuda" else "cpu"

        # Optional torch.compile optimization
        if getattr(rl_cfg.training, "torch_compile", False):
            try:
                self.policy = torch.compile(self.policy)
                self.q1 = torch.compile(self.q1)
                self.q2 = torch.compile(self.q2)
                logger.info("✅ torch.compile optimization enabled")
            except Exception as e:
                logger.warning(f"⚠️ torch.compile failed: {e}")

        # Efficient CQL setting
        self.efficient_cql = getattr(self.sac_cfg, "efficient_cql", True)

    def _autocast_context(self):
        """Create autocast context with version compatibility."""
        if not self.use_amp or autocast is None or self.device_type != "cuda":
            # Return a dummy context manager that does nothing but preserves gradients
            return contextlib.nullcontext()

        # When AMP is enabled on CUDA, use autocast
        return autocast(device_type=self.device_type, enabled=True)

    def _check_for_nan(self, tensor: torch.Tensor, name: str) -> bool:
        """Check tensor for NaN values and log if found."""
        if torch.isnan(tensor).any():
            logger.warning(f"NaN detected in {name}, skipping update")
            return True
        return False

    @property
    def alpha(self) -> float:
        """Current temperature parameter."""
        return self.log_alpha.exp().item()

    def _update_running_q_stats(self, targets: torch.Tensor) -> None:
        """Update running statistics for value normalization."""
        if self.running_q_mean is None or self.running_q_var is None:
            return

        with torch.no_grad():
            batch_mean = targets.mean()
            batch_var = (targets - batch_mean).pow(2).mean()
            self.running_q_mean.mul_(0.99).add_(batch_mean * 0.01)
            self.running_q_var.mul_(0.99).add_(batch_var * 0.01)

    def _normalize_q_values(self, q_values: torch.Tensor) -> torch.Tensor:
        """Normalize Q-values using running statistics."""
        if self.running_q_mean is None or self.running_q_var is None:
            return q_values

        std = torch.sqrt(self.running_q_var + 1e-6).clamp_min(0.5)
        return (q_values - self.running_q_mean) / std

    def _compute_improved_cql_loss(
        self,
        q1_pred: torch.Tensor,
        q2_pred: torch.Tensor,
        obs: Dict[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute improved CQL loss with efficient batched computation."""
        if self.efficient_cql:
            # Efficient version: compute all actions in one forward pass
            batch_size = actions.shape[0]

            # Sample OOD actions
            random_actions = torch.randn_like(actions) * self.delta_max
            policy_actions, _ = self.policy.sample(obs, deterministic=False)

            # Batch all actions together: [data_actions, random_actions, policy_actions]
            all_actions = torch.cat([actions, random_actions, policy_actions], dim=0)

            # Repeat observations to match action batch size
            obs_repeated = {}
            for key, tensor in obs.items():
                obs_repeated[key] = tensor.repeat(3, *([1] * (tensor.dim() - 1)))

            # Single forward pass for all actions
            q1_all = self.q1(obs_repeated, all_actions)
            q2_all = self.q2(obs_repeated, all_actions)

            # Split results
            q1_data, q1_rand, q1_policy = torch.chunk(q1_all, 3, dim=0)
            q2_data, q2_rand, q2_policy = torch.chunk(q2_all, 3, dim=0)

            # Normalize OOD Q-values if using value normalization
            if self.sac_cfg.use_value_norm:
                q1_rand = self._normalize_q_values(q1_rand)
                q2_rand = self._normalize_q_values(q2_rand)
                q1_policy = self._normalize_q_values(q1_policy)
                q2_policy = self._normalize_q_values(q2_policy)

            # Log-sum-exp over OOD actions
            cat_q1 = torch.cat([q1_rand, q1_policy], dim=0)
            cat_q2 = torch.cat([q2_rand, q2_policy], dim=0)

        else:
            # Original version: separate forward passes
            random_actions = torch.randn_like(actions) * self.delta_max
            policy_actions, _ = self.policy.sample(obs, deterministic=False)

            # Get Q-values for OOD actions
            q1_rand = self.q1(obs, random_actions)
            q2_rand = self.q2(obs, random_actions)
            q1_policy = self.q1(obs, policy_actions)
            q2_policy = self.q2(obs, policy_actions)

            # Normalize OOD Q-values if using value normalization
            if self.sac_cfg.use_value_norm:
                q1_rand = self._normalize_q_values(q1_rand)
                q2_rand = self._normalize_q_values(q2_rand)
                q1_policy = self._normalize_q_values(q1_policy)
                q2_policy = self._normalize_q_values(q2_policy)

            # Log-sum-exp over OOD actions
            cat_q1 = torch.cat([q1_rand, q1_policy], dim=0)
            cat_q2 = torch.cat([q2_rand, q2_policy], dim=0)

        logsumexp_q1 = (
            torch.logsumexp(cat_q1 / self.sac_cfg.cql_temp, dim=0)
            * self.sac_cfg.cql_temp
        )
        logsumexp_q2 = (
            torch.logsumexp(cat_q2 / self.sac_cfg.cql_temp, dim=0)
            * self.sac_cfg.cql_temp
        )

        # CQL loss: E[log-sum-exp(Q_OOD)] - E[Q_data]
        cql_loss_q1 = logsumexp_q1 - q1_pred.mean()
        cql_loss_q2 = logsumexp_q2 - q2_pred.mean()

        cql_loss = cql_loss_q1 + cql_loss_q2

        # Apply hinge loss if enabled
        if self.sac_cfg.cql_hinge:
            cql_loss = torch.relu(cql_loss)

        return cql_loss

    def _backward_and_step(
        self, loss: torch.Tensor, optimizer: optim.Optimizer, retain_graph: bool = False
    ) -> float:
        """AMP-aware backward pass and optimizer step."""
        if self.use_amp and self.scaler is not None:
            self.scaler.scale(loss).backward(retain_graph=retain_graph)
            self.scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for group in optimizer.param_groups for p in group["params"]],
                self.sac_cfg.grad_clip_norm,
            )
            self.scaler.step(optimizer)
            if not retain_graph:  # Only update scaler on the last backward pass
                self.scaler.update()
        else:
            loss.backward(retain_graph=retain_graph)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for group in optimizer.param_groups for p in group["params"]],
                self.sac_cfg.grad_clip_norm,
            )
            optimizer.step()

        return float(grad_norm)

    def get_current_delta_penalty(self) -> float:
        """Get current delta penalty coefficient with optional scheduling."""
        if not self.delta_penalty_schedule:
            return self.sac_cfg.delta_penalty_coeff

        # Linear schedule from 0 to max over specified epochs
        if self.current_epoch >= self.sac_cfg.delta_penalty_epochs:
            return self.sac_cfg.delta_penalty_max

        progress = self.current_epoch / self.sac_cfg.delta_penalty_epochs
        return progress * self.sac_cfg.delta_penalty_max

    def _move_obs_to_device(
        self, obs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Move observation tensors to the correct device."""
        return {key: tensor.to(self.device) for key, tensor in obs.items()}

    def collect_batch_from_dataset(
        self, dataloader: DataLoader, num_steps: int
    ) -> Dict:
        """Collect a batch of experiences from the dataset."""
        all_obs = {}
        all_next_obs = {}
        all_actions = []
        all_rewards = []
        all_dones = []
        all_terminated = []

        steps_collected = 0
        for batch in dataloader:
            if steps_collected >= num_steps:
                break

            # Sanity check for NaNs in dataset
            if torch.isnan(batch["action"]).any():
                logger.error("NaN found in dataset actions")
                continue
            if torch.isnan(batch["reward"]).any():
                logger.error("NaN found in dataset rewards")
                continue

            # Move to device
            obs = self._move_obs_to_device(batch["obs"])
            next_obs = self._move_obs_to_device(batch["next_obs"])

            # Accumulate batch data efficiently
            if not all_obs:  # Initialize observation dictionaries
                for key in obs.keys():
                    all_obs[key] = []
                for key in next_obs.keys():
                    all_next_obs[key] = []

            # Store tensors directly
            for key in obs.keys():
                all_obs[key].append(obs[key])
            for key in next_obs.keys():
                all_next_obs[key].append(next_obs[key])

            all_actions.append(batch["action"].to(self.device))
            all_rewards.append(batch["reward"].to(self.device))
            all_dones.append(batch["done"].to(self.device))
            all_terminated.append(batch["terminated"].to(self.device))

            steps_collected += len(batch["reward"])

        # Concatenate all collected data
        obs_final = {key: torch.cat(tensors, dim=0) for key, tensors in all_obs.items()}
        next_obs_final = {
            key: torch.cat(tensors, dim=0) for key, tensors in all_next_obs.items()
        }

        return {
            "obs": obs_final,
            "next_obs": next_obs_final,
            "actions": torch.cat(all_actions, dim=0),
            "rewards": torch.cat(all_rewards, dim=0),
            "dones": torch.cat(all_dones, dim=0),
            "terminated": torch.cat(all_terminated, dim=0),
        }

    def update_on_the_fly(self, dataloader: DataLoader) -> Dict[str, float]:
        """On-the-fly training that performs gradient_updates_per_epoch updates per call.

        This method implements proper UTD (Update-to-Data) ratio control by performing
        exactly sac_cfg.gradient_updates_per_epoch gradient updates, each on a mini-batch
        of size sac_cfg.mini_batch_size.
        """
        # Training metrics
        policy_losses = []
        q1_losses = []
        q2_losses = []
        alpha_losses = []
        q_values = []
        bc_losses = []
        action_reg_losses = []
        anchor_losses = []
        cql_losses = []

        updates_completed = 0
        dataloader_iter = iter(dataloader)

        # Perform exactly gradient_updates_per_epoch gradient updates
        while updates_completed < self.sac_cfg.gradient_updates_per_epoch:
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(dataloader)  # Reset iterator
                batch = next(dataloader_iter)

            # Use only mini_batch_size samples from the loaded batch
            actual_batch_size = min(len(batch["action"]), self.sac_cfg.mini_batch_size)
            if actual_batch_size < self.sac_cfg.mini_batch_size:
                # If we don't have enough samples, use what we have
                indices = torch.arange(actual_batch_size)
            else:
                # Randomly sample mini_batch_size samples from the loaded batch
                indices = torch.randperm(len(batch["action"]))[
                    : self.sac_cfg.mini_batch_size
                ]

            # Extract mini-batch
            mini_batch = {
                "obs": {key: tensor[indices] for key, tensor in batch["obs"].items()},
                "next_obs": {
                    key: tensor[indices] for key, tensor in batch["next_obs"].items()
                },
                "action": batch["action"][indices],
                "reward": batch["reward"][indices],
                "done": batch["done"][indices],
                "terminated": batch["terminated"][indices],
            }

            # Sanity check for NaNs
            if (
                torch.isnan(mini_batch["action"]).any()
                or torch.isnan(mini_batch["reward"]).any()
            ):
                continue

            # Move to device with non_blocking for performance
            obs = {
                key: tensor.to(self.device, non_blocking=True)
                for key, tensor in mini_batch["obs"].items()
            }
            next_obs = {
                key: tensor.to(self.device, non_blocking=True)
                for key, tensor in mini_batch["next_obs"].items()
            }
            actions = mini_batch["action"].to(self.device, non_blocking=True)
            rewards = mini_batch["reward"].to(self.device, non_blocking=True)
            terminated = mini_batch["terminated"].to(self.device, non_blocking=True)

            # Normalize rewards
            rewards_raw = rewards
            rewards = (rewards_raw - rewards_raw.mean()) / (rewards_raw.std() + 1e-8)

            # === Critic Update with AMP ===
            with self._autocast_context():
                # Target Q computation
                with torch.no_grad():
                    next_delta_actions, next_log_probs = self.policy.sample(next_obs)
                    q1_next = self.q1_target(next_obs, next_delta_actions)
                    q2_next = self.q2_target(next_obs, next_delta_actions)
                    min_q_next = torch.min(q1_next, q2_next)
                    soft_value_next = min_q_next - self.alpha * next_log_probs
                    soft_value_next *= 1 - terminated.float()
                    target_q = rewards + self.gamma * soft_value_next

                # Update running Q statistics
                self._update_running_q_stats(target_q)

                # Current Q-values
                current_q1 = self.q1(obs, actions)
                current_q2 = self.q2(obs, actions)

                # Apply value normalization if enabled
                if self.sac_cfg.use_value_norm:
                    target_q_norm = self._normalize_q_values(target_q)
                    current_q1_norm = self._normalize_q_values(current_q1)
                    current_q2_norm = self._normalize_q_values(current_q2)
                else:
                    target_q_norm = target_q
                    current_q1_norm = current_q1
                    current_q2_norm = current_q2

                # Standard Q-network losses
                q1_loss_mse = F.mse_loss(current_q1_norm, target_q_norm)
                q2_loss_mse = F.mse_loss(current_q2_norm, target_q_norm)

                # Check for NaN in losses
                if self._check_for_nan(
                    q1_loss_mse, "q1_loss_mse"
                ) or self._check_for_nan(q2_loss_mse, "q2_loss_mse"):
                    continue

                # CQL regularization
                cql_loss = self._compute_improved_cql_loss(
                    current_q1_norm, current_q2_norm, obs, actions
                )

                # Check for NaN in CQL loss
                if self._check_for_nan(cql_loss, "cql_loss"):
                    continue

                # Determine CQL weight
                if self.cql_lambda is not None:
                    self.cql_lambda = max(
                        0.0,
                        self.cql_lambda
                        + self.sac_cfg.cql_lambda_lr
                        * (cql_loss.detach().item() - self.sac_cfg.cql_target),
                    )
                    effective_cql_weight = self.cql_lambda
                else:
                    effective_cql_weight = self.sac_cfg.cql_weight

                # Anchor loss
                if self.sac_cfg.anchor_weight > 0:
                    batch_reward_mean = rewards.mean()
                    q_batch_mean = (current_q1.mean() + current_q2.mean()) / 2
                    anchor_loss = self.sac_cfg.anchor_weight * F.mse_loss(
                        q_batch_mean, batch_reward_mean
                    )
                else:
                    anchor_loss = torch.tensor(0.0, device=self.device)

                # Total Q-losses
                q1_loss = q1_loss_mse + effective_cql_weight * cql_loss + anchor_loss
                q2_loss = q2_loss_mse + effective_cql_weight * cql_loss + anchor_loss

                # Final NaN check before backward pass
                if self._check_for_nan(q1_loss, "q1_loss") or self._check_for_nan(
                    q2_loss, "q2_loss"
                ):
                    continue

            # Update Q-networks with AMP
            # Zero gradients for both critics
            self.q1_optimizer.zero_grad()
            self.q2_optimizer.zero_grad()

            # -------------------------------------------------------------
            # IMPORTANT: Backpropagate through *both* critic losses before
            #            performing any optimizer step.  Stepping one
            #            optimizer while the other graph is still needed
            #            would modify parameters in-place and invalidate the
            #            autograd graph, leading to the infamous
            #            "one of the variables needed for gradient computation
            #            has been modified by an inplace operation" error.
            # -------------------------------------------------------------

            total_critic_loss = q1_loss + q2_loss

            if self.use_amp and self.scaler is not None:
                # AMP backward
                self.scaler.scale(total_critic_loss).backward()

                # Unscale for gradient clipping
                self.scaler.unscale_(self.q1_optimizer)
                self.scaler.unscale_(self.q2_optimizer)

                # Clip gradients for numerical stability
                torch.nn.utils.clip_grad_norm_(
                    self.q1.parameters(), self.sac_cfg.grad_clip_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    self.q2.parameters(), self.sac_cfg.grad_clip_norm
                )

                # Step optimizers *after* both backward passes
                self.scaler.step(self.q1_optimizer)
                self.scaler.step(self.q2_optimizer)

                # Update scaler once per iteration
                self.scaler.update()
            else:
                # Standard FP32 training path
                total_critic_loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.q1.parameters(), self.sac_cfg.grad_clip_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    self.q2.parameters(), self.sac_cfg.grad_clip_norm
                )

                self.q1_optimizer.step()
                self.q2_optimizer.step()

            # === Actor Update with AMP ===
            with self._autocast_context():
                delta_actions, log_probs = self.policy.sample(obs)
                q1_vals = self.q1(obs, delta_actions)
                q2_vals = self.q2(obs, delta_actions)
                min_q_vals = torch.min(q1_vals, q2_vals)

                if self.sac_cfg.use_value_norm:
                    min_q_vals = self._normalize_q_values(min_q_vals)

                # Standard SAC actor loss
                actor_loss_sac = (self.alpha * log_probs - min_q_vals).mean()

                # Additional losses
                bc_loss = (
                    self.sac_cfg.bc_weight * F.mse_loss(delta_actions, actions)
                    if self.sac_cfg.bc_weight > 0
                    else torch.tensor(0.0, device=self.device)
                )
                action_reg_loss = (
                    self.sac_cfg.action_reg_weight * (delta_actions**2).mean()
                    if self.sac_cfg.action_reg_weight > 0
                    else torch.tensor(0.0, device=self.device)
                )

                current_delta_penalty = self.get_current_delta_penalty()
                delta_penalty = current_delta_penalty * (delta_actions**2).mean()

                total_actor_loss = (
                    actor_loss_sac + bc_loss + action_reg_loss + delta_penalty
                )

            # Update policy with AMP
            self.policy_optimizer.zero_grad()
            # Actor update does not share graph with other losses, so it's
            # safe to reuse the generic helper that properly handles AMP.
            self._backward_and_step(total_actor_loss, self.policy_optimizer)

            # === Temperature Update ===
            if self.sac_cfg.auto_temp:
                alpha_loss = -(
                    self.log_alpha * (log_probs + float(self.target_entropy)).detach()
                ).mean()
                self.alpha_optimizer.zero_grad()
                if self.use_amp and self.scaler is not None:
                    self.scaler.scale(alpha_loss).backward()
                    self.scaler.step(self.alpha_optimizer)
                else:
                    alpha_loss.backward()
                    self.alpha_optimizer.step()
                alpha_losses.append(alpha_loss.item())

            # === Target Network Updates ===
            with torch.no_grad():
                for param, target_param in zip(
                    self.q1.parameters(), self.q1_target.parameters()
                ):
                    target_param.data.copy_(
                        (1 - self.polyak) * target_param.data + self.polyak * param.data
                    )
                for param, target_param in zip(
                    self.q2.parameters(), self.q2_target.parameters()
                ):
                    target_param.data.copy_(
                        (1 - self.polyak) * target_param.data + self.polyak * param.data
                    )

            # Store metrics
            policy_losses.append(actor_loss_sac.item())
            q1_losses.append(q1_loss.item())
            q2_losses.append(q2_loss.item())
            q_values.append(torch.mean(torch.min(current_q1, current_q2)).item())
            bc_losses.append(bc_loss.item())
            action_reg_losses.append(action_reg_loss.item())
            anchor_losses.append(anchor_loss.item())
            cql_losses.append(cql_loss.item())

            updates_completed += 1

        # Return aggregated metrics
        metrics = {
            "policy_loss": float(np.mean(policy_losses)),
            "q1_loss": float(np.mean(q1_losses)),
            "q2_loss": float(np.mean(q2_losses)),
            "mean_q_value": float(np.mean(q_values)),
            "alpha": self.alpha,
            "mean_reward": float(rewards_raw.mean().item()),
            "bc_loss": float(np.mean(bc_losses)),
            "action_reg_loss": float(np.mean(action_reg_losses)),
            "anchor_loss": float(np.mean(anchor_losses)),
            "cql_loss": float(np.mean(cql_losses)),
            "policy_action_std": float(delta_actions.std().item()),
        }

        if alpha_losses:
            metrics["alpha_loss"] = float(np.mean(alpha_losses))

        if self.cql_lambda is not None:
            metrics["cql_lambda"] = float(self.cql_lambda)

        return metrics

    def update(self, batch_data: Dict) -> Dict[str, float]:
        """Update SAC networks using collected batch data with modern improvements."""
        all_obs = batch_data["obs"]
        all_next_obs = batch_data["next_obs"]
        all_actions = batch_data["actions"]  # These are delta actions from dataset
        all_rewards = batch_data["rewards"]
        all_terminated = batch_data["terminated"]

        # Normalize rewards (zero mean, unit std per batch)
        rewards_raw = all_rewards
        all_rewards = (rewards_raw - rewards_raw.mean()) / (rewards_raw.std() + 1e-8)

        # Training metrics
        policy_losses = []
        q1_losses = []
        q2_losses = []
        alpha_losses = []
        q_values = []
        bc_losses = []
        action_reg_losses = []
        anchor_losses = []
        cql_losses = []

        # Training with mini-batches from the collected data
        batch_size = len(all_actions)
        indices = np.arange(batch_size)

        for epoch in range(self.sac_cfg.gradient_updates_per_epoch):
            self.rng.shuffle(indices)

            for start in range(0, batch_size, self.sac_cfg.mini_batch_size):
                end = min(start + self.sac_cfg.mini_batch_size, batch_size)
                mb_idx = indices[start:end]

                # Mini-batch data
                mb_obs = {key: tensor[mb_idx] for key, tensor in all_obs.items()}
                mb_next_obs = {
                    key: tensor[mb_idx] for key, tensor in all_next_obs.items()
                }
                mb_actions = all_actions[mb_idx]  # Delta actions
                mb_rewards = all_rewards[mb_idx]
                mb_terminated = all_terminated[mb_idx]

                # === Critic Update ===
                with torch.no_grad():
                    # Sample next actions from current policy
                    next_delta_actions, next_log_probs = self.policy.sample(mb_next_obs)

                    # Target Q-values
                    q1_next = self.q1_target(mb_next_obs, next_delta_actions)
                    q2_next = self.q2_target(mb_next_obs, next_delta_actions)
                    min_q_next = torch.min(q1_next, q2_next)

                    # Soft value with entropy regularization
                    soft_value_next = min_q_next - self.alpha * next_log_probs

                    # Zero out value for terminal states
                    soft_value_next *= 1 - mb_terminated.float()

                    # Target Q-value
                    target_q = mb_rewards + self.gamma * soft_value_next

                # Update running Q statistics for value normalization
                self._update_running_q_stats(target_q)

                # Current Q-values for taken delta actions
                current_q1 = self.q1(mb_obs, mb_actions)
                current_q2 = self.q2(mb_obs, mb_actions)

                # Apply value normalization if enabled
                if self.sac_cfg.use_value_norm:
                    target_q_norm = self._normalize_q_values(target_q)
                    current_q1_norm = self._normalize_q_values(current_q1)
                    current_q2_norm = self._normalize_q_values(current_q2)
                else:
                    target_q_norm = target_q
                    current_q1_norm = current_q1
                    current_q2_norm = current_q2

                # Standard Q-network losses (MSE)
                q1_loss_mse = F.mse_loss(current_q1_norm, target_q_norm)
                q2_loss_mse = F.mse_loss(current_q2_norm, target_q_norm)

                # Check for NaN in losses
                if self._check_for_nan(
                    q1_loss_mse, "q1_loss_mse"
                ) or self._check_for_nan(q2_loss_mse, "q2_loss_mse"):
                    continue

                # CQL regularization
                cql_loss = self._compute_improved_cql_loss(
                    current_q1_norm, current_q2_norm, mb_obs, mb_actions
                )

                # Check for NaN in CQL loss
                if self._check_for_nan(cql_loss, "cql_loss"):
                    continue

                # Determine CQL weight (adaptive or static)
                if self.cql_lambda is not None:
                    # Adaptive CQL weight update
                    self.cql_lambda = max(
                        0.0,
                        self.cql_lambda
                        + self.sac_cfg.cql_lambda_lr
                        * (cql_loss.detach().item() - self.sac_cfg.cql_target),
                    )
                    effective_cql_weight = self.cql_lambda
                else:
                    effective_cql_weight = self.sac_cfg.cql_weight

                # Anchor loss (tie Q-values to batch reward mean)
                if self.sac_cfg.anchor_weight > 0:
                    batch_reward_mean = mb_rewards.mean()
                    q_batch_mean = (current_q1.mean() + current_q2.mean()) / 2
                    anchor_loss = self.sac_cfg.anchor_weight * F.mse_loss(
                        q_batch_mean, batch_reward_mean
                    )
                else:
                    anchor_loss = torch.tensor(0.0, device=self.device)

                # Total Q-losses
                q1_loss = q1_loss_mse + effective_cql_weight * cql_loss + anchor_loss
                q2_loss = q2_loss_mse + effective_cql_weight * cql_loss + anchor_loss

                # Final NaN check before backward pass
                if self._check_for_nan(q1_loss, "q1_loss") or self._check_for_nan(
                    q2_loss, "q2_loss"
                ):
                    continue

                # Update Q-networks
                self.q1_optimizer.zero_grad()
                self.q2_optimizer.zero_grad()

                q1_loss.backward(retain_graph=True)
                q2_loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.q1.parameters(), self.sac_cfg.grad_clip_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    self.q2.parameters(), self.sac_cfg.grad_clip_norm
                )

                self.q1_optimizer.step()
                self.q2_optimizer.step()

                # === Actor Update ===
                # Sample actions from current policy
                delta_actions, log_probs = self.policy.sample(mb_obs)
                q1_vals = self.q1(mb_obs, delta_actions)
                q2_vals = self.q2(mb_obs, delta_actions)
                min_q_vals = torch.min(q1_vals, q2_vals)

                # Apply value normalization to Q-values for actor loss
                if self.sac_cfg.use_value_norm:
                    min_q_vals = self._normalize_q_values(min_q_vals)

                # Standard SAC actor loss with entropy regularization
                actor_loss_sac = (self.alpha * log_probs - min_q_vals).mean()

                # Behavior cloning loss (regularize towards dataset actions)
                if self.sac_cfg.bc_weight > 0:
                    bc_loss = self.sac_cfg.bc_weight * F.mse_loss(
                        delta_actions, mb_actions
                    )
                else:
                    bc_loss = torch.tensor(0.0, device=self.device)

                # Action L2 regularization (prevent large delta actions)
                if self.sac_cfg.action_reg_weight > 0:
                    action_reg_loss = (
                        self.sac_cfg.action_reg_weight * (delta_actions**2).mean()
                    )
                else:
                    action_reg_loss = torch.tensor(0.0, device=self.device)

                # Scheduled delta magnitude penalty
                current_delta_penalty = self.get_current_delta_penalty()
                delta_penalty = current_delta_penalty * (delta_actions**2).mean()

                # Total actor loss
                total_actor_loss = (
                    actor_loss_sac + bc_loss + action_reg_loss + delta_penalty
                )

                # Update policy
                self.policy_optimizer.zero_grad()
                # Actor update does not share graph with other losses, so it's
                # safe to reuse the generic helper that properly handles AMP.
                self._backward_and_step(total_actor_loss, self.policy_optimizer)

                # === Temperature Update ===
                if self.sac_cfg.auto_temp:
                    alpha_loss = -(
                        self.log_alpha
                        * (log_probs + float(self.target_entropy)).detach()
                    ).mean()

                    self.alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optimizer.step()
                    alpha_losses.append(alpha_loss.item())

                # === Target Network Updates (Polyak averaging) ===
                with torch.no_grad():
                    for param, target_param in zip(
                        self.q1.parameters(), self.q1_target.parameters()
                    ):
                        target_param.data.copy_(
                            (1 - self.polyak) * target_param.data
                            + self.polyak * param.data
                        )
                    for param, target_param in zip(
                        self.q2.parameters(), self.q2_target.parameters()
                    ):
                        target_param.data.copy_(
                            (1 - self.polyak) * target_param.data
                            + self.polyak * param.data
                        )

                # Store metrics
                policy_losses.append(actor_loss_sac.item())
                q1_losses.append(q1_loss.item())
                q2_losses.append(q2_loss.item())
                q_values.append(torch.mean(torch.min(current_q1, current_q2)).item())
                bc_losses.append(bc_loss.item())
                action_reg_losses.append(action_reg_loss.item())
                anchor_losses.append(anchor_loss.item())
                cql_losses.append(cql_loss.item())

        # Return aggregated metrics
        metrics = {
            "policy_loss": float(np.mean(policy_losses)),
            "q1_loss": float(np.mean(q1_losses)),
            "q2_loss": float(np.mean(q2_losses)),
            "mean_q_value": float(np.mean(q_values)),
            "alpha": self.alpha,
            "mean_reward": float(rewards_raw.mean().item()),
            "bc_loss": float(np.mean(bc_losses)),
            "action_reg_loss": float(np.mean(action_reg_losses)),
            "anchor_loss": float(np.mean(anchor_losses)),
            "cql_loss": float(np.mean(cql_losses)),
            "policy_action_std": float(delta_actions.std().item()),
        }

        if alpha_losses:
            metrics["alpha_loss"] = float(np.mean(alpha_losses))

        if self.cql_lambda is not None:
            metrics["cql_lambda"] = float(self.cql_lambda)

        return metrics


def _create_output_dir(base_dir: str) -> Path:
    """Create a unique output directory using a timestamp-based model id.

    Args:
        base_dir: Base directory where all training runs are stored.

    Returns:
        The created directory path.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = f"model_{timestamp}"
    out_dir = Path(base_dir).expanduser().resolve() / model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Created output directory: %s", out_dir)
    return out_dir


def _save_checkpoint(
    agent: "SACDeltaAction",  # Forward reference
    epoch: int,
    metric_value: float,
    out_dir: Path,
    tag: str,
    raw_cfg: Dict[str, Any],
) -> None:
    """Save checkpoint with model/optimizer states and metadata.

    Args:
        agent: The SAC agent containing networks & optimizers.
        epoch: Current epoch number.
        metric_value: Metric value corresponding to this checkpoint.
        out_dir: Directory to save the checkpoint in.
        tag: Either "best" or "latest" (used in file name).
        raw_cfg: Original raw YAML config dict to embed in the checkpoint.
    """
    ckpt_path = out_dir / f"checkpoint_{tag}.pt"
    torch.save(
        {
            "epoch": epoch,
            "metric_value": metric_value,
            "encoder_main_state": agent.encoder_main.state_dict(),
            "encoder_target_state": agent.encoder_target.state_dict(),
            "policy_state": agent.policy.state_dict(),
            "q1_state": agent.q1.state_dict(),
            "q2_state": agent.q2.state_dict(),
            "q1_target_state": agent.q1_target.state_dict(),
            "q2_target_state": agent.q2_target.state_dict(),
            "policy_opt_state": agent.policy_optimizer.state_dict(),
            "q1_opt_state": agent.q1_optimizer.state_dict(),
            "q2_opt_state": agent.q2_optimizer.state_dict(),
            "alpha_opt_state": agent.alpha_optimizer.state_dict(),
            "log_alpha": agent.log_alpha.item(),
            "current_epoch": agent.current_epoch,
            "config": raw_cfg,
            "timestamp": datetime.now().isoformat(),
        },
        ckpt_path,
    )
    emoji = "🏆" if tag == "best" else "📄"
    logger.info(
        "%s Saved %s checkpoint to %s (metric=%.4f)",
        emoji,
        tag,
        ckpt_path,
        metric_value,
    )


def _load_checkpoint(
    agent: "SACDeltaAction",  # Forward reference
    checkpoint_path: str | Path,
    device: torch.device,
) -> Tuple[int, float]:
    """Load checkpoint and resume training state.

    Args:
        agent: The SAC agent to load states into.
        checkpoint_path: Path to the checkpoint file.
        device: Device to load the checkpoint on.

    Returns:
        Tuple of (starting_epoch, best_metric) for resuming training.
    """
    ckpt_path = Path(checkpoint_path).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    logger.info(f"🔄 Loading checkpoint from: {ckpt_path}")

    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Load network states
    agent.encoder_main.load_state_dict(checkpoint["encoder_main_state"])
    agent.encoder_target.load_state_dict(checkpoint["encoder_target_state"])
    agent.policy.load_state_dict(checkpoint["policy_state"])
    agent.q1.load_state_dict(checkpoint["q1_state"])
    agent.q2.load_state_dict(checkpoint["q2_state"])
    agent.q1_target.load_state_dict(checkpoint["q1_target_state"])
    agent.q2_target.load_state_dict(checkpoint["q2_target_state"])

    # Load optimizer states
    agent.policy_optimizer.load_state_dict(checkpoint["policy_opt_state"])
    agent.q1_optimizer.load_state_dict(checkpoint["q1_opt_state"])
    agent.q2_optimizer.load_state_dict(checkpoint["q2_opt_state"])
    agent.alpha_optimizer.load_state_dict(checkpoint["alpha_opt_state"])

    # Load scalar values
    agent.log_alpha.data.copy_(torch.tensor(checkpoint["log_alpha"], device=device))

    # Load training state
    starting_epoch = checkpoint.get("epoch", 0) + 1  # Resume from next epoch
    best_metric = checkpoint.get("metric_value", float("-inf"))
    current_epoch = checkpoint.get("current_epoch", starting_epoch - 1)
    agent.current_epoch = current_epoch

    logger.info("✅ Checkpoint loaded successfully")
    logger.info(f"  Resuming from epoch: {starting_epoch}")
    logger.info(f"  Best metric so far: {best_metric:.4f}")
    logger.info(f"  Current alpha: {agent.alpha:.4f}")

    return starting_epoch, best_metric


def train_sac_delta_action(
    *, rl_cfg: RLConfig, raw_cfg: Dict[str, Any], output_dir: Path
) -> None:
    """Train SAC agent for delta action learning with checkpointing.

    Args:
        rl_cfg: Structured RL configuration object (already validated).
        raw_cfg: Raw dictionary loaded from YAML for checkpoint embedding.
        output_dir: Directory where checkpoints & artifacts are stored.
    """
    # Enable anomaly detection for debugging
    torch.autograd.set_detect_anomaly(True)

    # Setup device
    device = torch.device(
        "cuda" if rl_cfg.training.use_gpu and torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Training SAC Delta Action using device: {device}")

    # Check for resume checkpoint
    resume_checkpoint = rl_cfg.training.resume_checkpoint_path
    is_resuming = resume_checkpoint is not None

    if is_resuming:
        logger.info(f"🔄 Resume mode enabled: will load from {resume_checkpoint}")
    else:
        logger.info("🆕 Training from scratch")

    # Log performance optimizations
    performance_features = []
    if getattr(rl_cfg.training, "use_amp", False) and device.type == "cuda":
        performance_features.append("AMP")
    if getattr(rl_cfg.training, "torch_compile", False):
        performance_features.append("torch.compile")
    if getattr(rl_cfg.training, "on_the_fly_sampling", True):
        performance_features.append("on-the-fly sampling")
    if getattr(rl_cfg.sac, "efficient_cql", True):
        performance_features.append("efficient CQL")

    if performance_features:
        logger.info(
            f"🚀 Performance optimizations enabled: {', '.join(performance_features)}"
        )
    else:
        logger.info("⚠️ No performance optimizations enabled")

    # Create dataset
    logger.info(f"Loading dataset... ({Path(rl_cfg.dataset.path).resolve()})")
    train_dataset = RLDataset(cfg_rl=rl_cfg, is_train_split=True)
    val_dataset = train_dataset.get_validation_dataset()

    logger.info(f"Training samples: {len(train_dataset)}")
    if val_dataset:
        logger.info(f"Validation samples: {len(val_dataset)}")

    # Create data loaders with optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=rl_cfg.training.dataloader_batch_size,
        shuffle=True,
        num_workers=rl_cfg.training.num_workers,
        pin_memory=getattr(rl_cfg.training, "pin_memory", True),
        prefetch_factor=getattr(rl_cfg.training, "prefetch_factor", 2),
        persistent_workers=rl_cfg.training.num_workers > 0,
    )

    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=rl_cfg.training.dataloader_batch_size,
            shuffle=False,
            num_workers=rl_cfg.training.num_workers,
            pin_memory=getattr(rl_cfg.training, "pin_memory", True),
            prefetch_factor=getattr(rl_cfg.training, "prefetch_factor", 2),
            persistent_workers=rl_cfg.training.num_workers > 0,
        )

    # Get shape metadata
    shape_meta = train_dataset.get_shape_meta()
    logger.info(f"Shape metadata: {shape_meta}")

    # Create SAC agent
    agent = SACDeltaAction(shape_meta, rl_cfg, device)

    # Resume from checkpoint if specified
    starting_epoch = 0
    best_metric = float("-inf")

    if is_resuming:
        try:
            _, last_best_metric = _load_checkpoint(agent, resume_checkpoint, device)
            logger.info("✅ Successfully resumed training")
            logger.info(f"  Last best metric: {last_best_metric:.4f}")
        except Exception as e:
            logger.error(f"❌ Failed to load checkpoint: {e}")
            logger.info("🆕 Falling back to training from scratch")

    # Initialize TensorBoard logger
    tb_logger = None
    if rl_cfg.training.enable_tb:
        tb_log_dir = output_dir / "tensorboard"
        tb_logger = SimpleTBLogger(str(tb_log_dir), enabled=True)

    # Verify data compatibility
    logger.info("Verifying dataset compatibility...")
    sample_batch = next(iter(train_loader))
    sample_obs = agent._move_obs_to_device(sample_batch["obs"])

    try:
        # Test forward passes
        with torch.no_grad():
            test_q1 = agent.q1(sample_obs, sample_batch["action"].to(device))
            test_q2 = agent.q2(sample_obs, sample_batch["action"].to(device))
            test_delta_action, test_log_prob = agent.policy.sample(sample_obs)
        logger.info("✓ Networks initialized successfully")
        logger.info(f"  Q1 output shape: {test_q1.shape}")
        logger.info(f"  Q2 output shape: {test_q2.shape}")
        logger.info(f"  Delta action output shape: {test_delta_action.shape}")
        logger.info(f"  Log prob shape: {test_log_prob.shape}")
        logger.info(f"  Current alpha: {agent.alpha:.4f}")
        logger.info(
            f"  Delta action range: "
            f"[{test_delta_action.min().item():.4f}, "
            f"{test_delta_action.max().item():.4f}]"
        )
    except Exception as e:
        logger.error(f"Network compatibility test failed: {e}")
        raise

    # Training loop
    if is_resuming:
        logger.info(
            f"Resuming training from the last checkpoint... {resume_checkpoint}"
        )
    else:
        logger.info("Starting training from scratch...")
    training_start_time = time.time()

    save_interval = rl_cfg.training.save_interval

    for epoch in range(starting_epoch, rl_cfg.training.num_epochs):
        epoch_start_time = time.time()
        val_mean_reward: float | None = None  # Populated if validation runs

        # Update epoch counter for scheduling
        agent.current_epoch = epoch

        # Choose training method based on configuration
        use_on_the_fly = getattr(rl_cfg.training, "on_the_fly_sampling", True)

        if use_on_the_fly:
            # On-the-fly training: process mini-batches directly from DataLoader
            metrics = agent.update_on_the_fly(train_loader)
        else:
            # Original batch collection method
            batch_data = agent.collect_batch_from_dataset(
                train_loader, rl_cfg.training.dataloader_batch_size
            )

            # Check if we have enough data
            if len(batch_data["actions"]) < rl_cfg.sac.mini_batch_size:
                logger.warning(
                    f"Collected only {len(batch_data['actions'])} samples, "
                    f"less than mini_batch_size {rl_cfg.sac.mini_batch_size}"
                )

            # Update agent
            metrics = agent.update(batch_data)

        epoch_time = time.time() - epoch_start_time

        # Log progress
        if epoch % rl_cfg.training.log_interval == 0:
            resume_prefix = (
                "[RESUMED] " if is_resuming and epoch == starting_epoch else ""
            )
            logger.info(
                f"{resume_prefix}Epoch {epoch:3d}: "
                f"Policy Loss = {metrics['policy_loss']:.4f}, "
                f"Q1 Loss = {metrics['q1_loss']:.4f}, "
                f"Q2 Loss = {metrics['q2_loss']:.4f}, "
                f"Mean Q = {metrics['mean_q_value']:.4f}, "
                f"Alpha = {metrics['alpha']:.4f}, "
                f"Mean Reward = {metrics['mean_reward']:.4f}, "
                f"BC Loss = {metrics['bc_loss']:.4f}, "
                f"Action Reg = {metrics['action_reg_loss']:.4f}, "
                f"CQL Loss = {metrics['cql_loss']:.4f}, "
                f"Policy Std = {metrics['policy_action_std']:.4f}, "
                f"Time = {epoch_time:.2f}s"
            )

            # TensorBoard logging
            if tb_logger is not None:
                tb_logger.log_scalar("train/policy_loss", metrics["policy_loss"], epoch)
                tb_logger.log_scalar("train/q1_loss", metrics["q1_loss"], epoch)
                tb_logger.log_scalar("train/q2_loss", metrics["q2_loss"], epoch)
                tb_logger.log_scalar(
                    "train/mean_q_value", metrics["mean_q_value"], epoch
                )
                tb_logger.log_scalar("train/alpha", metrics["alpha"], epoch)
                tb_logger.log_scalar("train/mean_reward", metrics["mean_reward"], epoch)
                tb_logger.log_scalar("train/bc_loss", metrics["bc_loss"], epoch)
                tb_logger.log_scalar(
                    "train/action_reg_loss", metrics["action_reg_loss"], epoch
                )
                tb_logger.log_scalar("train/anchor_loss", metrics["anchor_loss"], epoch)
                tb_logger.log_scalar("train/cql_loss", metrics["cql_loss"], epoch)
                tb_logger.log_scalar(
                    "train/policy_action_std", metrics["policy_action_std"], epoch
                )
                if "alpha_loss" in metrics:
                    tb_logger.log_scalar(
                        "train/alpha_loss", metrics["alpha_loss"], epoch
                    )
                if "cql_lambda" in metrics:
                    tb_logger.log_scalar(
                        "train/cql_lambda", metrics["cql_lambda"], epoch
                    )

        # Validation evaluation
        if val_dataset and epoch % rl_cfg.training.val_interval == 0:
            logger.info("Running validation...")
            val_batch_data = agent.collect_batch_from_dataset(
                val_loader, rl_cfg.training.dataloader_batch_size // 2
            )
            val_mean_reward = val_batch_data["rewards"].mean().item()
            logger.info("Validation mean reward: %.4f", val_mean_reward)

        # ----------------------------------------------------------------
        # Determine current metric & checkpointing -----------------------
        # ----------------------------------------------------------------
        # Determine metric: prefer validation reward if available
        if val_mean_reward is not None:
            current_metric = val_mean_reward
        else:
            current_metric = metrics["mean_reward"]
        current_metric = float(current_metric)

        # Save *best* checkpoint ----------------------------------------
        if current_metric >= best_metric:
            best_metric = current_metric
            _save_checkpoint(agent, epoch, best_metric, output_dir, "best", raw_cfg)

        # Save *latest* checkpoint periodically -------------------------
        if (epoch % save_interval == 0) or (epoch == rl_cfg.training.num_epochs - 1):
            _save_checkpoint(
                agent, epoch, current_metric, output_dir, "latest", raw_cfg
            )

        # Log total elapsed time periodically
        if epoch % rl_cfg.training.time_log_interval == 0 and epoch > 0:
            total_elapsed_time = time.time() - training_start_time
            logger.info(
                f"Total elapsed time after {epoch} epochs: {total_elapsed_time:.2f}s"
            )

    # Final evaluation
    total_training_time = time.time() - training_start_time
    logger.info("\nTraining completed in %.2fs", total_training_time)

    # Ensure final checkpoint is saved
    _save_checkpoint(
        agent,
        rl_cfg.training.num_epochs - 1,
        best_metric,
        output_dir,
        "latest",
        raw_cfg,
    )

    # Cleanup TensorBoard logger
    if tb_logger is not None:
        tb_logger.close()


def main() -> None:
    """CLI entry-point for SAC delta-action training."""
    parser = argparse.ArgumentParser(description="Train SAC for delta action learning")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="playground/rl/simple_rl/rl_galaxea_sac.yaml",
        help="Path to RL configuration file",
    )
    parser.add_argument(
        "-r",
        "--resume-checkpoint",
        type=str,
        help="Path to checkpoint file to resume training from (overrides config)",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load configuration ------------------------------------------------
    # ------------------------------------------------------------------
    cfg_path = Path(args.config).expanduser().resolve()
    with cfg_path.open("r", encoding="utf-8") as fp:
        raw_cfg: Dict[str, Any] = yaml.safe_load(fp)

    # Override resume checkpoint from command line if provided
    if args.resume_checkpoint:
        resume_path = Path(args.resume_checkpoint).expanduser().resolve()
        if not resume_path.exists():
            logger.error(f"Resume checkpoint not found: {resume_path}")
            sys.exit(1)

        # Update the config with the resume path
        if "training" not in raw_cfg:
            raw_cfg["training"] = {}
        raw_cfg["training"]["resume_checkpoint_path"] = str(resume_path)
        logger.info(f"🔄 Command line override: resuming from {resume_path}")

    rl_cfg = cattrs.structure(raw_cfg, RLConfig)

    # ------------------------------------------------------------------
    # Prepare output directory and meta ---------------------------------
    # ------------------------------------------------------------------
    # Always create a new output directory, even when resuming
    # This allows easy rollback if resumed training doesn't go well
    output_dir = _create_output_dir(rl_cfg.training.output_base_dir)

    # Log resume information
    if rl_cfg.training.resume_checkpoint_path:
        logger.info(f"🔄 Will resume from: {rl_cfg.training.resume_checkpoint_path}")
        logger.info(f"📁 New output directory: {output_dir}")
        logger.info("💡 This preserves the original training run for easy rollback")

    meta = {
        "created_at": datetime.now().isoformat(),
        "config_path": str(cfg_path),
        "config_values": raw_cfg,
        "resumed_from": rl_cfg.training.resume_checkpoint_path,
    }
    (output_dir / "meta.json").write_text(_custom_json_dumps(meta, max_indent_level=3))

    # ------------------------------------------------------------------
    # Run training with cleanup logic -----------------------------------
    # ------------------------------------------------------------------
    try:
        train_sac_delta_action(rl_cfg=rl_cfg, raw_cfg=raw_cfg, output_dir=output_dir)

    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt detected -- terminating training early.")
    except Exception as exc:
        logger.error("Training failed with exception: %s", exc)
        raise
    finally:
        # If no checkpoint *.pt exists in output_dir, remove the directory
        # (but only if this is a new directory, not a resumed one)
        if not rl_cfg.training.resume_checkpoint_path:
            has_ckpt = any(p.suffix == ".pt" for p in output_dir.glob("*.pt"))
            if not has_ckpt:
                logger.info(
                    "No checkpoint file found -- removing incomplete run directory %s",
                    output_dir,
                )
                try:
                    shutil.rmtree(output_dir)
                except OSError as err:
                    logger.error("Failed to remove directory %s: %s", output_dir, err)


if __name__ == "__main__":
    main()
