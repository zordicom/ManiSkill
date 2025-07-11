#!/usr/bin/env python3
"""
Copyright 2025 Zordi, Inc. All rights reserved.

PPO implementation for delta action learning with multi-modal observations.

This script implements Proximal Policy Optimization for learning residual/delta actions
on top of expert demonstrations. Key features:

- **Multi-modal observations** supporting state, images, and expert actions
- **Delta action learning** where the policy outputs corrections to expert actions
- **Dataset-based training** using RLDataset for offline RL data
- **GPU acceleration** with automatic device detection
- **GAE-lambda advantage estimation** for stable learning
- **Numerical stability** with log_std clamping, policy mean bounding, and value clipping

The policy network learns to output small delta actions that are added to the expert
actions to improve performance. This approach leverages expert demonstrations while
allowing the agent to learn improvements.

Data flow:
1. Dataset provides: state history, expert_action, residual_action (target), rewards
2. Policy observes: [state, expert_action, images] → outputs delta_action
3. Target: delta_action should match the stored residual_action from dataset
4. Final action = expert_action + delta_action

This is an offline RL approach where we use PPO loss on pre-collected data rather
than fresh environment rollouts.

Example usage:
    python playground/rl/try_ppo_delta_action.py --config playground/rl/rl_galaxea_ppo.yaml
"""  # noqa: E501

import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import cattrs
import numpy as np
import torch
import yaml
from rl_configs import NetworkConfig, RLConfig
from rl_dataset import RLDataset
from torch import nn, optim
from torch.utils.data import DataLoader

from zordi_vla.utils.logging_utils import setup_logger

logger = setup_logger("ppo_delta_action")

# Numerical stability constants
LOG_STD_MIN, LOG_STD_MAX = -10.0, 0.0  # Following SB3 defaults
DELTA_MAX = 0.1  # Maximum sensible delta action magnitude (domain-specific)


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
    """Encoder for multimodal observations (state + images + expert_action)."""

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

        # Image encoders
        self.image_encoders = nn.ModuleDict()
        image_feature_dim = 0

        for key, obs_meta in shape_meta["obs"].items():
            if key not in {"state", "expert_action"} and "image_type" in obs_meta:
                channels, _, _ = obs_meta["shape"]

                conv_layers = []
                prev_channels = channels

                for out_channels, kernel_size, stride in zip(
                    network_cfg.image_conv_channels,
                    network_cfg.image_conv_kernel_sizes,
                    network_cfg.image_conv_strides,
                ):
                    conv_layers.extend([
                        nn.Conv2d(
                            prev_channels,
                            out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                        ),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(),
                    ])
                    prev_channels = out_channels

                pool_h, pool_w = network_cfg.image_adaptive_pool_size
                conv_layers.extend([
                    nn.AdaptiveAvgPool2d((pool_h, pool_w)),
                    nn.Flatten(),
                    nn.Linear(
                        prev_channels * pool_h * pool_w, network_cfg.image_fc_dim
                    ),
                    nn.LayerNorm(network_cfg.image_fc_dim),
                    nn.ReLU(),
                ])

                encoder = nn.Sequential(*conv_layers).to(device)
                self.image_encoders[key] = encoder
                image_feature_dim += network_cfg.image_fc_dim

        # Fusion layer
        total_dim = (
            network_cfg.state_encoder_dims[-1]
            + network_cfg.expert_action_encoder_dims[-1]
            + image_feature_dim
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

        image_info = []
        for key in self.image_encoders.keys():
            channels = self.network_cfg.image_conv_channels
            image_info.append(
                f"{key}: "
                f"Conv({' → '.join(map(str, channels))}) → "
                f"FC({self.network_cfg.image_fc_dim})"
            )

        logger.info("MultimodalEncoder Architecture:")
        logger.info(
            f"  State Encoder: "
            f"{self.shape_meta['obs']['state']['dim']} → {state_dims} (LayerNorm)"
        )
        logger.info(
            f"  Action Encoder: "
            f"{self.shape_meta['action']['shape'][0]} → {action_dims} (LayerNorm)"
        )
        for img_info in image_info:
            logger.info(f"  Image Encoder - {img_info} (BatchNorm2d + LayerNorm)")
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

        # Encode images
        for key, encoder in self.image_encoders.items():
            if key in obs:
                img = obs[key].squeeze(1)  # Remove extra batch dim from dataset
                img_feat = encoder(img)
                features.append(img_feat)

        # Fuse all features
        combined = torch.cat(features, dim=1)
        return self.fusion(combined)


class DeltaPolicyNetwork(nn.Module):
    """Policy network that outputs delta actions to add to expert actions."""

    def __init__(
        self,
        shape_meta: Dict[str, Any],
        network_cfg: NetworkConfig,
        device: torch.device,
        *,
        encoder: "MultimodalEncoder | None" = None,
    ):
        super().__init__()

        self.action_dim = shape_meta["action"]["shape"][0]
        self.device = device

        # Use shared encoder if provided, otherwise create a new one
        if encoder is None:
            self.encoder = MultimodalEncoder(shape_meta, network_cfg, device)
            # Initialize encoder weights only when we own it
            self.encoder.apply(_init_weights)
            self.encoder.log_architecture()
        else:
            self.encoder = encoder  # shared instance

        # ------------------------------------------------------------------
        # Policy head -------------------------------------------------------
        # ------------------------------------------------------------------
        policy_layers = []
        prev_dim = self.encoder.output_dim
        for dim in network_cfg.policy_head_dims:
            policy_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
            ])
            prev_dim = dim
        policy_layers.append(nn.Linear(prev_dim, self.action_dim))
        self.policy_head = nn.Sequential(*policy_layers).to(device)

        # Learnable log standard deviation for delta actions
        self.log_std = nn.Parameter(
            torch.full((self.action_dim,), network_cfg.log_std_init, device=device)
        )

        # Initialize only the policy head (encoder was already handled)
        self.policy_head.apply(_init_weights)

        # Initialize last layer with smaller weights for stability -----------
        last_layer = self.policy_head[-1]
        if isinstance(last_layer, nn.Linear):
            nn.init.orthogonal_(last_layer.weight, gain=1)
            nn.init.constant_(last_layer.bias, 0.0)
            with torch.no_grad():
                last_layer.weight.data *= 0.01

        # Log architecture summary for the policy head
        self._log_architecture()

    def _log_architecture(self) -> None:
        """Log a concise summary of the delta policy network architecture."""
        policy_dims = " → ".join(map(str, self.encoder.network_cfg.policy_head_dims))
        logger.info("DeltaPolicyNetwork Architecture:")
        logger.info(
            f"  Policy Head: "
            f"{self.encoder.output_dim} → {policy_dims} → {self.action_dim} (LayerNorm)"
        )
        logger.info(
            f"  Log Std: Learnable parameter initialized to "
            f"{self.encoder.network_cfg.log_std_init}"
        )

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass returning delta action mean."""
        features = self.encoder(obs)
        raw_mean = self.policy_head(features)
        # Bound the policy mean with tanh squashing for numerical stability
        mean = torch.tanh(raw_mean) * DELTA_MAX
        return mean

    def get_distribution(
        self, obs: Dict[str, torch.Tensor]
    ) -> torch.distributions.Normal:
        """Get delta action distribution for given observations."""
        mean = self.forward(obs)
        # Clamp log_std for numerical stability
        clamped_log_std = torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(clamped_log_std)
        return torch.distributions.Normal(mean, std)

    def get_action(
        self, obs: Dict[str, torch.Tensor], deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action by adding delta to expert action."""
        with torch.no_grad():
            dist = self.get_distribution(obs)
            if deterministic:
                delta_action = dist.mean
            else:
                delta_action = dist.sample()

            # Add delta to expert action
            final_action = obs["expert_action"] + delta_action
            log_prob = dist.log_prob(delta_action).sum(-1)

        return final_action, log_prob


class ValueNetwork(nn.Module):
    """Value network for multimodal state value estimation."""

    def __init__(
        self,
        shape_meta: Dict[str, Any],
        network_cfg: NetworkConfig,
        device: torch.device,
        *,
        encoder: "MultimodalEncoder | None" = None,
    ):
        super().__init__()
        self.device = device

        # ------------------------------------------------------------------
        # Shared or private encoder ----------------------------------------
        # ------------------------------------------------------------------
        if encoder is None:
            self.encoder = MultimodalEncoder(shape_meta, network_cfg, device)
            self.encoder.apply(_init_weights)
            self.encoder.log_architecture()
        else:
            self.encoder = encoder

        # ------------------------------------------------------------------
        # Value head --------------------------------------------------------
        # ------------------------------------------------------------------
        value_layers = []
        prev_dim = self.encoder.output_dim
        for dim in network_cfg.value_head_dims:
            value_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
            ])
            prev_dim = dim
        value_layers.append(nn.Linear(prev_dim, 1))
        self.value_head = nn.Sequential(*value_layers).to(device)

        # Initialize only the value head
        self.value_head.apply(_init_weights)

        # Fine-tune last layer initialization ------------------------------
        last_layer = self.value_head[-1]
        if isinstance(last_layer, nn.Linear):
            nn.init.orthogonal_(last_layer.weight, gain=1)
            nn.init.constant_(last_layer.bias, 0.0)
            with torch.no_grad():
                last_layer.weight.data *= 0.01

        # Log architecture summary
        self._log_architecture()

    def _log_architecture(self) -> None:
        """Log a concise summary of the value network architecture."""
        value_dims = " → ".join(map(str, self.encoder.network_cfg.value_head_dims))
        logger.info("ValueNetwork Architecture:")
        logger.info(
            f"  Value Head: {self.encoder.output_dim} → {value_dims} → 1 (LayerNorm)"
        )

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass returning state values."""
        features = self.encoder(obs)
        return self.value_head(features).squeeze(-1)


class PPODeltaAction:
    """PPO algorithm for delta action learning with multi-modal observations."""

    def __init__(
        self,
        shape_meta: Dict[str, Any],
        rl_cfg: RLConfig,
        device: torch.device,
    ):
        """Initialize PPO agent for delta action learning."""
        self.device = device
        self.shape_meta = shape_meta
        self.action_dim = shape_meta["action"]["shape"][0]
        self.ppo_cfg = rl_cfg.ppo

        # --------------------------------------------------------------
        # Build networks (optionally with shared encoder) --------------
        # --------------------------------------------------------------
        self.use_shared_encoder = rl_cfg.network.use_shared_encoder

        if self.use_shared_encoder:
            shared_encoder = MultimodalEncoder(shape_meta, rl_cfg.network, device)

            self.policy = DeltaPolicyNetwork(
                shape_meta, rl_cfg.network, device, encoder=shared_encoder
            )
            self.value = ValueNetwork(
                shape_meta, rl_cfg.network, device, encoder=shared_encoder
            )

            # Optimizers ------------------------------------------------
            # 1) Policy optimizer updates policy parameters **including** the shared
            # encoder.
            self.policy_optimizer = optim.Adam(
                self.policy.parameters(), lr=self.ppo_cfg.policy_learning_rate
            )

            # 2) Value optimizer updates only parameters **unique** to the value head.
            policy_param_ids = {id(p) for p in self.policy.parameters()}
            value_head_only_params = [
                p for p in self.value.parameters() if id(p) not in policy_param_ids
            ]
            self.value_optimizer = optim.Adam(
                value_head_only_params, lr=self.ppo_cfg.value_learning_rate
            )
        else:
            # Independent encoders (default behavior) ------------------
            self.policy = DeltaPolicyNetwork(shape_meta, rl_cfg.network, device)
            self.value = ValueNetwork(shape_meta, rl_cfg.network, device)

            # Optimizers with separate learning rates -------------------
            self.policy_optimizer = optim.Adam(
                self.policy.parameters(), lr=self.ppo_cfg.policy_learning_rate
            )
            self.value_optimizer = optim.Adam(
                self.value.parameters(), lr=self.ppo_cfg.value_learning_rate
            )

        self.rng = np.random.default_rng()

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
        all_values = []
        all_log_probs = []

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

            # Get values and action probabilities
            with torch.no_grad():
                values = self.value(obs)

                # Get delta action distribution and compute log prob for stored action
                dist = self.policy.get_distribution(obs)
                # The stored action is the residual/delta action from the dataset
                stored_delta_action = batch["action"].to(self.device)
                log_probs = dist.log_prob(stored_delta_action).sum(-1)

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

            all_actions.append(stored_delta_action)
            all_rewards.append(batch["reward"].to(self.device))
            all_dones.append(batch["done"].to(self.device))
            all_terminated.append(batch["terminated"].to(self.device))
            all_values.append(values)
            all_log_probs.append(log_probs)

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
            "values": torch.cat(all_values, dim=0),
            "log_probs": torch.cat(all_log_probs, dim=0),
        }

    def compute_advantages(self, batch_data: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute 1-step TD advantages and returns (TD(0)).

        Because we train on shuffled, offline batches (not full trajectories),
        the classic GAE recursion cannot be applied reliably.  Instead we use
        a single-step bootstrap:

        δ_t = r_t + γ V(s_{t+1}) - V(s_t)
        A_t = δ_t
        R_t = r_t + γ V(s_{t+1})
        """
        rewards_raw = batch_data["rewards"]
        # Reward normalization (zero mean, unit std per batch)
        rewards = (rewards_raw - rewards_raw.mean()) / (rewards_raw.std() + 1e-8)
        terminated = batch_data["terminated"]
        values = batch_data["values"]

        # Compute next values using next observations
        with torch.no_grad():
            next_values = self.value(batch_data["next_obs"])

        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        gamma = self.ppo_cfg.gamma

        # For offline RL, we compute GAE step-by-step
        # Since our dataset doesn't have trajectory structure, we use 1-step bootstrap
        for i in range(len(rewards)):
            if terminated[i]:
                next_value = 0.0
            else:
                next_value = next_values[i]

            # 1-step TD error
            delta = rewards[i] + gamma * next_value - values[i]

            advantages[i] = delta
            returns[i] = rewards[i] + gamma * next_value

        return advantages, returns

    def update(self, batch_data: Dict) -> Dict[str, float]:
        """Update policy and value networks."""
        advantages, returns = self.compute_advantages(batch_data)

        # Data is already concatenated from collect_batch_from_dataset
        all_obs = batch_data["obs"]
        all_actions = batch_data["actions"]
        all_log_probs = batch_data["log_probs"]
        all_values = batch_data["values"]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Training metrics
        policy_losses = []
        value_losses = []
        entropy_losses = []
        kl_divergences = []

        # Training epochs
        batch_size = len(all_actions)
        indices = np.arange(batch_size)

        for epoch in range(self.ppo_cfg.train_epochs):
            self.rng.shuffle(indices)

            epoch_kl = 0.0
            epoch_batches = 0

            for start in range(0, batch_size, self.ppo_cfg.mini_batch_size):
                end = min(start + self.ppo_cfg.mini_batch_size, batch_size)
                mb_idx = indices[start:end]

                # Mini-batch data
                mb_obs = {key: tensor[mb_idx] for key, tensor in all_obs.items()}
                mb_actions = all_actions[mb_idx]
                mb_old_log_probs = all_log_probs[mb_idx]
                mb_old_values = all_values[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]

                # Policy loss with entropy bonus
                dist = self.policy.get_distribution(mb_obs)
                new_log_probs = dist.log_prob(mb_actions).sum(-1)
                entropy = dist.entropy().sum(-1).mean()

                # Compute approximate KL divergence
                approx_kl = (mb_old_log_probs - new_log_probs).mean()
                kl_divergences.append(approx_kl.item())
                epoch_kl += approx_kl.item()
                epoch_batches += 1

                # Early stopping on large KL divergence
                if approx_kl > self.ppo_cfg.target_kl:
                    logger.warning(
                        f"Early stopping at epoch {epoch}, "
                        f"batch {start // self.ppo_cfg.mini_batch_size} "
                        f"due to KL divergence "
                        f"{approx_kl:.4f} > {self.ppo_cfg.target_kl}"
                    )
                    break

                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = (
                    torch.clamp(
                        ratio, 1 - self.ppo_cfg.clip_ratio, 1 + self.ppo_cfg.clip_ratio
                    )
                    * mb_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Add entropy bonus to encourage exploration
                entropy_bonus = self.ppo_cfg.entropy_bonus * entropy
                # Delta magnitude penalty (L2 regularizer on actions)
                delta_penalty = (
                    self.ppo_cfg.delta_penalty_coeff * (mb_actions**2).mean()
                )
                total_policy_loss = policy_loss - entropy_bonus + delta_penalty

                # Value loss with clipping
                values_pred = self.value(mb_obs)
                values_clipped = mb_old_values + (values_pred - mb_old_values).clamp(
                    -self.ppo_cfg.clip_ratio, self.ppo_cfg.clip_ratio
                )
                value_loss = (
                    0.5
                    * torch.max(
                        (values_pred - mb_returns) ** 2,
                        (values_clipped - mb_returns) ** 2,
                    ).mean()
                )

                # --------------------------------------------------
                # Optimization step (different if encoder is shared)
                # --------------------------------------------------

                if self.use_shared_encoder:
                    # Zero all gradients first
                    self.policy_optimizer.zero_grad()
                    self.value_optimizer.zero_grad()

                    # Backward passes (retain graph because encoder is shared)
                    total_policy_loss.backward(retain_graph=True)
                    value_loss.backward()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(), self.ppo_cfg.grad_clip_norm
                    )
                    torch.nn.utils.clip_grad_norm_(
                        self.value.parameters(), self.ppo_cfg.grad_clip_norm
                    )

                    # Apply updates: encoder params live in policy optimizer only
                    self.policy_optimizer.step()
                    self.value_optimizer.step()
                else:
                    # Original independent-update behavior ---------
                    self.policy_optimizer.zero_grad()
                    total_policy_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(), self.ppo_cfg.grad_clip_norm
                    )
                    self.policy_optimizer.step()

                    self.value_optimizer.zero_grad()
                    value_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.value.parameters(), self.ppo_cfg.grad_clip_norm
                    )
                    self.value_optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy.item())

            # Early stopping check for the entire epoch
            avg_epoch_kl = epoch_kl / max(epoch_batches, 1)
            if avg_epoch_kl > self.ppo_cfg.target_kl:
                logger.warning(
                    f"Early stopping at epoch {epoch} due to average KL divergence "
                    f"{avg_epoch_kl:.4f} > {self.ppo_cfg.target_kl}"
                )
                break

        # --------------------------------------------------------------
        # Aggregate metrics across training epochs and return ------------
        # --------------------------------------------------------------
        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropy_losses)),
            "mean_advantage": float(advantages.mean().item()),
            "mean_return": float(returns.mean().item()),
            "mean_kl": float(np.mean(kl_divergences)),
        }


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
    agent: "PPODeltaAction",  # Forward reference
    epoch: int,
    metric_value: float,
    out_dir: Path,
    tag: str,
    raw_cfg: Dict[str, Any],
) -> None:
    """Save checkpoint with model/optimizer states and metadata.

    Args:
        agent: The PPO agent containing networks & optimizers.
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
            "policy_state": agent.policy.state_dict(),
            "value_state": agent.value.state_dict(),
            "policy_opt_state": agent.policy_optimizer.state_dict(),
            "value_opt_state": agent.value_optimizer.state_dict(),
            "config": raw_cfg,
            "timestamp": datetime.now().isoformat(),
        },
        ckpt_path,
    )
    logger.info("Saved %s checkpoint to %s (metric=%.4f)", tag, ckpt_path, metric_value)


def train_ppo_delta_action(
    *, rl_cfg: RLConfig, raw_cfg: Dict[str, Any], output_dir: Path
) -> None:
    """Train PPO agent for delta action learning with checkpointing.

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
    logger.info(f"Training PPO Delta Action using device: {device}")

    # Create dataset
    logger.info(f"Loading dataset... ({Path(rl_cfg.dataset.path).resolve()})")
    train_dataset = RLDataset(cfg_rl=rl_cfg, is_train_split=True)
    val_dataset = train_dataset.get_validation_dataset()

    logger.info(f"Training samples: {len(train_dataset)}")
    if val_dataset:
        logger.info(f"Validation samples: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=rl_cfg.training.batch_size,
        shuffle=True,
        num_workers=rl_cfg.training.num_workers,
    )

    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=rl_cfg.training.batch_size,
            shuffle=False,
            num_workers=rl_cfg.training.num_workers,
        )

    # Get shape metadata
    shape_meta = train_dataset.get_shape_meta()
    logger.info(f"Shape metadata: {shape_meta}")

    # Create PPO agent
    agent = PPODeltaAction(shape_meta, rl_cfg, device)

    # Verify data compatibility
    logger.info("Verifying dataset compatibility...")
    sample_batch = next(iter(train_loader))
    sample_obs = agent._move_obs_to_device(sample_batch["obs"])

    try:
        # Test forward passes
        with torch.no_grad():
            test_value = agent.value(sample_obs)
            test_dist = agent.policy.get_distribution(sample_obs)
            test_action = test_dist.sample()
        logger.info("✓ Networks initialized successfully")
        logger.info(f"  Value output shape: {test_value.shape}")
        logger.info(f"  Action output shape: {test_action.shape}")

        policy_std = (
            torch.exp(torch.clamp(agent.policy.log_std, LOG_STD_MIN, LOG_STD_MAX))
            .mean()
            .item()
        )
        logger.info(f"  Policy std: {policy_std:.4f}")
        logger.info(
            f"  Policy mean range: "
            f"[{test_action.min().item():.4f}, {test_action.max().item():.4f}]"
        )
    except Exception as e:
        logger.error(f"Network compatibility test failed: {e}")
        raise

    # Training loop
    logger.info("Starting training...")
    training_start_time = time.time()

    save_interval = rl_cfg.training.save_interval

    best_metric = float("-inf")
    for epoch in range(rl_cfg.training.num_epochs):
        epoch_start_time = time.time()
        val_mean_return: float | None = None  # Populated if validation runs

        # Collect batch from dataset
        batch_data = agent.collect_batch_from_dataset(
            train_loader, rl_cfg.training.batch_size
        )

        # Check if we have enough data
        if len(batch_data["actions"]) < rl_cfg.ppo.mini_batch_size:
            logger.warning(
                f"Collected only {len(batch_data['actions'])} samples, "
                f"less than mini_batch_size {rl_cfg.ppo.mini_batch_size}"
            )

        # Update agent
        metrics = agent.update(batch_data)

        epoch_time = time.time() - epoch_start_time

        # Log progress
        if epoch % rl_cfg.training.log_interval == 0:
            logger.info(
                f"Epoch {epoch:3d}: "
                f"Policy Loss = {metrics['policy_loss']:.4f}, "
                f"Value Loss = {metrics['value_loss']:.4f}, "
                f"Entropy = {metrics['entropy']:.4f}, "
                f"Mean Return = {metrics['mean_return']:.4f}, "
                f"Mean KL = {metrics['mean_kl']:.6f}, "
                f"Time = {epoch_time:.2f}s"
            )

        # Validation evaluation
        if val_dataset:
            logger.info("Running validation...")
            val_batch_data = agent.collect_batch_from_dataset(
                val_loader, rl_cfg.training.batch_size // 2
            )
            val_advantages, val_returns = agent.compute_advantages(val_batch_data)
            val_mean_return = val_returns.mean().item()
            logger.info("Validation mean return: %.4f", val_mean_return)

        # ----------------------------------------------------------------
        # Determine current metric & checkpointing -----------------------
        # ----------------------------------------------------------------
        # Determine metric: prefer validation return if available
        if val_mean_return is not None:
            current_metric = val_mean_return
        else:
            logger.warning("No validation dataset available, using mean return")
            current_metric = metrics["mean_return"]
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

    # Ensure final checkpoint is saved (could be missed if save_interval
    # does not divide num_epochs evenly)
    _save_checkpoint(
        agent,
        rl_cfg.training.num_epochs - 1,
        best_metric,
        output_dir,
        "latest",
        raw_cfg,
    )


def main() -> None:
    """CLI entry-point for PPO delta-action training."""
    parser = argparse.ArgumentParser(description="Train PPO for delta action learning")
    parser.add_argument(
        "--config",
        type=str,
        default="playground/rl/rl_galaxea_ppo.yaml",
        help="Path to RL configuration file",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load configuration ------------------------------------------------
    # ------------------------------------------------------------------
    cfg_path = Path(args.config).expanduser().resolve()
    with cfg_path.open("r", encoding="utf-8") as fp:
        raw_cfg: Dict[str, Any] = yaml.safe_load(fp)

    rl_cfg = cattrs.structure(raw_cfg, RLConfig)

    # ------------------------------------------------------------------
    # Prepare output directory and meta ---------------------------------
    # ------------------------------------------------------------------
    output_dir = _create_output_dir(rl_cfg.training.output_base_dir)

    meta = {
        "created_at": datetime.now().isoformat(),
        "config_path": str(cfg_path),
        "config_values": raw_cfg,
    }
    (output_dir / "meta.json").write_text(_custom_json_dumps(meta, max_indent_level=3))

    # ------------------------------------------------------------------
    # Run training with cleanup logic -----------------------------------
    # ------------------------------------------------------------------
    try:
        train_ppo_delta_action(rl_cfg=rl_cfg, raw_cfg=raw_cfg, output_dir=output_dir)

    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt detected -- terminating training early.")
    except Exception as exc:
        logger.error("Training failed with exception: %s", exc)
        raise
    finally:
        # If no checkpoint *.pt exists in output_dir, remove the directory
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
