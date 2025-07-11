"""
Copyright 2025 Zordi, Inc. All rights reserved.

SAC networks for custom multimodal encoders.
"""

import numpy as np
import torch
from multimodal_encoders import MultimodalEncoder
from torch import nn

LOG_STD_MAX = 2
LOG_STD_MIN = -5


class SACPolicyNetwork(nn.Module):
    """SAC policy network with multimodal encoder."""

    def __init__(
        self, action_dim: int, encoder: MultimodalEncoder, action_scale: torch.Tensor, action_bias: torch.Tensor
    ):
        super().__init__()
        self.encoder = encoder
        self.action_dim = action_dim

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(encoder.output_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )

        # Mean and log_std layers
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_logstd = nn.Linear(256, action_dim)

        # Action rescaling
        self.register_buffer("action_scale", action_scale)
        self.register_buffer("action_bias", action_bias)

    def forward(self, obs):
        features = self.encoder(obs)
        x = self.policy_head(features)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_eval_action(self, obs):
        mean, log_std = self.forward(obs)
        action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action

    def get_action(self, obs):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class SACQNetwork(nn.Module):
    """SAC Q-network with multimodal encoder."""

    def __init__(self, action_dim: int, encoder: MultimodalEncoder):
        super().__init__()
        self.encoder = encoder
        self.action_dim = action_dim

        self.q_head = nn.Sequential(
            nn.Linear(encoder.output_dim + action_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, obs, action):
        obs_features = self.encoder(obs)
        q_input = torch.cat([obs_features, action], dim=-1)
        return self.q_head(q_input).squeeze(-1)


def create_sac_networks(
    state_dim: int,
    action_dim: int,
    action_high: np.ndarray,
    action_low: np.ndarray,
    image_channels: int = 3,
    image_size: tuple = (64, 64),
    use_dinov2: bool = False,
):
    """Factory function to create SAC networks."""
    # Create multimodal encoder
    encoder = MultimodalEncoder(
        state_dim=state_dim,
        image_channels=image_channels,
        image_size=image_size,
        output_dim=512,
        use_dinov2=use_dinov2,
    )

    # Action scaling
    action_scale = torch.FloatTensor((action_high - action_low) / 2.0)
    action_bias = torch.FloatTensor((action_high + action_low) / 2.0)

    # Create networks
    policy = SACPolicyNetwork(action_dim, encoder, action_scale, action_bias)
    q1 = SACQNetwork(action_dim, encoder)
    q2 = SACQNetwork(action_dim, encoder)

    return policy, q1, q2
