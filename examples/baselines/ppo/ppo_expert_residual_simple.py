"""
Simple Expert+Residual PPO for ManiSkill

A simplified, working implementation that avoids wrapper conflicts
by using Expert+Residual environments directly with proper PPO training.
"""

import math
import os
import random
import time
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import mani_skill.envs
import numpy as np
import torch
import tyro
from mani_skill.envs.wrappers.expert_residual import (
    ExpertResidualWrapper,
    create_expert_policy,
)
from torch import nn, optim
from torch.distributions.normal import Normal


@dataclass
class Args:
    # Experiment
    exp_name: Optional[str] = None
    seed: int = 1

    # Environment
    env_id: str = "PickBox-v1"
    robot_uids: str = "a1_galaxea"
    num_envs: int = 1024
    num_steps: int = 50
    total_timesteps: int = 100000

    # Expert+Residual
    expert_type: str = "zero"
    residual_scale: float = 1.0

    # PPO
    learning_rate: float = 3e-4
    gamma: float = 0.8
    gae_lambda: float = 0.9
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Computed
    batch_size: int = 0
    num_iterations: int = 0


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, n_obs, n_act, device=None):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(n_obs, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1, device=device)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(n_obs, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, n_act, device=device), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, n_act, device=device))

    def get_value(self, x):
        return self.critic(x)

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


def main():
    args = tyro.cli(Args)

    # Computed args
    args.batch_size = args.num_envs * args.num_steps
    args.num_iterations = args.total_timesteps // args.batch_size

    if args.exp_name is None:
        args.exp_name = (
            f"{args.env_id}_expert_residual_{args.expert_type}_{int(time.time())}"
        )

    print(f"Expert+Residual PPO Training: {args.exp_name}")
    print(f"Environment: {args.env_id} with {args.robot_uids}")
    print(f"Expert type: {args.expert_type}")
    print(f"Parallel environments: {args.num_envs}")
    print(f"Total timesteps: {args.total_timesteps}")

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Environment setup
    env_kwargs = dict(
        obs_mode="state",
        render_mode="rgb_array",
        sim_backend="physx_cuda",
        control_mode="pd_joint_delta_pos",
        robot_uids=args.robot_uids,
    )

    # Create expert policy
    temp_env = gym.make(args.env_id, num_envs=1, **env_kwargs)
    action_dim = temp_env.action_space.shape[0]
    temp_env.close()

    expert_policy = create_expert_policy(args.expert_type, action_dim)

    # Create Expert+Residual environment
    env = ExpertResidualWrapper(
        env_id=args.env_id,
        expert_policy_fn=expert_policy,
        num_envs=args.num_envs,
        residual_scale=args.residual_scale,
        clip_final_action=True,
        expert_action_noise=0.0,
        log_actions=False,
        track_action_stats=False,
        device=str(device),
        **env_kwargs,
    )

    # Get dimensions
    if hasattr(env.observation_space, "shape"):
        if len(env.observation_space.shape) == 1:
            n_obs = env.observation_space.shape[0]
        else:
            n_obs = env.observation_space.shape[-1]  # Per-environment obs dim
    else:
        n_obs = env.observation_space.shape[0]

    if hasattr(env.action_space, "shape"):
        if len(env.action_space.shape) == 1:
            n_act = env.action_space.shape[0]
        else:
            n_act = env.action_space.shape[-1]  # Per-environment action dim
    else:
        n_act = env.action_space.shape[0]

    print(f"Environment observation space: {env.observation_space.shape}")
    print(f"Environment action space: {env.action_space.shape}")
    print(f"Per-environment observation dimensions: {n_obs}")
    print(f"Per-environment action dimensions: {n_act}")

    # Agent setup
    agent = Agent(n_obs, n_act, device=device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Storage
    obs = torch.zeros((args.num_steps, args.num_envs, n_obs), device=device)
    actions = torch.zeros((args.num_steps, args.num_envs, n_act), device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    # Start training
    global_step = 0
    start_time = time.time()
    next_obs, _ = env.reset()
    next_done = torch.zeros(args.num_envs, device=device)

    print("Starting training...")
    print(f"Initial observation shape: {next_obs.shape}")

    for iteration in range(1, args.num_iterations + 1):
        # Rollout phase
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = env.step(action)
            rewards[step] = reward
            next_done = torch.logical_or(terminations, truncations)

        # Bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards, device=device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done.float()
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1].float()
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        # Flatten batch
        b_obs = obs.reshape((-1, n_obs))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1, n_act))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimize
        _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs, b_actions)
        logratio = newlogprob - b_logprobs
        ratio = logratio.exp()

        # Normalize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (
            b_advantages.std() + 1e-8
        )

        # Policy loss
        pg_loss1 = -b_advantages * ratio
        pg_loss2 = -b_advantages * torch.clamp(
            ratio, 1 - args.clip_coef, 1 + args.clip_coef
        )
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss
        v_loss = 0.5 * ((newvalue.flatten() - b_returns) ** 2).mean()

        # Entropy loss
        entropy_loss = entropy.mean()

        # Total loss
        loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
        optimizer.step()

        # Logging
        if iteration % 10 == 0:
            elapsed_time = time.time() - start_time
            sps = int(global_step / elapsed_time)
            avg_reward = rewards.mean().item()

            print(
                f"Iter {iteration:4d} | Steps {global_step:8d} | "
                f"SPS {sps:5d} | Reward {avg_reward:7.3f} | "
                f"PG Loss {pg_loss.item():.4f} | V Loss {v_loss.item():.4f}"
            )

    env.close()
    print(f"Training completed! Total time: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
