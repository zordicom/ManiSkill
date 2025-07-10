#!/usr/bin/env python3
"""
PPO with RGB observations and Expert+Residual Action Decomposition for ManiSkill.

This script reuses components from:
- ppo_rgb.py: CNN vision backbone, DictArray storage, CleanRL-style training loop.
- ppo_fast_expert_residual.py: ExpertResidualWrapper utilities and expert policy helpers.

It serves as the non-"fast" counterpart to ppo_rgb_fast_expert_residual.py (i.e. no
Torch Compile / CUDA-Graphs) while preserving the readable single-file style of
CleanRL examples.

Example usage (Zero-expert, 64 envs):

```bash
python ppo_rgb_expert_residual.py \
    --env-id PickBox-v1 \
    --robot-uids a1_galaxea \
    --expert-type zero \
    --num-envs 64 \
    --total-timesteps 1_000_000
```
"""

import math
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Tuple

import gymnasium as gym
import mani_skill.envs  # Register environments
import numpy as np
import torch
import tyro
from mani_skill.envs.wrappers.expert_residual import (
    ExpertResidualWrapper,
    create_expert_policy,
)
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import (
    FlattenActionSpaceWrapper,
    FlattenRGBDObservationWrapper,
)
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from torch import nn, optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

# -----------------------------------------------------------------------------
# 1.  Hyper-parameters & CLI (borrowed from ppo_rgb.py and extended)
# -----------------------------------------------------------------------------


@dataclass
class Args:
    """Command-line arguments (superset of ppo_rgb.py + expert fields)."""

    # Experiment setup
    exp_name: Optional[str] = None
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "ManiSkill-ExpertResidual-RGB"
    wandb_entity: Optional[str] = None
    wandb_group: str = "PPO-ExpertResidual-RGB"
    capture_video: bool = True
    save_trajectory: bool = False
    save_model: bool = True
    evaluate: bool = False
    checkpoint: Optional[str] = None
    render_mode: str = "all"

    # Environment
    env_id: str = "PickCube-v1"
    robot_uids: Optional[str] = None
    include_state: bool = True
    env_vectorization: str = "gpu"
    num_envs: int = 256
    num_eval_envs: int = 16
    partial_reset: bool = True
    eval_partial_reset: bool = False
    num_steps: int = 50
    num_eval_steps: int = 50
    reconfiguration_freq: Optional[int] = None
    eval_reconfiguration_freq: Optional[int] = 1
    eval_freq: int = 25
    save_train_video_freq: Optional[int] = None
    control_mode: Optional[str] = "pd_joint_delta_pos"

    # Expert+Residual
    expert_type: str = "zero"  # zero | ik | model
    residual_scale: float = 1.0
    expert_action_noise: float = 0.0
    track_action_stats: bool = False
    ik_gain: float = 2.0
    model_path: Optional[str] = None

    # PPO algorithm (same as ppo_rgb.py)
    total_timesteps: int = 10_000_000
    learning_rate: float = 3e-4
    anneal_lr: bool = False
    gamma: float = 0.8
    gae_lambda: float = 0.9
    num_minibatches: int = 32
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = False
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.2
    reward_scale: float = 1.0
    finite_horizon_gae: bool = False

    # Filled at runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


# -----------------------------------------------------------------------------
# 2.  Utility functions / modules (borrowed from ppo_rgb.py)
# -----------------------------------------------------------------------------


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0):
    """Orthogonal weight init (same as CleanRL)."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class DictArray:
    """Lightweight replay/storage for dict observations (from ppo_rgb.py)."""

    def __init__(self, buffer_shape, element_space, data_dict=None, device=None):
        self.buffer_shape = buffer_shape
        if data_dict is not None:
            self.data = data_dict
        else:
            assert isinstance(element_space, gym.spaces.Dict)
            self.data = {}
            for k, v in element_space.items():
                if isinstance(v, gym.spaces.Dict):
                    self.data[k] = DictArray(buffer_shape, v, device=device)
                else:
                    # Convert numpy dtypes to torch dtypes
                    if v.dtype in (np.float32, np.float64):
                        dtype = torch.float32
                    elif v.dtype == np.uint8:
                        dtype = torch.uint8
                    elif v.dtype == np.int16:
                        dtype = torch.int16
                    elif v.dtype == np.int32:
                        dtype = torch.int32
                    elif v.dtype == np.bool_:
                        dtype = torch.bool
                    else:
                        dtype = torch.float32  # Default fallback
                    self.data[k] = torch.zeros(
                        buffer_shape + v.shape, dtype=dtype, device=device
                    )

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.data[index]
        return {k: v[index] for k, v in self.data.items()}

    def __setitem__(self, index, value):
        if isinstance(index, str):
            self.data[index] = value
            return
        for k, v in value.items():
            self.data[k][index] = v

    @property
    def shape(self):
        return self.buffer_shape

    def reshape(self, shape):
        t = len(self.buffer_shape)
        new_dict = {}
        for k, v in self.data.items():
            new_dict[k] = (
                v.reshape(shape)
                if isinstance(v, DictArray)
                else v.reshape(shape + v.shape[t:])
            )
        new_buffer_shape = next(iter(new_dict.values())).shape[: len(shape)]
        return DictArray(new_buffer_shape, None, data_dict=new_dict)


class NatureCNN(nn.Module):
    """CNN backbone adapted for ExpertResidualWrapper nested observation structure."""

    def __init__(self, sample_obs, include_state=True):
        super().__init__()
        extractors = {}
        self.out_features = 0
        feature_size = 256

        # Handle RGB observations from sensor_data (ExpertResidualWrapper structure)
        if "sensor_data" in sample_obs:
            for camera_name, camera_data in sample_obs["sensor_data"].items():
                if "rgb" in camera_data:
                    rgb_obs = camera_data["rgb"]
                    c = rgb_obs.shape[-1]
                    img_size = (rgb_obs.shape[1], rgb_obs.shape[2])

                    cnn = nn.Sequential(
                        nn.Conv2d(c, 32, kernel_size=8, stride=4),
                        nn.ReLU(),
                        nn.Conv2d(32, 64, kernel_size=4, stride=2),
                        nn.ReLU(),
                        nn.Conv2d(64, 64, kernel_size=3, stride=1),
                        nn.ReLU(),
                        nn.Flatten(),
                    )
                    # compute flatten dim
                    with torch.no_grad():
                        n_flat = cnn(torch.zeros(1, c, *img_size)).shape[1]
                    fc = nn.Sequential(nn.Linear(n_flat, feature_size), nn.ReLU())
                    extractors[f"{camera_name}_rgb"] = nn.Sequential(cnn, fc)
                    self.out_features += feature_size

        # Handle state observations (from "extra" field in ExpertResidualWrapper)
        if "extra" in sample_obs and include_state:
            # Flatten extra observations (state data)
            extra_obs = sample_obs["extra"]
            if isinstance(extra_obs, dict):
                # Count total state dimensions
                total_state_dim = 0
                for key, value in extra_obs.items():
                    if hasattr(value, "shape"):
                        total_state_dim += np.prod(value.shape)
            else:
                total_state_dim = np.prod(extra_obs.shape)

            if total_state_dim > 0:
                extractors["state"] = nn.Sequential(
                    nn.Linear(total_state_dim, 256), nn.ReLU()
                )
                self.out_features += 256

        # Handle expert actions (added by ExpertResidualWrapper)
        if "expert_action" in sample_obs:
            expert_action_dim = sample_obs["expert_action"].shape[-1]
            extractors["expert_action"] = nn.Sequential(
                nn.Linear(expert_action_dim, 128), nn.ReLU()
            )
            self.out_features += 128

        self.extractors = nn.ModuleDict(extractors)

    def forward(self, obs: dict):
        tensors = []

        # Process RGB observations from sensor_data
        if "sensor_data" in obs:
            for camera_name, camera_data in obs["sensor_data"].items():
                if "rgb" in camera_data:
                    extractor_key = f"{camera_name}_rgb"
                    if extractor_key in self.extractors:
                        rgb_data = camera_data["rgb"]
                        # Convert from (B, H, W, C) to (B, C, H, W) and normalize
                        rgb_data = rgb_data.float().permute(0, 3, 1, 2) / 255.0
                        tensors.append(self.extractors[extractor_key](rgb_data))

        # Process state observations from extra
        if "extra" in obs and "state" in self.extractors:
            extra_obs = obs["extra"]
            if isinstance(extra_obs, dict):
                # Flatten all extra observations
                state_parts = []
                for key in sorted(extra_obs.keys()):
                    value = extra_obs[key]
                    if hasattr(value, "flatten"):
                        state_parts.append(value.flatten(start_dim=1))
                if state_parts:
                    state_tensor = torch.cat(state_parts, dim=1)
                    tensors.append(self.extractors["state"](state_tensor))
            else:
                # Direct tensor
                state_tensor = extra_obs.flatten(start_dim=1)
                tensors.append(self.extractors["state"](state_tensor))

        # Process expert actions
        if "expert_action" in obs and "expert_action" in self.extractors:
            expert_action = obs["expert_action"]
            tensors.append(self.extractors["expert_action"](expert_action))

        if not tensors:
            # Fallback if no valid observations found
            return torch.zeros(
                obs.get("expert_action", torch.zeros(1, 1)).shape[0],
                1,
                device=obs.get("expert_action", torch.zeros(1, 1)).device,
            )

        return torch.cat(tensors, dim=1)


class Agent(nn.Module):
    """Gaussian policy + value using NatureCNN features."""

    def __init__(self, envs, sample_obs, include_state=True):
        super().__init__()
        self.feature_net = NatureCNN(sample_obs, include_state=include_state)
        latent = self.feature_net.out_features
        act_dim = envs.single_action_space.shape[0]
        self.critic = nn.Sequential(
            layer_init(nn.Linear(latent, 512)), nn.ReLU(), layer_init(nn.Linear(512, 1))
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(latent, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, act_dim), std=0.01 * np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    # Helpers
    def get_features(self, obs):
        return self.feature_net(obs)

    def get_value(self, obs):
        return self.critic(self.get_features(obs))

    def get_action_and_value(self, obs, action=None):
        feats = self.get_features(obs)
        mean = self.actor_mean(feats)
        std = torch.exp(self.actor_logstd.expand_as(mean))
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        return (
            action,
            dist.log_prob(action).sum(1),
            dist.entropy().sum(1),
            self.critic(feats),
        )


class Logger:
    """Tiny helper for TensorBoard + optional WandB."""

    def __init__(self, log_wandb: bool, tb_writer: SummaryWriter):
        self.writer = tb_writer
        self.log_wandb = log_wandb
        if log_wandb:
            import wandb

            self.wandb = wandb
        else:
            self.wandb = None

    def add_scalar(self, tag, val, step):
        if self.wandb is not None:
            self.wandb.log({tag: val}, step=step)
        self.writer.add_scalar(tag, val, step)

    def close(self):
        self.writer.close()


# -----------------------------------------------------------------------------
# 3.  Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    args = tyro.cli(Args)

    # Derived params
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size

    run_name = (
        args.exp_name
        or f"{args.env_id}__ppo_rgb_expert_residual__{args.expert_type}__{args.seed}__{int(time.time())}"
    )

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # ---------------------------------------------------------------------
    # Environment creation
    # ---------------------------------------------------------------------

    env_kwargs = dict(
        obs_mode="rgb",
        render_mode=args.render_mode,
        sim_backend="physx_cuda",  # Always use physx_cuda - CPU backend not supported
    )
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode
    if args.robot_uids is not None:
        env_kwargs["robot_uids"] = args.robot_uids

    # Create dummy env to get action dim for expert
    dummy = gym.make(args.env_id, num_envs=1, **env_kwargs)
    action_dim = dummy.action_space.shape[-1]
    dummy.close()

    expert_kwargs = {}
    if args.expert_type == "ik":
        expert_kwargs["gain"] = args.ik_gain
    elif args.expert_type == "model":
        expert_kwargs.update(dict(model_path=args.model_path, device=str(device)))

    expert_policy = create_expert_policy(args.expert_type, action_dim, **expert_kwargs)

    # -- Training envs --
    envs = ExpertResidualWrapper(
        env_id=args.env_id,
        expert_policy_fn=expert_policy,
        num_envs=args.num_envs if not args.evaluate else 1,
        residual_scale=args.residual_scale,
        clip_final_action=True,
        expert_action_noise=args.expert_action_noise,
        log_actions=False,
        track_action_stats=args.track_action_stats,
        device=str(device),
        reconfiguration_freq=args.reconfiguration_freq,
        **env_kwargs,
    )

    # -- Evaluation envs --
    eval_envs = ExpertResidualWrapper(
        env_id=args.env_id,
        expert_policy_fn=expert_policy,
        num_envs=args.num_eval_envs,
        residual_scale=args.residual_scale,
        clip_final_action=True,
        expert_action_noise=0.0,
        log_actions=False,
        track_action_stats=False,
        device=str(device),
        reconfiguration_freq=args.eval_reconfiguration_freq,
        human_render_camera_configs=dict(shader_pack="default"),
        **env_kwargs,
    )

    # Note: Do NOT apply FlattenRGBDObservationWrapper since we handle dict observations directly
    # The ExpertResidualWrapper creates nested observations that conflict with flattening

    # Flatten action space if needed (e.g., Dict actions)
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)

    # Record videos / trajectories
    if args.capture_video or args.save_trajectory:
        vid_dir = f"runs/{run_name}/videos"
        if args.evaluate:
            vid_dir = f"{os.path.dirname(args.checkpoint)}/test_videos"
        print(f"Saving eval trajectories/videos to {vid_dir}")
        eval_envs = RecordEpisode(
            eval_envs,
            output_dir=vid_dir,
            save_trajectory=args.save_trajectory,
            save_video=args.capture_video,
            trajectory_name="trajectory",
            max_steps_per_video=args.num_eval_steps,
            video_fps=30,
        )
        if args.save_train_video_freq is not None:
            train_vid_trigger = (
                lambda x: (x // args.num_steps) % args.save_train_video_freq == 0
            )
            envs = RecordEpisode(
                envs,
                output_dir=f"runs/{run_name}/train_videos",
                save_trajectory=False,
                save_video_trigger=train_vid_trigger,
                max_steps_per_video=args.num_steps,
                video_fps=30,
            )

    # Wrap with ManiSkillVectorEnv for metric tracking
    envs = ManiSkillVectorEnv(
        envs,
        args.num_envs,
        ignore_terminations=not args.partial_reset,
        record_metrics=True,
    )
    eval_envs = ManiSkillVectorEnv(
        eval_envs,
        args.num_eval_envs,
        ignore_terminations=not args.eval_partial_reset,
        record_metrics=True,
    )

    assert isinstance(envs.single_action_space, gym.spaces.Box), (
        "Only continuous actions supported"
    )

    # ---------------------------------------------------------------------
    # Storage tensors
    # ---------------------------------------------------------------------

    sample_obs = envs.reset()[0]
    agent = Agent(envs, sample_obs, include_state=args.include_state).to(device)
    if args.checkpoint:
        agent.load_state_dict(torch.load(args.checkpoint, map_location=device))

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # DictArray storage for observations
    obs_buf = DictArray(
        (args.num_steps, args.num_envs), envs.single_observation_space, device=device
    )
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape, device=device
    )
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    # Logger
    tb_writer = SummaryWriter(f"runs/{run_name}")
    tb_writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n"
        + "\n".join([f"|{k}|{v}|" for k, v in vars(args).items()]),
    )
    logger = Logger(log_wandb=args.track, tb_writer=tb_writer)

    # ---------------------------------------------------------------------
    # Training/evaluation loop (largely identical to ppo_rgb.py)
    # ---------------------------------------------------------------------

    global_step = 0
    start_time = time.time()
    next_obs = sample_obs  # already obtained
    next_done = torch.zeros(args.num_envs, device=device)

    def clip_action(act):
        low = torch.from_numpy(envs.single_action_space.low).to(device)
        high = torch.from_numpy(envs.single_action_space.high).to(device)
        return torch.clamp(act, low, high)

    cumulative_times = defaultdict(float)

    for iteration in range(1, args.num_iterations + 1):
        # Evaluation --------------------------------------------------
        if iteration % args.eval_freq == 1:
            agent.eval()
            eval_obs, _ = eval_envs.reset()
            eval_returns = []
            eval_success = []
            for _ in range(args.num_eval_steps):
                with torch.no_grad():
                    eval_actions = agent.get_action_and_value(eval_obs)[0]
                    eval_obs, _, term, trunc, infos = eval_envs.step(eval_actions)
                    if "final_info" in infos:
                        mask = infos["_final_info"]
                        for k, v in infos["final_info"]["episode"].items():
                            if logger:
                                logger.add_scalar(
                                    f"eval/{k}", v[mask].float().mean(), global_step
                                )
                        eval_success.extend(
                            infos["final_info"]
                            .get("success", torch.zeros_like(mask))
                            .tolist()
                        )
                        eval_returns.extend(
                            infos["final_info"]["episode"]["r"].tolist()
                        )
            if eval_returns:
                logger.add_scalar(
                    "eval/return_mean", np.mean(eval_returns), global_step
                )
                logger.add_scalar(
                    "eval/success_rate", np.mean(eval_success), global_step
                )
            agent.train()

        # Learning rate anneal
        if args.anneal_lr:
            frac = 1 - (iteration - 1) / args.num_iterations
            for pg in optimizer.param_groups:
                pg["lr"] = frac * args.learning_rate

        # Rollout ------------------------------------------------------
        rollout_start = time.perf_counter()
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs_buf[step] = next_obs
            dones[step] = next_done
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
            actions[step], logprobs[step], values[step] = (
                action,
                logprob,
                value.flatten(),
            )
            # Step env
            next_obs, rew, term, trunc, infos = envs.step(clip_action(action))
            next_done = torch.logical_or(term, trunc)
            rewards[step] = rew * args.reward_scale

            # Optional: log expert/residual norms every 25 env-steps
            if "expert_action" in infos and step % 25 == 0 and logger is not None:
                e_norm = torch.norm(infos["expert_action"], dim=1).mean()
                r_norm = torch.norm(infos["residual_action"], dim=1).mean()
                logger.add_scalar(
                    "expert_residual/expert_action_norm", e_norm.item(), global_step
                )
                logger.add_scalar(
                    "expert_residual/residual_action_norm", r_norm.item(), global_step
                )

            # Per-episode metrics
            if "final_info" in infos:
                mask = infos["_final_info"]
                for k, v in infos["final_info"]["episode"].items():
                    logger.add_scalar(f"train/{k}", v[mask].float().mean(), global_step)
        rollout_time = time.perf_counter() - rollout_start
        cumulative_times["rollout_time"] += rollout_time

        # Compute returns/advantages ----------------------------------
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards, device=device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_non_terminal = 1.0 - next_done
                    next_values = next_value
                else:
                    next_non_terminal = 1.0 - dones[t + 1]
                    next_values = values[t + 1]
                delta = (
                    rewards[t]
                    + args.gamma * next_values * next_non_terminal
                    - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta
                    + args.gamma * args.gae_lambda * next_non_terminal * lastgaelam
                )
            returns = advantages + values

        # Flatten batch ----------------------------------------------
        b_obs = obs_buf.reshape((args.num_steps * args.num_envs,))
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_adv = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # PPO update --------------------------------------------------
        agent.train()
        inds = np.arange(args.batch_size)
        clipfracs = []
        update_start = time.perf_counter()
        for epoch in range(args.update_epochs):
            np.random.shuffle(inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                mb_inds = inds[start : start + args.minibatch_size]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                ratio = (newlogprob - b_logprobs[mb_inds]).exp()

                with torch.no_grad():
                    approx_kl = (
                        (ratio - 1) - (newlogprob - b_logprobs[mb_inds])
                    ).mean()
                    clipfracs.append(
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    )

                mb_adv = b_adv[mb_inds]
                if args.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv
                    * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef),
                ).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef
                    )
                    v_loss = 0.5 * torch.max(v_unclipped, v_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break  # early stop PPO epoch
        update_time = time.perf_counter() - update_start
        cumulative_times["update_time"] += update_time

        # Logging -----------------------------------------------------
        sps = int((global_step) / (time.time() - start_time))
        logger.add_scalar("charts/SPS", sps, global_step)
        logger.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        logger.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        logger.add_scalar("time/rollout_time", rollout_time, global_step)
        logger.add_scalar("time/update_time", update_time, global_step)

    # -----------------------------------------------------------------
    # Save model / cleanup
    # -----------------------------------------------------------------
    if not args.evaluate and args.save_model:
        model_path = f"runs/{run_name}/final_ckpt.pt"
        torch.save(agent.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    envs.close()
    eval_envs.close()
    logger.close()
