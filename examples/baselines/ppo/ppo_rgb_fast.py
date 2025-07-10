#!/usr/bin/env python3
"""
Fast PPO with RGB observations for ManiSkill environments.

Combines the CNN/vision capabilities of ppo_rgb.py with all the performance
optimizations from ppo_fast.py:
- TensorDict integration for efficient tensor operations
- torch.compile support for 2-3x speed improvement
- CUDA graphs for maximum performance
- Optimized data pipeline with reduced memory allocations
- Better GPU memory utilization for vision-based training

Usage:
    python ppo_rgb_fast.py --env-id PickCube-v1 --robot-uids a1_galaxea --compile
"""

import math
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Tuple

import gymnasium as gym
import mani_skill.envs
import numpy as np
import tensordict
import torch
import tqdm
import tyro
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import (
    FlattenActionSpaceWrapper,
    FlattenRGBDObservationWrapper,
)
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from tensordict import from_module
from tensordict.nn import CudaGraphModule
from torch import nn, optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

import wandb

# Expert+Residual wrapper utility
try:
    from util import create_expert_residual_envs
except ImportError:
    create_expert_residual_envs = None

# Set optimization flags
os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"


@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ManiSkill"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    wandb_group: str = "PPO"
    """the group of the run for wandb"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_trajectory: bool = False
    """whether to save trajectory data into the `videos` folder"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    evaluate: bool = False
    """if toggled, only runs evaluation with the given model checkpoint and saves the evaluation trajectories"""
    checkpoint: Optional[str] = None
    """path to a pretrained checkpoint file to start evaluation/training from"""
    render_mode: str = "all"
    """the environment rendering mode"""

    # Environment specific arguments
    env_id: str = "PickCube-v1"
    """the id of the environment"""
    robot_uids: Optional[str] = None
    """robot uid(s) to use for the environment"""
    include_state: bool = True
    """whether to include state information in observations"""
    env_vectorization: str = "gpu"
    """the type of environment vectorization to use"""
    num_envs: int = 256
    """the number of parallel environments"""
    num_eval_envs: int = 16
    """the number of parallel evaluation environments"""
    partial_reset: bool = True
    """whether to let parallel environments reset upon termination instead of truncation"""
    eval_partial_reset: bool = False
    """whether to let parallel evaluation environments reset upon termination instead of truncation"""
    num_steps: int = 50
    """the number of steps to run in each environment per policy rollout"""
    num_eval_steps: int = 50
    """the number of steps to run in each evaluation environment during evaluation"""
    reconfiguration_freq: Optional[int] = None
    """how often to reconfigure the environment during training"""
    eval_reconfiguration_freq: Optional[int] = 1
    """for benchmarking purposes we want to reconfigure the eval environment each reset to ensure objects are randomized in some tasks"""
    eval_freq: int = 25
    """evaluation frequency in terms of iterations"""
    save_train_video_freq: Optional[int] = None
    """frequency to save training videos in terms of iterations"""
    control_mode: Optional[str] = "pd_joint_delta_pos"
    """the control mode to use for the environment"""

    # Algorithm specific arguments
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.8
    """the discount factor gamma"""
    gae_lambda: float = 0.9
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = 0.1
    """the target KL divergence threshold"""
    reward_scale: float = 1.0
    """Scale the reward by this factor"""
    finite_horizon_gae: bool = False

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    # Torch optimizations
    compile: bool = False
    """whether to use torch.compile."""
    cudagraphs: bool = False
    """whether to use cudagraphs on top of compile."""

    # Expert+Residual parameters (optional)
    expert_type: str = "none"
    """type of expert policy: 'none' (regular PPO), 'zero', 'ik', 'model'"""
    residual_scale: float = 1.0
    """scale factor for residual actions"""
    expert_action_noise: float = 0.0
    """Gaussian noise std to add to expert actions"""
    track_action_stats: bool = False
    """whether to track expert/residual action statistics"""
    ik_gain: float = 2.0
    """proportional gain for IK expert policy"""
    model_path: Optional[str] = None
    """path to pre-trained model for model expert policy"""


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class DictArray(object):
    """Efficient dictionary-based array for handling complex observations."""

    def __init__(self, buffer_shape, element_space, data_dict=None, device=None):
        self.buffer_shape = buffer_shape
        if data_dict:
            self.data = data_dict
        else:
            assert isinstance(element_space, gym.spaces.dict.Dict)
            self.data = {}
            for k, v in element_space.items():
                if isinstance(v, gym.spaces.dict.Dict):
                    self.data[k] = DictArray(buffer_shape, v, device=device)
                else:
                    dtype = (
                        torch.float32
                        if v.dtype in (np.float32, np.float64)
                        else torch.uint8
                        if v.dtype == np.uint8
                        else torch.int16
                        if v.dtype == np.int16
                        else torch.int32
                        if v.dtype == np.int32
                        else v.dtype
                    )
                    self.data[k] = torch.zeros(
                        buffer_shape + v.shape, dtype=dtype, device=device
                    )

    def keys(self):
        return self.data.keys()

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.data[index]
        return {k: v[index] for k, v in self.data.items()}

    def __setitem__(self, index, value):
        if isinstance(index, str):
            self.data[index] = value
        for k, v in value.items():
            self.data[k][index] = v

    @property
    def shape(self):
        return self.buffer_shape

    def reshape(self, shape):
        t = len(self.buffer_shape)
        new_dict = {}
        for k, v in self.data.items():
            if isinstance(v, DictArray):
                new_dict[k] = v.reshape(shape)
            else:
                new_dict[k] = v.reshape(shape + v.shape[t:])
        new_buffer_shape = next(iter(new_dict.values())).shape[: len(shape)]
        return DictArray(new_buffer_shape, None, data_dict=new_dict)


class NatureCNN(nn.Module):
    """Optimized CNN for processing RGB observations."""

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


class Agent(nn.Module):
    """Optimized agent with CNN and fast execution capabilities."""

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


class Logger:
    """Optimized logger with WandB and TensorBoard support."""

    def __init__(self, log_wandb=False, tensorboard: SummaryWriter = None) -> None:
        self.writer = tensorboard
        self.log_wandb = log_wandb

    def add_scalar(self, tag, scalar_value, step):
        if self.log_wandb:
            wandb.log({tag: scalar_value}, step=step)
        self.writer.add_scalar(tag, scalar_value, step)

    def close(self):
        self.writer.close()


def gae(next_obs, next_done, container, final_values):
    """Optimized Generalized Advantage Estimation."""
    # Bootstrap value if not done
    next_value = get_value(next_obs).reshape(-1)
    lastgaelam = 0
    nextnonterminals = (~container["dones"]).float().unbind(0)
    vals = container["vals"]
    vals_unbind = vals.unbind(0)
    rewards = container["rewards"].unbind(0)

    advantages = []
    nextnonterminal = (~next_done).float()
    nextvalues = next_value

    for t in range(args.num_steps - 1, -1, -1):
        cur_val = vals_unbind[t]
        real_next_values = (
            nextnonterminal * nextvalues + final_values[t]
        )  # t instead of t+1
        delta = rewards[t] + args.gamma * real_next_values - cur_val
        advantages.append(
            delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
        )
        lastgaelam = advantages[-1]

        nextnonterminal = nextnonterminals[t]
        nextvalues = cur_val

    advantages = container["advantages"] = torch.stack(list(reversed(advantages)))
    container["returns"] = advantages + vals
    return container


def rollout(obs, done):
    """Optimized rollout function with TensorDict integration."""
    ts = []
    final_values = torch.zeros((args.num_steps, args.num_envs), device=device)

    for step in range(args.num_steps):
        # ALGO LOGIC: action logic
        action, logprob, _, value = policy(obs=obs)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, next_done, infos = step_func(action)

        if "final_info" in infos:
            final_info = infos["final_info"]
            done_mask = infos["_final_info"]
            for k, v in final_info["episode"].items():
                logger.add_scalar(
                    f"train/{k}", v[done_mask].float().mean(), global_step
                )
            # Apply mask to each key in final_observation dictionary
            for k in infos["final_observation"]:
                infos["final_observation"][k] = infos["final_observation"][k][done_mask]

            with torch.no_grad():
                final_values[
                    step, torch.arange(args.num_envs, device=device)[done_mask]
                ] = agent.get_value(infos["final_observation"]).view(-1)

        ts.append(
            tensordict.TensorDict._new_unsafe(
                obs=obs,
                # cleanrl ppo examples associate the done with the previous obs (not the done resulting from action)
                dones=done,
                vals=value.flatten(),
                actions=action,
                logprobs=logprob,
                rewards=reward,
                batch_size=(args.num_envs,),
            )
        )
        # NOTE (stao): change here for gpu env
        obs = next_obs = next_obs
        done = next_done

    # NOTE (stao): need to do .to(device) i think? otherwise container.device is None, not sure if this affects anything
    container = torch.stack(ts, 0).to(device)
    return next_obs, done, container, final_values


def update(obs, actions, logprobs, advantages, returns, vals):
    """Optimized update function."""
    optimizer.zero_grad()
    _, newlogprob, entropy, newvalue = agent.get_action_and_value(obs, actions)
    logratio = newlogprob - logprobs
    ratio = logratio.exp()

    with torch.no_grad():
        # calculate approx_kl http://joschu.net/blog/kl-approx.html
        old_approx_kl = (-logratio).mean()
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean()

    if args.norm_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Policy loss
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    # Value loss
    newvalue = newvalue.view(-1)
    if args.clip_vloss:
        v_loss_unclipped = (newvalue - returns) ** 2
        v_clipped = vals + torch.clamp(
            newvalue - vals,
            -args.clip_coef,
            args.clip_coef,
        )
        v_loss_clipped = (v_clipped - returns) ** 2
        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
        v_loss = 0.5 * v_loss_max.mean()
    else:
        v_loss = 0.5 * ((newvalue - returns) ** 2).mean()

    entropy_loss = entropy.mean()
    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

    loss.backward()
    gn = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
    optimizer.step()

    return (
        approx_kl,
        v_loss.detach(),
        pg_loss.detach(),
        entropy_loss.detach(),
        old_approx_kl,
        clipfrac,
        gn,
    )


# TensorDict wrapper for update function
update = tensordict.nn.TensorDictModule(
    update,
    in_keys=["obs", "actions", "logprobs", "advantages", "returns", "vals"],
    out_keys=[
        "approx_kl",
        "v_loss",
        "pg_loss",
        "entropy_loss",
        "old_approx_kl",
        "clipfrac",
        "gn",
    ],
)


if __name__ == "__main__":
    args = tyro.cli(Args)

    # Calculate runtime parameters
    batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = batch_size // args.num_minibatches
    args.batch_size = args.num_minibatches * args.minibatch_size
    args.num_iterations = args.total_timesteps // args.batch_size

    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Environment setup
    env_kwargs = dict(
        obs_mode="rgb", render_mode=args.render_mode, sim_backend="physx_cuda"
    )
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode
    if args.robot_uids is not None:
        env_kwargs["robot_uids"] = args.robot_uids

    # Create environments conditionally based on expert mode
    if args.expert_type != "none":
        print(
            f"Creating Expert+Residual environments with expert type: {args.expert_type}"
        )
        if create_expert_residual_envs is None:
            raise ImportError(
                "Expert+Residual wrapper not available. Please ensure util.py is importable."
            )
        envs, eval_envs = create_expert_residual_envs(args, env_kwargs, device)
    else:
        # Regular environments
        envs = gym.make(
            args.env_id,
            num_envs=args.num_envs if not args.evaluate else 1,
            reconfiguration_freq=args.reconfiguration_freq,
            **env_kwargs,
        )
        eval_envs = gym.make(
            args.env_id,
            num_envs=args.num_eval_envs,
            reconfiguration_freq=args.eval_reconfiguration_freq,
            human_render_camera_configs=dict(shader_pack="default"),
            **env_kwargs,
        )

        # Apply observation and action space wrappers for regular envs only
        envs = FlattenRGBDObservationWrapper(envs, rgb=True, state=args.include_state)
        eval_envs = FlattenRGBDObservationWrapper(
            eval_envs, rgb=True, state=args.include_state
        )

        if isinstance(envs.action_space, gym.spaces.Dict):
            envs = FlattenActionSpaceWrapper(envs)
            eval_envs = FlattenActionSpaceWrapper(eval_envs)

    # Recording setup
    if args.capture_video or args.save_trajectory:
        eval_output_dir = f"runs/{run_name}/videos"
        if args.evaluate:
            eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos"
        print(f"Saving eval trajectories/videos to {eval_output_dir}")

        if args.save_train_video_freq is not None:
            save_video_trigger = (
                lambda x: (x // args.num_steps) % args.save_train_video_freq == 0
            )
            envs = RecordEpisode(
                envs,
                output_dir=f"runs/{run_name}/train_videos",
                save_trajectory=False,
                save_video_trigger=save_video_trigger,
                max_steps_per_video=args.num_steps,
                video_fps=30,
            )
        eval_envs = RecordEpisode(
            eval_envs,
            output_dir=eval_output_dir,
            save_trajectory=args.save_trajectory,
            save_video=args.capture_video,
            trajectory_name="trajectory",
            max_steps_per_video=args.num_eval_steps,
            video_fps=30,
        )

    # Vector environment setup
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
        "only continuous action space is supported"
    )

    max_episode_steps = gym_utils.find_max_episode_steps_value(envs._env)
    logger = None

    if not args.evaluate:
        print("Running training")
        if args.track:
            import wandb

            config = vars(args)
            config["env_cfg"] = dict(
                **env_kwargs,
                num_envs=args.num_envs,
                env_id=args.env_id,
                reward_mode="normalized_dense",
                env_horizon=max_episode_steps,
                partial_reset=args.partial_reset,
            )
            config["eval_env_cfg"] = dict(
                **env_kwargs,
                num_envs=args.num_eval_envs,
                env_id=args.env_id,
                reward_mode="normalized_dense",
                env_horizon=max_episode_steps,
                partial_reset=False,
            )
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=False,
                config=config,
                name=run_name,
                save_code=True,
                group=args.wandb_group,
                tags=[
                    "ppo",
                    "rgb",
                    "vision",
                    "fast",
                    f"GPU:{torch.cuda.get_device_name()}",
                ],
            )
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        logger = Logger(log_wandb=args.track, tensorboard=writer)
    else:
        print("Running evaluation")

    # Get environment info
    n_act = math.prod(envs.single_action_space.shape)
    sample_obs = envs.reset()[0]

    # Agent setup
    agent = Agent(n_act, sample_obs, device=device)
    if args.checkpoint:
        agent.load_state_dict(torch.load(args.checkpoint))

    # Make a version of agent with detached params for inference
    agent_inference = Agent(n_act, sample_obs, device=device)
    agent_inference_p = from_module(agent).data
    agent_inference_p.to_module(agent_inference)

    # Optimizer setup
    optimizer = optim.Adam(
        agent.parameters(),
        lr=torch.tensor(args.learning_rate, device=device),
        eps=1e-5,
        capturable=args.cudagraphs and not args.compile,
    )

    # Define step function
    def step_func(
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # NOTE (stao): change here for gpu env
        next_obs, reward, terminations, truncations, info = envs.step(action)
        next_done = torch.logical_or(terminations, truncations)
        return next_obs, reward, next_done, info

    # Define executables
    policy = agent_inference.get_action_and_value
    get_value = agent_inference.get_value

    # Compile optimizations
    if args.compile:
        print("🔥 Compiling policy and training functions...")
        policy = torch.compile(policy)
        gae = torch.compile(gae, fullgraph=True)
        update = torch.compile(update)
        print("✅ Compilation complete!")

    if args.cudagraphs:
        print("🚀 Enabling CUDA graphs...")
        policy = CudaGraphModule(policy)
        gae = CudaGraphModule(gae)
        update = CudaGraphModule(update)
        print("✅ CUDA graphs enabled!")

    # Training loop
    global_step = 0
    start_time = time.time()
    container_local = None
    next_obs = envs.reset()[0]
    next_done = torch.zeros(args.num_envs, device=device, dtype=torch.bool)
    pbar = tqdm.tqdm(range(1, args.num_iterations + 1))

    cumulative_times = defaultdict(float)

    for iteration in pbar:
        agent.eval()

        # Evaluation
        if iteration % args.eval_freq == 1:
            stime = time.perf_counter()
            eval_obs, _ = eval_envs.reset()
            eval_metrics = defaultdict(list)
            num_episodes = 0

            for _ in range(args.num_eval_steps):
                with torch.no_grad():
                    (
                        eval_obs,
                        eval_rew,
                        eval_terminations,
                        eval_truncations,
                        eval_infos,
                    ) = eval_envs.step(agent.actor_mean(agent.get_features(eval_obs)))

                    if "final_info" in eval_infos:
                        mask = eval_infos["_final_info"]
                        num_episodes += mask.sum()
                        for k, v in eval_infos["final_info"]["episode"].items():
                            eval_metrics[k].append(v)

            eval_metrics_mean = {}
            for k, v in eval_metrics.items():
                mean = torch.stack(v).float().mean()
                eval_metrics_mean[k] = mean
                if logger is not None:
                    logger.add_scalar(f"eval/{k}", mean, global_step)

            pbar.set_description(
                f"success_once: {eval_metrics_mean['success_once']:.2f}, "
                f"return: {eval_metrics_mean['return']:.2f}"
            )

            if logger is not None:
                eval_time = time.perf_counter() - stime
                cumulative_times["eval_time"] += eval_time
                logger.add_scalar("time/eval_time", eval_time, global_step)

            if args.evaluate:
                break

        # Save model
        if args.save_model and iteration % args.eval_freq == 1:
            model_path = f"runs/{run_name}/ckpt_{iteration}.pt"
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")

        # Learning rate annealing
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"].copy_(lrnow)

        # Training step
        agent.train()
        torch.compiler.cudagraph_mark_step_begin()

        rollout_time = time.perf_counter()
        next_obs, next_done, container, final_values = rollout(next_obs, next_done)
        rollout_time = time.perf_counter() - rollout_time
        cumulative_times["rollout_time"] += rollout_time
        global_step += container.numel()

        update_time = time.perf_counter()
        container = gae(next_obs, next_done, container, final_values)
        container_flat = container.view(-1)

        # Policy optimization
        clipfracs = []
        for epoch in range(args.update_epochs):
            b_inds = torch.randperm(container_flat.shape[0], device=device).split(
                args.minibatch_size
            )
            for b in b_inds:
                container_local = container_flat[b]

                out = update(container_local, tensordict_out=tensordict.TensorDict())
                clipfracs.append(out["clipfrac"])

                if args.target_kl is not None and out["approx_kl"] > args.target_kl:
                    break
            else:
                continue
            break

        update_time = time.perf_counter() - update_time
        cumulative_times["update_time"] += update_time

        # Logging
        if logger is not None:
            logger.add_scalar(
                "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
            )
            logger.add_scalar("losses/value_loss", out["v_loss"].item(), global_step)
            logger.add_scalar("losses/policy_loss", out["pg_loss"].item(), global_step)
            logger.add_scalar("losses/entropy", out["entropy_loss"].item(), global_step)
            logger.add_scalar(
                "losses/old_approx_kl", out["old_approx_kl"].item(), global_step
            )
            logger.add_scalar("losses/approx_kl", out["approx_kl"].item(), global_step)
            logger.add_scalar(
                "losses/clipfrac",
                torch.stack(clipfracs).mean().cpu().item(),
                global_step,
            )
            logger.add_scalar(
                "charts/SPS", int(global_step / (time.time() - start_time)), global_step
            )
            logger.add_scalar("time/step", global_step, global_step)
            logger.add_scalar("time/update_time", update_time, global_step)
            logger.add_scalar("time/rollout_time", rollout_time, global_step)
            logger.add_scalar(
                "time/rollout_fps",
                args.num_envs * args.num_steps / rollout_time,
                global_step,
            )
            for k, v in cumulative_times.items():
                logger.add_scalar(f"time/total_{k}", v, global_step)
            logger.add_scalar(
                "time/total_rollout+update_time",
                cumulative_times["rollout_time"] + cumulative_times["update_time"],
                global_step,
            )

    # Final model save
    if not args.evaluate:
        if args.save_model:
            model_path = f"runs/{run_name}/final_ckpt.pt"
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")
        logger.close()

    envs.close()
    eval_envs.close()
