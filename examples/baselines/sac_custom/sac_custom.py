"""
Copyright 2025 Zordi, Inc. All rights reserved.

SAC with custom multimodal encoders for ManiSkill environments.

python sac_custom.py --env_id="PickCube-v1" --obs_mode="rgb" \
  --num_envs=32 --utd=0.5 --buffer_size=300_000 \
  --control-mode="pd_ee_delta_pos" --camera_width=64 --camera_height=64 \
  --total_timesteps=1_000_000 --eval_freq=10_000
"""

import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import tqdm
import tyro
from multimodal_encoders import DINOv2Encoder, EfficientNetB0Encoder, MultimodalEncoder, SimpleConvEncoder
from torch import nn, optim

# AMP for mixed-precision training
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.tensorboard.writer import SummaryWriter

from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv


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
    wandb_group: str = "SAC_Custom"
    """the group of the run for wandb"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_trajectory: bool = False
    """whether to save trajectory data into the `videos` folder"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    evaluate: bool = False
    """if toggled, only runs evaluation with the given model checkpoint and saves the evaluation trajectories"""
    checkpoint: Optional[str] = None
    """path to a pretrained checkpoint file to start evaluation/training from"""
    log_freq: int = 1_000
    """logging frequency in terms of environment steps"""

    # Environment specific arguments
    env_id: str = "PickBox-v1"
    """the id of the environment"""
    obs_mode: str = "rgb"
    """the observation mode to use"""
    include_state: bool = True
    """whether to include the state in the observation"""
    env_vectorization: str = "gpu"
    """the type of environment vectorization to use"""
    num_envs: int = 16
    """the number of parallel environments"""
    num_eval_envs: int = 8
    """the number of parallel evaluation environments"""
    partial_reset: bool = False
    """whether to let parallel environments reset upon termination instead of truncation"""
    eval_partial_reset: bool = False
    """whether to let parallel evaluation environments reset upon termination instead of truncation"""
    num_steps: int = 100
    """the number of steps to run in each environment per policy rollout"""
    num_eval_steps: int = 100
    """the number of steps to run in each evaluation environment during evaluation"""
    reconfiguration_freq: Optional[int] = None
    """how often to reconfigure the environment during training"""
    eval_reconfiguration_freq: Optional[int] = 1
    """for benchmarking purposes we want to reconfigure the eval environment each reset to ensure objects are randomized in some tasks"""
    eval_freq: int = 10_000
    """evaluation frequency in terms of iterations"""
    save_train_video_freq: Optional[int] = None
    """frequency to save training videos in terms of iterations"""
    control_mode: Optional[str] = "pd_joint_delta_pos"
    """the control mode to use for the environment"""
    render_mode: str = "all"
    """the environment rendering mode"""

    # Algorithm specific arguments
    total_timesteps: int = 1_000_000
    """total timesteps of the experiments"""
    buffer_size: int = 12_000
    """the replay memory buffer size"""
    buffer_device: str = "cuda"
    """where the replay buffer is stored. Can be 'cpu' or 'cuda' for GPU"""
    gamma: float = 0.8
    """the discount factor gamma"""
    tau: float = 0.01
    """target smoothing coefficient"""
    batch_size: int = 64
    """the batch size of sample from the replay memory"""
    learning_starts: int = 4_000
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 3e-4
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 1
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    training_freq: int = 64
    """training frequency (in steps)"""
    utd: float = 0.25
    """update to data ratio"""
    bootstrap_at_done: str = "always"
    """the bootstrap method to use when a done signal is received. Can be 'always' or 'never'"""
    camera_width: Optional[int] = 112
    """the width of the camera image. If none it will use the default the environment specifies"""
    camera_height: Optional[int] = 112
    """the height of the camera image. If none it will use the default the environment specifies."""

    # Custom encoder arguments
    use_dinov2: bool = False
    """whether to use DINOv2 encoder for images (otherwise uses simple CNN)"""
    use_efficientnet: bool = True
    """whether to use EfficientNet-B0 encoder for images (trainable, lighter than DINOv2)"""

    # to be filled in runtime
    grad_steps_per_iteration: int = 0
    """the number of gradient updates per iteration"""
    steps_per_env: int = 0
    """the number of steps each parallel env takes per iteration"""


class DictArray(object):
    def __init__(self, buffer_shape, element_space, data_dict=None, device=None):
        self.buffer_shape = buffer_shape
        if data_dict:
            self.data = data_dict
        else:
            assert isinstance(element_space, gym.spaces.Dict)
            self.data = {}
            for k, v in element_space.items():
                if isinstance(v, gym.spaces.Dict):
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
                    self.data[k] = torch.zeros(buffer_shape + v.shape, dtype=dtype, device=device)

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


@dataclass
class ReplayBufferSample:
    obs: dict
    next_obs: dict
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    def __init__(self, env, num_envs: int, buffer_size: int, storage_device: torch.device, sample_device: torch.device):
        self.buffer_size = buffer_size
        self.pos = 0
        self.full = False
        self.num_envs = num_envs
        self.storage_device = storage_device
        self.sample_device = sample_device
        self.per_env_buffer_size = buffer_size // num_envs

        self.obs = DictArray((self.per_env_buffer_size, num_envs), env.single_observation_space, device=storage_device)
        self.next_obs = DictArray(
            (self.per_env_buffer_size, num_envs), env.single_observation_space, device=storage_device
        )
        self.actions = torch.zeros(
            (self.per_env_buffer_size, num_envs) + env.single_action_space.shape, device=storage_device
        )
        self.rewards = torch.zeros((self.per_env_buffer_size, num_envs), device=storage_device)
        self.dones = torch.zeros((self.per_env_buffer_size, num_envs), device=storage_device)

    def add(self, obs: dict, next_obs: dict, action: torch.Tensor, reward: torch.Tensor, done: torch.Tensor):
        if self.storage_device == torch.device("cpu"):
            obs = {k: v.cpu() for k, v in obs.items()}
            next_obs = {k: v.cpu() for k, v in next_obs.items()}
            action = action.cpu()
            reward = reward.cpu()
            done = done.cpu()

        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done

        self.pos += 1
        if self.pos == self.per_env_buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int):
        if self.full:
            batch_inds = torch.randint(0, self.per_env_buffer_size, size=(batch_size,))
        else:
            batch_inds = torch.randint(0, self.pos, size=(batch_size,))
        env_inds = torch.randint(0, self.num_envs, size=(batch_size,))

        obs_sample = self.obs[batch_inds, env_inds]
        next_obs_sample = self.next_obs[batch_inds, env_inds]
        obs_sample = {k: v.to(self.sample_device) for k, v in obs_sample.items()}
        next_obs_sample = {k: v.to(self.sample_device) for k, v in next_obs_sample.items()}

        return ReplayBufferSample(
            obs=obs_sample,
            next_obs=next_obs_sample,
            actions=self.actions[batch_inds, env_inds].to(self.sample_device),
            rewards=self.rewards[batch_inds, env_inds].to(self.sample_device),
            dones=self.dones[batch_inds, env_inds].to(self.sample_device),
        )


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class SACPolicyNetwork(nn.Module):
    """SAC policy network with multimodal encoder."""

    def __init__(self, envs, encoder: MultimodalEncoder):
        super().__init__()
        self.encoder = encoder
        action_dim = np.prod(envs.single_action_space.shape)

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
        self.register_buffer(
            "action_scale", torch.FloatTensor((envs.single_action_space.high - envs.single_action_space.low) / 2.0)
        )
        self.register_buffer(
            "action_bias", torch.FloatTensor((envs.single_action_space.high + envs.single_action_space.low) / 2.0)
        )

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

    def __init__(self, envs, encoder: MultimodalEncoder):
        super().__init__()
        self.encoder = encoder
        action_dim = np.prod(envs.single_action_space.shape)

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


class Logger:
    def __init__(self, log_wandb=False, tensorboard: Optional[SummaryWriter] = None) -> None:
        self.writer = tensorboard
        self.log_wandb = log_wandb

    def add_scalar(self, tag, scalar_value, step):
        if self.log_wandb:
            import wandb

            wandb.log({tag: scalar_value}, step=step)
        if self.writer is not None:
            self.writer.add_scalar(tag, scalar_value, step)

    def close(self):
        if self.writer is not None:
            self.writer.close()


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.grad_steps_per_iteration = int(args.training_freq * args.utd)
    args.steps_per_env = args.training_freq // args.num_envs
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
    env_kwargs = dict(obs_mode=args.obs_mode, render_mode=args.render_mode, sim_backend="gpu", sensor_configs=dict())
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode
    if args.camera_width is not None:
        env_kwargs["sensor_configs"]["width"] = args.camera_width
    if args.camera_height is not None:
        env_kwargs["sensor_configs"]["height"] = args.camera_height

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

    # Apply wrappers
    envs = FlattenRGBDObservationWrapper(envs, rgb=True, depth=False, state=args.include_state)
    eval_envs = FlattenRGBDObservationWrapper(eval_envs, rgb=True, depth=False, state=args.include_state)

    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)

    if args.capture_video or args.save_trajectory:
        eval_output_dir = f"runs/{run_name}/videos"
        if args.evaluate:
            eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos"
        print(f"Saving eval trajectories/videos to {eval_output_dir}")
        if args.save_train_video_freq is not None:
            save_video_trigger = lambda x: (x // args.num_steps) % args.save_train_video_freq == 0
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

    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(
        eval_envs, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, record_metrics=True
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

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
                tags=["sac", "custom_encoder"],
            )
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        logger = Logger(log_wandb=args.track, tensorboard=writer)
    else:
        print("Running evaluation")

    # Create replay buffer
    rb = ReplayBuffer(
        env=envs,
        num_envs=args.num_envs,
        buffer_size=args.buffer_size,
        storage_device=torch.device(args.buffer_device),
        sample_device=device,
    )

    # Initialize environment
    obs, info = envs.reset(seed=args.seed)
    eval_obs, _ = eval_envs.reset(seed=args.seed)

    # Get observation dimensions
    state_dim = envs.single_observation_space["state"].shape[0]
    rgb_shape = obs["rgb"].shape
    if len(rgb_shape) == 4:  # [B, H, W, C]
        image_size = (rgb_shape[1], rgb_shape[2])
        image_channels = rgb_shape[3]
    else:
        raise ValueError(f"Unexpected RGB shape: {rgb_shape}")

    print(f"State dim: {state_dim}")
    print(f"Image size: {image_size}")
    print(f"Image channels: {image_channels}")
    print(f"Using DINOv2: {args.use_dinov2}")

    # ---------------------------------------------------------------------
    # Build ONE shared image encoder (heavy part) and reuse it across all
    # actor / critic networks to save memory and initialization time.
    # ---------------------------------------------------------------------
    if args.use_efficientnet:
        if image_channels % 3 == 0:
            # Multi-camera setup: use EfficientNet-B0 for processing each camera
            shared_image_encoder = EfficientNetB0Encoder(3, 256).to(device)  # Process 3 channels at a time
            print(f"📷 Multi-camera setup detected: {image_channels // 3} cameras, using EfficientNet-B0 for each")
        else:
            print(f"⚠️ Warning: EfficientNet-B0 works best with 3 channels, but got {image_channels}. Using anyway.")
            shared_image_encoder = EfficientNetB0Encoder(image_channels, 256).to(device)
    elif args.use_dinov2:
        if image_channels % 3 == 0:
            # Multi-camera setup: use DINOv2 for processing each camera
            shared_image_encoder = DINOv2Encoder(3, 256).to(device)  # Process 3 channels at a time
            print(f"📷 Multi-camera setup detected: {image_channels // 3} cameras, using DINOv2 for each")
        else:
            print(
                f"⚠️ Warning: DINOv2 requires channels divisible by 3, but got {image_channels}. Using SimpleConvEncoder instead."
            )
            shared_image_encoder = SimpleConvEncoder(image_channels, 256, image_size).to(device)
    else:
        shared_image_encoder = SimpleConvEncoder(image_channels, 256, image_size).to(device)

    # Factory to create a multimodal encoder that reuses the shared image encoder
    def create_encoder():
        return MultimodalEncoder(
            state_dim=state_dim,
            image_channels=image_channels,
            image_size=image_size,
            output_dim=512,
            use_dinov2=args.use_dinov2,
            use_efficientnet=args.use_efficientnet,
            image_encoder=shared_image_encoder,
        )

    actor = SACPolicyNetwork(envs, create_encoder()).to(device)
    qf1 = SACQNetwork(envs, create_encoder()).to(device)
    qf2 = SACQNetwork(envs, create_encoder()).to(device)
    qf1_target = SACQNetwork(envs, create_encoder()).to(device)
    qf2_target = SACQNetwork(envs, create_encoder()).to(device)

    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint)
        actor.load_state_dict(ckpt["actor"])
        qf1.load_state_dict(ckpt["qf1"])
        qf2.load_state_dict(ckpt["qf2"])

    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    # Optimizers
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # GradScaler for AMP (only active if CUDA is available)
    scaler = GradScaler(enabled=device.type == "cuda")

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    global_step = 0
    global_update = 0
    learning_has_started = False

    global_steps_per_iteration = args.num_envs * args.steps_per_env
    pbar = tqdm.tqdm(range(args.total_timesteps))
    cumulative_times = defaultdict(float)

    while global_step < args.total_timesteps:
        if args.eval_freq > 0 and (global_step - args.training_freq) // args.eval_freq < global_step // args.eval_freq:
            # Evaluate
            actor.eval()
            stime = time.perf_counter()
            eval_obs, _ = eval_envs.reset()
            eval_metrics = defaultdict(list)
            num_episodes = 0
            for _ in range(args.num_eval_steps):
                with torch.no_grad():
                    eval_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(
                        actor.get_eval_action(eval_obs)
                    )
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
                f"success_once: {eval_metrics_mean.get('success_once', 0):.2f}, return: {eval_metrics_mean.get('return', 0):.2f}"
            )
            if logger is not None:
                eval_time = time.perf_counter() - stime
                cumulative_times["eval_time"] += eval_time
                logger.add_scalar("time/eval_time", eval_time, global_step)
            if args.evaluate:
                break
            actor.train()

            if args.save_model:
                model_path = f"runs/{run_name}/ckpt_{global_step}.pt"
                torch.save(
                    {
                        "actor": actor.state_dict(),
                        "qf1": qf1_target.state_dict(),
                        "qf2": qf2_target.state_dict(),
                        "log_alpha": log_alpha if args.autotune else None,
                    },
                    model_path,
                )
                print(f"model saved to {model_path}")

        # Collect samples from environments
        rollout_time = time.perf_counter()
        for local_step in range(args.steps_per_env):
            global_step += args.num_envs

            # Action selection
            if not learning_has_started:
                actions = torch.tensor(envs.action_space.sample(), dtype=torch.float32, device=device)
            else:
                actions, _, _ = actor.get_action(obs)
                actions = actions.detach()

            # Execute environment step
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)
            real_next_obs = {k: v.clone() for k, v in next_obs.items()}

            if args.bootstrap_at_done == "never":
                need_final_obs = torch.ones_like(terminations, dtype=torch.bool)
                stop_bootstrap = truncations | terminations
            elif args.bootstrap_at_done == "always":
                need_final_obs = truncations | terminations
                stop_bootstrap = torch.zeros_like(terminations, dtype=torch.bool)
            else:  # bootstrap at truncated
                need_final_obs = truncations & (~terminations)
                stop_bootstrap = terminations

            if "final_info" in infos:
                final_info = infos["final_info"]
                done_mask = infos["_final_info"]
                for k in real_next_obs.keys():
                    real_next_obs[k][need_final_obs] = infos["final_observation"][k][need_final_obs].clone()
                for k, v in final_info["episode"].items():
                    if logger is not None:
                        logger.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)

            rb.add(obs, real_next_obs, actions, rewards, stop_bootstrap)
            obs = next_obs

        rollout_time = time.perf_counter() - rollout_time
        cumulative_times["rollout_time"] += rollout_time
        pbar.update(args.num_envs * args.steps_per_env)

        # Training
        if global_step < args.learning_starts:
            continue

        update_time = time.perf_counter()
        learning_has_started = True

        # Initialize logging variables
        qf1_a_values = qf2_a_values = qf1_loss = qf2_loss = qf_loss = actor_loss = alpha_loss = None

        for local_update in range(args.grad_steps_per_iteration):
            global_update += 1
            data = rb.sample(args.batch_size)

            # Update value networks
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_obs)
                # Ensure log probabilities have shape (batch_size,) for correct broadcasting
                next_state_log_pi = next_state_log_pi.view(-1)
                qf1_next_target = qf1_target(data.next_obs, next_state_actions)
                qf2_next_target = qf2_target(data.next_obs, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (
                    min_qf_next_target
                ).view(-1)

            # Mixed-precision forward & loss
            with autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                qf1_a_values = qf1(data.obs, data.actions).view(-1)
                qf2_a_values = qf2(data.obs, data.actions).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad(set_to_none=True)
            scaler.scale(qf_loss).backward()
            scaler.step(q_optimizer)
            scaler.update()

            # Update policy network
            if global_update % args.policy_frequency == 0:
                with autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                    pi, log_pi, _ = actor.get_action(data.obs)
                    # Flatten log_pi to (batch_size,) to match Q-value shapes
                    log_pi = log_pi.view(-1)
                    qf1_pi = qf1(data.obs, pi)
                    qf2_pi = qf2(data.obs, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                actor_optimizer.zero_grad(set_to_none=True)
                scaler.scale(actor_loss).backward()
                scaler.step(actor_optimizer)
                scaler.update()
                if args.autotune:
                    with torch.no_grad():
                        _, log_pi, _ = actor.get_action(data.obs)
                    # Flatten log_pi for scalar loss computation
                    log_pi = log_pi.view(-1)
                    alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                    a_optimizer.zero_grad(set_to_none=True)
                    scaler.scale(alpha_loss).backward()
                    scaler.step(a_optimizer)
                    scaler.update()
                    alpha = log_alpha.exp().item()

            # Update target networks
            if global_update % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

        update_time = time.perf_counter() - update_time
        cumulative_times["update_time"] += update_time

        # Log training data
        if (global_step - args.training_freq) // args.log_freq < global_step // args.log_freq:
            if logger is not None and qf1_a_values is not None:
                logger.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                logger.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                logger.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                logger.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                logger.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                if actor_loss is not None:
                    logger.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                logger.add_scalar("losses/alpha", alpha, global_step)
                logger.add_scalar("time/update_time", update_time, global_step)
                logger.add_scalar("time/rollout_time", rollout_time, global_step)
                logger.add_scalar("time/rollout_fps", global_steps_per_iteration / rollout_time, global_step)
                for k, v in cumulative_times.items():
                    logger.add_scalar(f"time/total_{k}", v, global_step)
                logger.add_scalar(
                    "time/total_rollout+update_time",
                    cumulative_times["rollout_time"] + cumulative_times["update_time"],
                    global_step,
                )
                if args.autotune and alpha_loss is not None:
                    logger.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    if not args.evaluate and args.save_model:
        model_path = f"runs/{run_name}/final_ckpt.pt"
        torch.save(
            {
                "actor": actor.state_dict(),
                "qf1": qf1_target.state_dict(),
                "qf2": qf2_target.state_dict(),
                "log_alpha": log_alpha if args.autotune else None,
            },
            model_path,
        )
        print(f"model saved to {model_path}")
        if logger is not None:
            logger.close()
    envs.close()
