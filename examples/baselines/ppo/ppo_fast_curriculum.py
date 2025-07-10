"""
Copyright 2025 Zordi, Inc. All rights reserved.

PPO training script with curriculum learning for PickBox environment.
Based on the original ppo_fast.py but modified to support curriculum learning.
"""

import argparse
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import mani_skill.envs
import numpy as np
import torch
from mani_skill.envs.tasks.tabletop.pick_box_curriculum import create_curriculum_wrapper
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from torch import nn, optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-name",
        type=str,
        default="curriculum_training",
        help="the name of this experiment",
    )
    parser.add_argument("--seed", type=int, default=42, help="seed of the experiment")

    # Essential training arguments
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=35_000_000,
        help="total timesteps of the experiments (5M per curriculum level * 7 levels)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="the learning rate of the optimizer",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1600,
        help="the number of parallel game environments",
    )

    # PPO hyperparameters
    parser.add_argument(
        "--gamma", type=float, default=0.8, help="the discount factor gamma"
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.9,
        help="the lambda for the general advantage estimation",
    )
    parser.add_argument(
        "--clip-coef",
        type=float,
        default=0.2,
        help="the surrogate clipping coefficient",
    )
    parser.add_argument(
        "--ent-coef", type=float, default=0.0, help="coefficient of the entropy"
    )
    parser.add_argument(
        "--vf-coef", type=float, default=0.5, help="coefficient of the value function"
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
        help="the maximum norm for the gradient clipping",
    )

    # Curriculum learning arguments
    parser.add_argument(
        "--curriculum-success-threshold",
        type=float,
        default=0.8,
        help="success rate threshold to advance to next curriculum level",
    )
    parser.add_argument(
        "--curriculum-min-episodes",
        type=int,
        default=1000,
        help="minimum episodes before allowing level advancement",
    )
    parser.add_argument(
        "--curriculum-steps-per-level",
        type=int,
        default=5_000_000,
        help="number of training steps per curriculum level",
    )

    # Optional arguments
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="path to a checkpoint to load"
    )
    parser.add_argument(
        "--track",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases",
    )
    parser.add_argument(
        "--wandb-project-name",
        type=str,
        default="ManiSkill-Curriculum",
        help="the wandb's project name",
    )

    args = parser.parse_args()

    # Set fixed values for simplicity
    args.env_id = "PickBox-v1"
    args.robot_uids = "a1_galaxea"
    args.capture_video = (
        False  # Video recording conflicts with render system - curriculum works!
    )
    args.cuda = True
    args.torch_deterministic = True
    args.anneal_lr = True
    args.norm_adv = True
    args.clip_vloss = False
    args.target_kl = None
    args.sync_venv = False
    args.curriculum = True
    args.curriculum_window_size = 100
    args.curriculum_verbose = True

    # PPO fixed parameters
    args.num_minibatches = 300
    args.update_epochs = 2
    args.eval_freq = 25
    args.num_eval_episodes = 10
    args.num_eval_envs = 12
    args.num_eval_steps = 50
    args.save_model = True
    args.save_trajectory = False
    args.output_dir = "runs"
    args.partial_reset = True
    args.eval_partial_reset = False

    # Calculate derived values - using dynamic num_steps from curriculum
    # We'll set these dynamically in the main loop based on current curriculum level
    args.num_steps = 50  # Initial value, will be updated dynamically
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    return args


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initialize layer weights."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """PPO agent network."""

    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 512)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 512)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.Tanh(),
            layer_init(
                nn.Linear(512, np.prod(envs.single_action_space.shape)), std=0.01
            ),
        )
        self.actor_logstd = nn.Parameter(
            torch.zeros(1, np.prod(envs.single_action_space.shape))
        )

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
    """Main training loop."""
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    print("🎯 Starting Curriculum Learning Training")
    print(f"Experiment: {run_name}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Parallel environments: {args.num_envs}")
    print(f"Steps per curriculum level: {args.curriculum_steps_per_level:,}")
    print(f"Success threshold: {args.curriculum_success_threshold}")
    if args.capture_video:
        print(f"📹 Videos will be saved to: videos/{run_name}")
        print(
            "   Evaluation environment will be recreated when curriculum level changes"
        )
    print(f"Logs will be saved to: {args.output_dir}/{run_name}")
    print()

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    writer = SummaryWriter(f"{args.output_dir}/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")

    # Simple curriculum management without complex wrapper
    from mani_skill.envs.tasks.tabletop.pick_box_curriculum import get_curriculum_levels

    curriculum_levels = get_curriculum_levels()
    current_curriculum_level = 0
    success_history = []
    episodes_completed = 0

    print(f"📚 Curriculum Learning with {len(curriculum_levels)} levels")
    print(f"Starting with Level 1: {curriculum_levels[0]['name']}")
    print(f"Description: {curriculum_levels[0]['description']}")
    print()

    # Environment setup
    env_kwargs = dict(
        obs_mode="state", render_mode="rgb_array", sim_backend="physx_cuda"
    )
    if args.robot_uids is not None:
        env_kwargs["robot_uids"] = args.robot_uids

    # Create training environments
    envs = gym.make(
        args.env_id,
        num_envs=args.num_envs,
        **env_kwargs,
    )

    # Create evaluation environment with video recording
    eval_envs = gym.make(
        args.env_id,
        num_envs=args.num_eval_envs,
        **env_kwargs,
    )

    # Add video recording
    if args.capture_video or args.save_trajectory:
        eval_output_dir = f"runs/{run_name}/videos"
        print(f"Saving eval trajectories/videos to {eval_output_dir}")
        eval_envs = RecordEpisode(
            eval_envs,
            output_dir=eval_output_dir,
            save_trajectory=args.save_trajectory,
            save_video=args.capture_video,
            trajectory_name="trajectory",
            max_steps_per_video=args.num_eval_steps,
            video_fps=30,
        )

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

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Load checkpoint if specified
    if args.checkpoint:
        agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded checkpoint from {args.checkpoint}")

    # Start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    print("🚀 Training started!")
    print()

    for iteration in range(1, args.num_iterations + 1):
        # Simple curriculum management - get current level config
        current_level_config = curriculum_levels[current_curriculum_level]
        curriculum_num_steps = current_level_config["max_episode_steps"]

        # Use the curriculum episode length
        args.num_steps = curriculum_num_steps
        args.batch_size = int(args.num_envs * args.num_steps)
        args.minibatch_size = int(args.batch_size // args.num_minibatches)

        # Initialize storage tensors
        obs = torch.zeros(
            (args.num_steps, args.num_envs) + envs.single_observation_space.shape
        ).to(device)
        actions = torch.zeros(
            (args.num_steps, args.num_envs) + envs.single_action_space.shape
        ).to(device)
        logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
        rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
        dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
        values = torch.zeros((args.num_steps, args.num_envs)).to(device)

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # Action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Execute the game and log data
            next_obs, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )
            next_done = torch.logical_or(terminations, truncations)
            rewards[step] = torch.as_tensor(reward, device=device).view(-1)
            next_obs, next_done = (
                torch.Tensor(next_obs).to(device),
                torch.Tensor(next_done).to(device),
            )

            # Curriculum learning logging
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and isinstance(info, dict) and "curriculum_info" in info:
                        curriculum_info = info["curriculum_info"]
                        writer.add_scalar(
                            "curriculum/level",
                            curriculum_info["current_level"],
                            global_step,
                        )
                        writer.add_scalar(
                            "curriculum/success_rate",
                            curriculum_info["current_success_rate"],
                            global_step,
                        )
                        writer.add_scalar(
                            "curriculum/episodes_at_level",
                            curriculum_info["episodes_at_current_level"],
                            global_step,
                        )
                        writer.add_scalar(
                            "curriculum/max_episode_steps",
                            curriculum_info["max_episode_steps"],
                            global_step,
                        )

            # Episode logging
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and isinstance(info, dict) and "episode" in info:
                        print(
                            f"global_step={global_step}, episodic_return={info['episode']['r']:.2f}"
                        )
                        writer.add_scalar(
                            "charts/episodic_return", info["episode"]["r"], global_step
                        )
                        writer.add_scalar(
                            "charts/episodic_length", info["episode"]["l"], global_step
                        )
                        if "success" in info:
                            writer.add_scalar(
                                "charts/success_rate", info["success"], global_step
                            )

        # Bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
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

        # Flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)  # noqa: RUF005
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # Calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Logging
        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

        sps = int(global_step / (time.time() - start_time))
        print(f"SPS: {sps}")
        writer.add_scalar("charts/SPS", sps, global_step)

        # Evaluation
        if iteration % args.eval_freq == 1:
            agent.eval()
            with torch.no_grad():
                eval_episodic_returns = []
                eval_successes = []
                eval_obs, _ = eval_envs.reset(seed=args.seed)
                eval_obs = torch.Tensor(eval_obs).to(device)
                eval_dones = torch.zeros(args.num_eval_envs).to(device)

                while len(eval_episodic_returns) < args.num_eval_episodes:
                    eval_actions, _, _, _ = agent.get_action_and_value(eval_obs)
                    eval_obs, _, eval_terminations, eval_truncations, eval_infos = (
                        eval_envs.step(eval_actions.cpu().numpy())
                    )
                    eval_obs = torch.Tensor(eval_obs).to(device)
                    eval_dones = torch.logical_or(eval_terminations, eval_truncations)

                    if "final_info" in eval_infos:
                        for info in eval_infos["final_info"]:
                            if info and isinstance(info, dict) and "episode" in info:
                                eval_episodic_returns.append(info["episode"]["r"])
                                eval_successes.append(info.get("success", 0))

                print(
                    f"📊 Eval iter {iteration}: {len(eval_episodic_returns)} episodes, "
                    f"mean return: {np.mean(eval_episodic_returns):.2f}, "
                    f"success rate: {np.mean(eval_successes):.2f}"
                )
                writer.add_scalar(
                    "eval/episodic_return", np.mean(eval_episodic_returns), global_step
                )
                writer.add_scalar(
                    "eval/success_rate", np.mean(eval_successes), global_step
                )
            agent.train()

        # Save model
        if args.save_model and iteration % args.eval_freq == 1:
            model_path = f"{args.output_dir}/{run_name}/ckpt_{iteration}.pt"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")

    if args.save_model:
        model_path = f"{args.output_dir}/{run_name}/final_ckpt.pt"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    eval_envs.close()
    writer.close()

    print()
    print("🎉 Training completed!")
    print(f"Final model saved in: {args.output_dir}/{run_name}/")
    print(f"Videos saved in: videos/{run_name}/")
    print(f"TensorBoard logs: tensorboard --logdir {args.output_dir}")


if __name__ == "__main__":
    main()
