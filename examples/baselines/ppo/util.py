"""
Utility functions for PPO training, including Expert+Residual environment creation.
"""

import gymnasium as gym
from mani_skill.envs.wrappers.expert_residual import ExpertResidualWrapper
from mani_skill.envs.wrappers.experts import create_expert_policy


def create_expert_residual_envs(args, env_kwargs, device):
    """
    Create training and evaluation environments with Expert+Residual wrapper.

    Args:
        args: Arguments object with expert-specific parameters
        env_kwargs: Base environment arguments
        device: Device for expert policy

    Returns:
        tuple: (envs, eval_envs) - training and evaluation environments
    """
    # Get action dimension for expert policy
    dummy = gym.make(args.env_id, num_envs=1, **env_kwargs)
    action_dim = dummy.action_space.shape[-1]
    dummy.close()

    # Create expert policy
    expert_kwargs = {}
    if args.expert_type == "ik":
        expert_kwargs["gain"] = getattr(args, "ik_gain", 2.0)
    elif args.expert_type == "model":
        expert_kwargs.update(
            dict(model_path=getattr(args, "model_path", None), device=str(device))
        )
    elif args.expert_type == "act":
        # ACT expert policy parameters
        expert_kwargs.update({
            "config_path": getattr(args, "config_path", None),
            "checkpoint_path": getattr(args, "checkpoint_path", None),
            "output_format": getattr(args, "output_format", "absolute_joints"),
            "action_offset": getattr(args, "action_offset", 0),
            "action_clamp": getattr(args, "action_clamp", 0.5),
            "device": str(device),
        })
        if args.expert_type == "dummy_act":
            expert_kwargs["act_output_dim"] = getattr(args, "act_output_dim", 14)

    expert_policy = create_expert_policy(args.expert_type, action_dim, **expert_kwargs)

    # Remove control_mode from env_kwargs to avoid duplicate parameter
    wrapper_env_kwargs = env_kwargs.copy()
    wrapper_env_kwargs.pop("control_mode", None)

    # Create training environment with Expert+Residual wrapper
    envs = ExpertResidualWrapper(
        env_id=args.env_id,
        expert_policy_fn=expert_policy,
        num_envs=args.num_envs if not args.evaluate else 1,
        residual_scale=getattr(args, "residual_scale", 1.0),
        clip_final_action=True,
        expert_action_noise=getattr(args, "expert_action_noise", 0.0),
        log_actions=False,
        track_action_stats=getattr(args, "track_action_stats", False),
        device=str(device),
        control_mode=getattr(args, "control_mode", "pd_joint_delta_pos"),
        reconfiguration_freq=args.reconfiguration_freq,
        **wrapper_env_kwargs,
    )

    # Create evaluation environment with Expert+Residual wrapper
    eval_envs = ExpertResidualWrapper(
        env_id=args.env_id,
        expert_policy_fn=expert_policy,
        num_envs=args.num_eval_envs,
        residual_scale=getattr(args, "residual_scale", 1.0),
        clip_final_action=True,
        expert_action_noise=0.0,  # No noise during evaluation
        log_actions=False,
        track_action_stats=False,
        device=str(device),
        control_mode=getattr(args, "control_mode", "pd_joint_delta_pos"),
        reconfiguration_freq=args.eval_reconfiguration_freq,
        human_render_camera_configs=dict(shader_pack="default"),
        **wrapper_env_kwargs,
    )

    return envs, eval_envs
