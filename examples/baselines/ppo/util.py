"""
Utility functions for PPO training, including Expert+Residual environment creation.
"""

import gymnasium as gym
import torch
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
        # For model expert, we need to use the action dimension from the model, not the environment
        # The expert policy will handle any dimension mismatch
        expert_policy = create_expert_policy(
            args.expert_type, action_dim, **expert_kwargs
        )

        # Check if there's an action dimension mismatch and handle it
        if hasattr(expert_policy, "_model_action_dim"):
            model_action_dim = expert_policy._model_action_dim
            if model_action_dim != action_dim:
                print("🔧 Action dimension mismatch detected:")
                print(f"   Environment action dim: {action_dim}")
                print(f"   Model action dim: {model_action_dim}")
                print("   Creating action adapter...")

                # Create an action adapter
                original_expert_policy = expert_policy

                def adapted_expert_policy(obs):
                    # Get the full action from the original expert
                    full_action = original_expert_policy(obs)

                    # Handle different action dimension mappings
                    if model_action_dim == 8 and action_dim == 4:
                        # Likely pd_joint_delta_pos (8) -> pd_ee_delta_pos (4)
                        # Take the first 3 dimensions as position and last 1 as gripper
                        if full_action.dim() == 1:
                            adapted_action = torch.zeros(
                                action_dim,
                                device=full_action.device,
                                dtype=full_action.dtype,
                            )
                            adapted_action[:3] = full_action[:3]  # Position
                            adapted_action[3] = full_action[7]  # Gripper
                        else:
                            batch_size = full_action.shape[0]
                            adapted_action = torch.zeros(
                                batch_size,
                                action_dim,
                                device=full_action.device,
                                dtype=full_action.dtype,
                            )
                            adapted_action[:, :3] = full_action[:, :3]  # Position
                            adapted_action[:, 3] = full_action[:, 7]  # Gripper
                        return adapted_action
                    # For other mismatches, just truncate or pad
                    elif full_action.dim() == 1:
                        if model_action_dim > action_dim:
                            return full_action[:action_dim]
                        else:
                            adapted_action = torch.zeros(
                                action_dim,
                                device=full_action.device,
                                dtype=full_action.dtype,
                            )
                            adapted_action[:model_action_dim] = full_action
                            return adapted_action
                    elif model_action_dim > action_dim:
                        return full_action[:, :action_dim]
                    else:
                        batch_size = full_action.shape[0]
                        adapted_action = torch.zeros(
                            batch_size,
                            action_dim,
                            device=full_action.device,
                            dtype=full_action.dtype,
                        )
                        adapted_action[:, :model_action_dim] = full_action
                        return adapted_action

                expert_policy = adapted_expert_policy

        # Skip the normal expert policy creation since we already created it
        expert_policy_created = True
    else:
        expert_policy_created = False

    # Create expert policy for other types if not already created
    if not expert_policy_created:
        if args.expert_type == "act":
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

        expert_policy = create_expert_policy(
            args.expert_type, action_dim, **expert_kwargs
        )

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
        expert_type=args.expert_type,  # Pass expert_type to wrapper
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
        expert_type=args.expert_type,  # Pass expert_type to wrapper
        reconfiguration_freq=args.eval_reconfiguration_freq,
        human_render_camera_configs=dict(shader_pack="default"),
        **wrapper_env_kwargs,
    )

    return envs, eval_envs
