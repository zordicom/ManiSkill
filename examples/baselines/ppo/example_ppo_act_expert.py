#!/usr/bin/env python3
"""
Example usage of PPO with ACT experts for different output formats and control modes.

This script demonstrates how to use the enhanced PPO implementation with ACT experts
that can output different formats (absolute joints, absolute tool-pose, deltas) and
work with different ManiSkill control modes.
"""

import subprocess
import sys
from pathlib import Path

# Path to the PPO script
PPO_SCRIPT = Path(__file__).resolve().parent / "ppo.py"


def run_ppo_with_dummy_act_expert():
    """Example: PPO with dummy ACT expert (for testing without real ACT models)."""
    print("=== Running PPO with Dummy ACT Expert ===")

    # Example 1: Joint control with absolute joints output
    cmd_joint = [
        sys.executable,
        str(PPO_SCRIPT),
        "--env-id",
        "PickCube-v1",
        "--expert-type",
        "dummy_act",
        "--control-mode",
        "pd_joint_delta_pos",
        "--output-format",
        "absolute_joints",
        "--act-output-dim",
        "14",
        "--action-clamp",
        "0.5",
        "--total-timesteps",
        "10000",
        "--num-envs",
        "64",
        "--eval-freq",
        "10",
        "--residual-scale",
        "1.0",
        "--track-action-stats",
    ]

    print("Command:", " ".join(cmd_joint))
    print("This would train PPO with dummy ACT expert using joint control...")
    # subprocess.run(cmd_joint)  # Uncomment to actually run

    # Example 2: End-effector control with absolute EE position output
    cmd_ee = [
        sys.executable,
        str(PPO_SCRIPT),
        "--env-id",
        "PickCube-v1",
        "--expert-type",
        "dummy_act",
        "--control-mode",
        "pd_ee_delta_pos",
        "--output-format",
        "absolute_ee_pos",
        "--act-output-dim",
        "14",
        "--action-clamp",
        "0.5",
        "--total-timesteps",
        "10000",
        "--num-envs",
        "64",
        "--eval-freq",
        "10",
        "--residual-scale",
        "0.5",  # Smaller residual for EE control
        "--track-action-stats",
    ]

    print("\nCommand:", " ".join(cmd_ee))
    print("This would train PPO with dummy ACT expert using EE position control...")
    # subprocess.run(cmd_ee)  # Uncomment to actually run


def run_ppo_with_real_act_expert():
    """Example: PPO with real ACT expert (requires actual ACT model files)."""
    print("\n=== Running PPO with Real ACT Expert ===")

    # Example with real ACT model (you would need actual config and checkpoint files)
    cmd_real = [
        sys.executable,
        str(PPO_SCRIPT),
        "--env-id",
        "PickCube-v1",
        "--expert-type",
        "act",
        "--config-path",
        "/path/to/act_config.yaml",
        "--checkpoint-path",
        "/path/to/act_model.ckpt",
        "--control-mode",
        "pd_joint_delta_pos",
        "--output-format",
        "absolute_joints",
        "--action-offset",
        "0",
        "--action-clamp",
        "0.5",
        "--total-timesteps",
        "100000",
        "--num-envs",
        "128",
        "--eval-freq",
        "25",
        "--residual-scale",
        "1.0",
        "--track-action-stats",
    ]

    print("Command:", " ".join(cmd_real))
    print(
        "This would train PPO with real ACT expert (requires actual ACT model files)..."
    )
    # subprocess.run(cmd_real)  # Uncomment to actually run


def show_available_configurations():
    """Show available output formats and control modes."""
    print("\n=== Available Configurations ===")

    configurations = [
        {
            "name": "Joint Control with Absolute Joints",
            "output_format": "absolute_joints",
            "control_mode": "pd_joint_delta_pos",
            "description": "ACT outputs absolute joint positions, converted to joint deltas for environment",
        },
        {
            "name": "Joint Control with Joint Deltas",
            "output_format": "joint_deltas",
            "control_mode": "pd_joint_delta_pos",
            "description": "ACT outputs joint deltas directly",
        },
    ]

    for config in configurations:
        print(f"\n{config['name']}:")
        print(f"  Output Format: {config['output_format']}")
        print(f"  Control Mode: {config['control_mode']}")
        print(f"  Description: {config['description']}")


def show_command_line_arguments():
    """Show new command-line arguments for ACT expert support."""
    print("\n=== New Command-Line Arguments ===")

    args = [
        {
            "name": "--expert-type",
            "options": "none, zero, ik, model, act, dummy_act",
            "description": "Type of expert policy to use",
        },
        {
            "name": "--config-path",
            "options": "path/to/config.yaml",
            "description": "Path to ACT model configuration file (for act expert)",
        },
        {
            "name": "--checkpoint-path",
            "options": "path/to/model.ckpt",
            "description": "Path to ACT model checkpoint file (for act expert)",
        },
        {
            "name": "--output-format",
            "options": "absolute_joints, joint_deltas",
            "description": "Format of ACT model output",
        },
        {
            "name": "--action-offset",
            "options": "0, 1, 2, ...",
            "description": "Offset for selecting action from ACT action chunk",
        },
        {
            "name": "--action-clamp",
            "options": "0.1, 0.5, 1.0, ...",
            "description": "Clamp range for ACT actions",
        },
        {
            "name": "--act-output-dim",
            "options": "7, 14, 21, ...",
            "description": "Dimension of ACT model output (for dummy_act)",
        },
        {
            "name": "--residual-scale",
            "options": "0.1, 0.5, 1.0, ...",
            "description": "Scale factor for residual actions",
        },
        {
            "name": "--track-action-stats",
            "options": "flag",
            "description": "Track expert/residual action statistics",
        },
    ]

    for arg in args:
        print(f"\n{arg['name']}")
        print(f"  Options: {arg['options']}")
        print(f"  Description: {arg['description']}")


if __name__ == "__main__":
    print("PPO with ACT Expert Examples")
    print("=" * 50)

    # Show available configurations
    show_available_configurations()

    # Show command-line arguments
    show_command_line_arguments()

    # Show example usage
    run_ppo_with_dummy_act_expert()
    run_ppo_with_real_act_expert()

    print("\n=== Notes ===")
    print("1. Use dummy_act expert for testing without real ACT models")
    print("2. Use act expert with real ACT model files for actual training")
    print("3. Match output_format and control_mode appropriately")
    print("4. Adjust residual_scale based on your task and expert quality")
    print("5. Use track_action_stats to monitor expert/residual action statistics")

    print("\n=== Quick Start ===")
    print("To test with dummy ACT expert:")
    print(
        f"python {PPO_SCRIPT} --env-id PickCube-v1 --expert-type dummy_act --total-timesteps 10000"
    )

    print("\nTo use with real ACT expert:")
    print(
        f"python {PPO_SCRIPT} --env-id PickCube-v1 --expert-type act --config-path config.yaml --checkpoint-path model.ckpt"
    )
