#!/usr/bin/env python3
"""
Test script for SAC RGBD with PPO RGB Fast expert.

This script tests the updated SAC RGBD implementation with different expert types,
particularly the PPO RGB Fast expert trained on PickCube-v1.
"""

import subprocess
import sys
from pathlib import Path

# Path to the SAC RGBD script
SAC_RGBD_SCRIPT = Path(__file__).resolve().parent / "sac_rgbd.py"

# Default PPO RGB Fast checkpoint path (from dataset generator)
DEFAULT_PPO_RGB_FAST_CHECKPOINT = "/home/gilwoo/workspace/ManiSkill/runs/PickCube-v1__ppo_rgb_fast__1__1752546740/ckpt_326.pt"


def test_sac_rgbd_no_expert():
    """Test SAC RGBD without expert (baseline)."""
    print("=== Testing SAC RGBD without expert ===")

    cmd = [
        sys.executable,
        str(SAC_RGBD_SCRIPT),
        "--env-id",
        "PickCube-v1",
        "--expert-type",
        "none",
        "--num-envs",
        "4",
        "--num-eval-envs",
        "2",
        "--total-timesteps",
        "1000",
        "--eval-freq",
        "10",
        "--batch-size",
        "64",
        "--learning-starts",
        "100",
        "--utd",
        "0.5",
        "--include-state",
        "--camera-width",
        "128",
        "--camera-height",
        "128",
    ]

    print("Command:", " ".join(cmd))
    print("This runs regular SAC RGBD without expert...")
    # subprocess.run(cmd)  # Uncomment to actually run


def test_sac_rgbd_zero_expert():
    """Test SAC RGBD with zero expert."""
    print("\n=== Testing SAC RGBD with zero expert ===")

    cmd = [
        sys.executable,
        str(SAC_RGBD_SCRIPT),
        "--env-id",
        "PickCube-v1",
        "--expert-type",
        "zero",
        "--residual-scale",
        "1.0",
        "--num-envs",
        "4",
        "--num-eval-envs",
        "2",
        "--total-timesteps",
        "1000",
        "--eval-freq",
        "10",
        "--batch-size",
        "64",
        "--learning-starts",
        "100",
        "--utd",
        "0.5",
        "--include-state",
        "--camera-width",
        "128",
        "--camera-height",
        "128",
        "--track-action-stats",
    ]

    print("Command:", " ".join(cmd))
    print("This runs SAC RGBD with zero expert (extended observation space)...")
    # subprocess.run(cmd)  # Uncomment to actually run


def test_sac_rgbd_ppo_rgb_fast_expert():
    """Test SAC RGBD with PPO RGB Fast expert."""
    print("\n=== Testing SAC RGBD with PPO RGB Fast expert ===")

    cmd = [
        sys.executable,
        str(SAC_RGBD_SCRIPT),
        "--env-id",
        "PickCube-v1",
        "--expert-type",
        "ppo_rgb_fast",
        "--ppo-rgb-fast-path",
        DEFAULT_PPO_RGB_FAST_CHECKPOINT,
        "--residual-scale",
        "0.5",
        "--num-envs",
        "4",
        "--num-eval-envs",
        "2",
        "--total-timesteps",
        "1000",
        "--eval-freq",
        "10",
        "--batch-size",
        "64",
        "--learning-starts",
        "100",
        "--utd",
        "0.5",
        "--include-state",
        "--camera-width",
        "128",
        "--camera-height",
        "128",
        "--track-action-stats",
    ]

    print("Command:", " ".join(cmd))
    print("This runs SAC RGBD with PPO RGB Fast expert...")
    print(f"Using checkpoint: {DEFAULT_PPO_RGB_FAST_CHECKPOINT}")
    # subprocess.run(cmd)  # Uncomment to actually run


def test_sac_rgbd_model_expert():
    """Test SAC RGBD with model expert."""
    print("\n=== Testing SAC RGBD with model expert ===")

    cmd = [
        sys.executable,
        str(SAC_RGBD_SCRIPT),
        "--env-id",
        "PickCube-v1",
        "--expert-type",
        "model",
        "--model-path",
        DEFAULT_PPO_RGB_FAST_CHECKPOINT,
        "--residual-scale",
        "0.3",
        "--num-envs",
        "4",
        "--num-eval-envs",
        "2",
        "--total-timesteps",
        "1000",
        "--eval-freq",
        "10",
        "--batch-size",
        "64",
        "--learning-starts",
        "100",
        "--utd",
        "0.5",
        "--include-state",
        "--camera-width",
        "128",
        "--camera-height",
        "128",
        "--track-action-stats",
    ]

    print("Command:", " ".join(cmd))
    print("This runs SAC RGBD with model expert...")
    print(f"Using checkpoint: {DEFAULT_PPO_RGB_FAST_CHECKPOINT}")
    # subprocess.run(cmd)  # Uncomment to actually run


def show_expert_parameters():
    """Show expert-related parameters."""
    print("\n=== Expert Parameters ===")

    expert_params = [
        {
            "name": "--expert-type",
            "options": "none, zero, ik, model, ppo_rgb_fast",
            "description": "Type of expert policy to use",
        },
        {
            "name": "--ppo-rgb-fast-path",
            "options": "path/to/checkpoint.pt",
            "description": "Path to pre-trained PPO RGB Fast model checkpoint",
        },
        {
            "name": "--model-path",
            "options": "path/to/model.pt",
            "description": "Path to pre-trained model for model expert policy",
        },
        {
            "name": "--residual-scale",
            "options": "0.1, 0.5, 1.0",
            "description": "Scale factor for residual actions",
        },
        {
            "name": "--expert-action-noise",
            "options": "0.0, 0.1, 0.2",
            "description": "Gaussian noise std to add to expert actions",
        },
        {
            "name": "--track-action-stats",
            "options": "flag",
            "description": "Track expert/residual action statistics",
        },
        {
            "name": "--ik-gain",
            "options": "1.0, 2.0, 3.0",
            "description": "Proportional gain for IK expert policy",
        },
    ]

    for param in expert_params:
        print(f"\n{param['name']}")
        print(f"  Options: {param['options']}")
        print(f"  Description: {param['description']}")


def main():
    """Main function to run tests."""
    print("Testing SAC RGBD with Expert Support")
    print("=" * 50)

    # Show parameters
    show_expert_parameters()

    # Run tests (commented out to prevent accidental execution)
    test_sac_rgbd_no_expert()
    test_sac_rgbd_zero_expert()
    test_sac_rgbd_ppo_rgb_fast_expert()
    test_sac_rgbd_model_expert()

    print("\n" + "=" * 50)
    print("All tests defined. Uncomment subprocess.run() calls to execute.")
    print("=" * 50)


if __name__ == "__main__":
    main()
