#!/usr/bin/env python3
"""
Demo script for A1BimanualPickPlace environment

This script demonstrates:
1. Creating the A1BimanualPickPlace environment
2. Understanding the MultiAgent action space
3. Basic teleoperation with keyboard controls
4. Visualizing the task with cameras
"""

import gymnasium as gym
import numpy as np
import torch

import mani_skill.envs


def demo_a1_bimanual_environment():
    """Demonstrate the A1BimanualPickPlace environment."""
    print("🤖 A1 Bimanual Pick-and-Place Demo")
    print("=" * 50)

    # Create environment with RGB rendering
    env = gym.make(
        "A1BimanualPickPlace-v1",
        render_mode="rgb_array",
        obs_mode="state",  # Use state observations for clearer debugging
        num_envs=1,
    )

    print(f"✓ Environment created: {env}")
    print(f"  - Action space: {env.action_space}")
    print(f"  - Observation space: {env.observation_space}")

    # Reset environment
    obs, info = env.reset()
    print("\n✓ Environment reset")
    print(
        f"  - Observation keys: {list(obs.keys()) if isinstance(obs, dict) else 'Flattened tensor'}"
    )
    print(f"  - Info: {info}")

    # Demonstrate action structure
    print("\n🎮 Action Space Structure:")
    for agent_name, action_space in env.action_space.spaces.items():
        print(f"  - {agent_name}: {action_space} (7-DoF: 6 arm joints + 1 gripper)")

    # Run a few episodes with different action strategies
    strategies = [
        ("Random Actions", lambda: env.action_space.sample()),
        (
            "Zero Actions (Gravity Test)",
            lambda: {
                "a1_galaxea-0": np.zeros(7, dtype=np.float32),
                "a1_galaxea-1": np.zeros(7, dtype=np.float32),
            },
        ),
        (
            "Right Arm Reach (Left Idle)",
            lambda: {
                "a1_galaxea-0": np.zeros(7, dtype=np.float32),  # Left arm idle
                "a1_galaxea-1": np.random.uniform(-0.1, 0.1, 7).astype(
                    np.float32
                ),  # Right arm small movements
            },
        ),
    ]

    for strategy_name, action_fn in strategies:
        print(f"\n🎯 Testing: {strategy_name}")
        print("-" * 30)

        obs, info = env.reset()

        for step in range(10):
            action = action_fn()
            obs, reward, terminated, truncated, info = env.step(action)

            # Extract evaluation metrics
            eval_info = env.unwrapped.evaluate()

            # Handle tensor values for printing
            reward_val = reward.item() if hasattr(reward, "item") else reward
            success = (
                eval_info["success"].item()
                if hasattr(eval_info["success"], "item")
                else eval_info["success"]
            )
            obj_placed = (
                eval_info["is_obj_placed"].item()
                if hasattr(eval_info["is_obj_placed"], "item")
                else eval_info["is_obj_placed"]
            )
            arm_static = (
                eval_info["is_right_arm_static"].item()
                if hasattr(eval_info["is_right_arm_static"], "item")
                else eval_info["is_right_arm_static"]
            )
            distance = (
                eval_info["b5box_to_basket_dist"].item()
                if hasattr(eval_info["b5box_to_basket_dist"], "item")
                else eval_info["b5box_to_basket_dist"]
            )

            print(
                f"  Step {step + 1:2d}: reward={reward_val:6.3f} | success={success} | obj_placed={obj_placed} | arm_static={arm_static} | dist={distance:.3f}"
            )

            if terminated or truncated:
                print(
                    f"    Episode ended: terminated={terminated}, truncated={truncated}"
                )
                break

        # Render final state
        try:
            rgb_array = env.render()
            print(f"  ✓ Rendered frame: {rgb_array.shape}")
        except Exception as e:
            print(f"  ✗ Render failed: {e}")

    # Test camera observations
    print("\n📷 Camera Observations Test:")
    env_with_cameras = gym.make(
        "A1BimanualPickPlace-v1",
        render_mode="rgb_array",
        obs_mode="rgbd",  # Include camera observations
        num_envs=1,
    )

    obs, info = env_with_cameras.reset()
    if isinstance(obs, dict):
        print(f"  - Camera observation keys: {list(obs.keys())}")
        for key, value in obs.items():
            if hasattr(value, "shape"):
                print(f"    - {key}: {value.shape}")
    else:
        print(f"  - Flattened observation shape: {obs.shape}")

    env_with_cameras.close()
    env.close()

    print("\n🎉 Demo completed successfully!")
    print("\n💡 Usage Tips:")
    print("  - Use obs_mode='state' for proprioceptive observations")
    print("  - Use obs_mode='rgbd' to include camera images")
    print(
        "  - Action space is Dict with keys 'a1_galaxea-0' (left) and 'a1_galaxea-1' (right)"
    )
    print("  - Each action is 7D: [arm_joint1-6, gripper] with range [-1, 1]")
    print("  - Success requires: b5box in basket + right arm static")


if __name__ == "__main__":
    demo_a1_bimanual_environment()
