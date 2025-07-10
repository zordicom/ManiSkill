#!/usr/bin/env python3
"""
Demo script for PickCubeNoisy environment
Shows different noise configurations for PPO training
"""

import gymnasium as gym
import mani_skill.envs.tasks.tabletop.pick_cube_noisy
import numpy as np


def demo_pick_cube_noisy():
    """Demonstrate different noise configurations for PickCubeNoisy environment."""
    print("PickCubeNoisy Environment Demo")
    print("=" * 50)

    # Configuration 1: Light noise for basic robustness
    print("\n1. Light noise configuration (basic robustness):")
    env = gym.make(
        "PickCubeNoisy-v1",
        obs_mode="state",
        obs_noise_type="gaussian",
        obs_noise_std=0.005,
        reward_noise_type="gaussian",
        reward_noise_std=0.05,
        action_noise_type="none",
    )

    obs, info = env.reset()
    print(f"   Noise info: {env.get_noise_info()}")
    action = np.random.uniform(-1, 1, size=env.action_space.shape)
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"   Sample reward: {reward.item():.3f}")
    env.close()

    # Configuration 2: Medium noise for training robustness
    print("\n2. Medium noise configuration (training robustness):")
    env = gym.make(
        "PickCubeNoisy-v1",
        obs_mode="state",
        obs_noise_type="gaussian",
        obs_noise_std=0.01,
        reward_noise_type="gaussian",
        reward_noise_std=0.1,
        action_noise_type="gaussian",
        action_noise_std=0.05,
    )

    obs, info = env.reset()
    print(f"   Noise info: {env.get_noise_info()}")
    action = np.random.uniform(-1, 1, size=env.action_space.shape)
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"   Sample reward: {reward.item():.3f}")
    env.close()

    # Configuration 3: Heavy noise for stress testing
    print("\n3. Heavy noise configuration (stress testing):")
    env = gym.make(
        "PickCubeNoisy-v1",
        obs_mode="state",
        obs_noise_type="gaussian",
        obs_noise_std=0.02,
        reward_noise_type="gaussian",
        reward_noise_std=0.2,
        action_noise_type="gaussian",
        action_noise_std=0.1,
    )

    obs, info = env.reset()
    print(f"   Noise info: {env.get_noise_info()}")
    action = np.random.uniform(-1, 1, size=env.action_space.shape)
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"   Sample reward: {reward.item():.3f}")
    env.close()

    # Configuration 4: Growing noise for curriculum learning
    print("\n4. Growing noise configuration (curriculum learning):")
    env = gym.make(
        "PickCubeNoisy-v1",
        obs_mode="state",
        obs_noise_type="gaussian",
        obs_noise_std=0.02,
        reward_noise_type="gaussian",
        reward_noise_std=0.15,
        noise_growth_rate=0.05,  # Increase noise by 0.05 each episode
        min_noise_factor=0.1,  # Start with 10% noise
        max_noise_factor=1.0,  # Cap at 100% noise
    )

    for episode in range(5):
        obs, info = env.reset()
        noise_info = env.get_noise_info()
        print(
            f"   Episode {episode}: noise_factor={noise_info['current_noise_factor']:.3f}"
        )

    env.close()

    # Configuration 5: Different robot examples
    print("\n5. Different robot configurations:")

    robot_envs = [
        ("PickCubeNoisy-v1", "Panda (default)"),
        ("PickCubeNoisyFetch-v1", "Fetch robot"),
        ("PickCubeNoisyXArm6Robotiq-v1", "XArm6 with Robotiq gripper"),
        ("PickCubeNoisySO100-v1", "SO100 robot"),
        ("PickCubeNoisyWidowXAI-v1", "WidowXAI robot"),
        ("PickCubeNoisyA1Galaxea-v1", "A1Galaxea robot"),
    ]

    for env_name, description in robot_envs:
        try:
            env = gym.make(env_name, obs_mode="state", obs_noise_std=0.01)
            obs, info = env.reset()
            print(f"   ✅ {description}: {env_name}")
            env.close()
        except Exception as e:
            print(f"   ❌ {description}: {env_name} - Error: {e}")

    print("\n" + "=" * 50)
    print("PPO Training Tips:")
    print("- Start with light noise for initial learning")
    print("- Gradually increase noise as training progresses")
    print("- Use growing noise for curriculum learning (start easy, get harder)")
    print("- Monitor success rate - if it drops too low, reduce noise growth rate")
    print("- Action noise can help with exploration but may slow learning")
    print("- Observation noise helps with sensor robustness")
    print("- Reward noise helps with policy robustness")
    print(
        "- Different robots have different configurations (cube size, workspace, etc.)"
    )
    print("- Choose robot based on your specific research needs")
    print("- All robots support the same noise parameters")


if __name__ == "__main__":
    demo_pick_cube_noisy()
