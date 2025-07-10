"""
Copyright 2025 Zordi, Inc. All rights reserved.

Example script demonstrating curriculum learning with PickBox environment.
"""

import gymnasium as gym
import mani_skill.envs
import numpy as np
import torch
from mani_skill.envs.tasks.tabletop.pick_box_curriculum import create_curriculum_wrapper


def main():
    """Demonstrate curriculum learning with PickBox environment."""
    # Create the base environment
    env = gym.make(
        "PickBox-v1",
        num_envs=1,
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        render_mode="rgb_array",
        robot_uids="a1_galaxea",
        verbose=True,
    )

    # Wrap with curriculum learning
    curriculum_env = create_curriculum_wrapper(
        env,
        success_threshold=0.8,
        window_size=50,  # Smaller window for demo
        min_episodes_per_level=10,  # Fewer episodes for demo
        steps_per_level=1000,  # Fewer steps for demo
        verbose=True,
    )

    print("🎯 Starting curriculum learning demonstration...")
    print("This demo will run through curriculum levels automatically.")
    print(
        "In real training, you would use your RL algorithm instead of random actions.\n"
    )

    # Simulate training episodes
    total_episodes = 100
    episode_count = 0

    while episode_count < total_episodes:
        # Reset environment
        obs, info = curriculum_env.reset()

        episode_reward = 0
        episode_steps = 0

        # Run episode with random actions (replace with your RL algorithm)
        while True:
            # Sample random action
            action = curriculum_env.action_space.sample()

            # Step environment
            obs, reward, terminated, truncated, info = curriculum_env.step(action)

            episode_reward += reward
            episode_steps += 1

            # Check if episode is done
            if terminated or truncated:
                break

        episode_count += 1

        # Print episode summary
        success = info.get("success", False)
        if isinstance(success, torch.Tensor):
            success = success.item()
        elif isinstance(success, np.ndarray):
            success = success.item()

        print(
            f"Episode {episode_count}: "
            f"{'✅ Success' if success else '❌ Failed'} "
            f"(reward: {episode_reward:.3f}, steps: {episode_steps})"
        )

        # Print curriculum status every 10 episodes
        if episode_count % 10 == 0:
            curriculum_env.print_curriculum_status()

        # Force advancement for demo purposes (remove in real training)
        if episode_count % 15 == 0:
            curriculum_env.force_advance_curriculum()

    # Final curriculum status
    print("\n🏁 Training completed!")
    curriculum_env.print_curriculum_status()

    # Get final curriculum info
    final_info = curriculum_env.get_curriculum_info()
    print("\nFinal Statistics:")
    print(f"Total Episodes: {final_info['total_episodes']}")
    print(f"Total Steps: {final_info['total_steps']}")
    print(
        f"Highest Level Reached: {final_info['current_level'] + 1}/{final_info['total_levels']}"
    )

    # Close environment
    curriculum_env.close()


def test_curriculum_levels():
    """Test each curriculum level individually."""
    print("🧪 Testing individual curriculum levels...")

    # Create base environment
    env = gym.make(
        "PickBox-v1",
        num_envs=1,
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        render_mode="rgb_array",
        robot_uids="a1_galaxea",
        verbose=False,
    )

    # Create curriculum wrapper
    curriculum_env = create_curriculum_wrapper(
        env,
        success_threshold=0.8,
        window_size=10,
        min_episodes_per_level=5,
        steps_per_level=100,
        verbose=False,
    )

    # Test each level
    for level in range(len(curriculum_env.curriculum_levels)):
        print(
            f"\n📚 Testing Level {level + 1}: {curriculum_env.curriculum_levels[level]['name']}"
        )
        print(f"Description: {curriculum_env.curriculum_levels[level]['description']}")

        # Set to specific level
        curriculum_env.set_curriculum_level(level)

        # Run a few episodes
        for episode in range(3):
            obs, info = curriculum_env.reset()

            episode_steps = 0
            while episode_steps < 10:  # Short episodes for testing
                action = curriculum_env.action_space.sample()
                obs, reward, terminated, truncated, info = curriculum_env.step(action)
                episode_steps += 1

                if terminated or truncated:
                    break

            success = info.get("success", False)
            if isinstance(success, torch.Tensor):
                success = success.item()
            elif isinstance(success, np.ndarray):
                success = success.item()

            print(
                f"  Episode {episode + 1}: {'✅' if success else '❌'} ({episode_steps} steps)"
            )

    curriculum_env.close()
    print("\n✅ All curriculum levels tested successfully!")


if __name__ == "__main__":
    # Run the main demo
    main()

    print("\n" + "=" * 50)

    # Run level testing
    test_curriculum_levels()
