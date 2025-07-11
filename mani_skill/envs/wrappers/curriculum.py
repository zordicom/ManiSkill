"""
Copyright 2025 Zordi, Inc. All rights reserved.

Curriculum learning wrapper for ManiSkill environments.
"""

import math
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch


class SuccessRateCurriculumWrapper(gym.Wrapper):
    """Curriculum learning wrapper that tracks success rates and manages difficulty progression.

    This wrapper automatically adjusts the curriculum level based on success rates.
    When the success rate reaches a threshold, it advances to the next curriculum level.

    This wrapper also handles episode time limits by overriding the TimeLimit wrapper behavior.

    Args:
        env: The environment to wrap
        curriculum_levels: List of curriculum level configurations
        success_threshold: Success rate threshold to advance to next level (default: 0.8)
        window_size: Number of episodes to track for success rate calculation (default: 100)
        min_episodes_per_level: Minimum episodes before allowing level advancement (default: 1000)
        steps_per_level: Number of training steps per curriculum level (default: 1_000_000)
        verbose: Whether to print curriculum progression info (default: True)
    """

    def __init__(
        self,
        env: gym.Env,
        curriculum_levels: List[Dict[str, Any]],
        success_threshold: float = 0.8,
        window_size: int = 100,
        min_episodes_per_level: int = 1000,
        steps_per_level: int = 1_000_000,
        verbose: bool = True,
    ):
        super().__init__(env)

        self.curriculum_levels = curriculum_levels
        self.success_threshold = success_threshold
        self.window_size = window_size
        self.min_episodes_per_level = min_episodes_per_level
        self.steps_per_level = steps_per_level
        self.verbose = verbose

        # Current curriculum state
        self.current_level = 0
        self.total_episodes = 0
        self.episodes_at_current_level = 0
        self.total_steps = 0
        self.steps_at_current_level = 0

        # Episode step tracking for time limits
        self._episode_steps = 0
        self._max_episode_steps = self._get_current_max_episode_steps()

        # Success tracking
        self.success_history = deque(maxlen=window_size)
        self.episode_rewards = deque(maxlen=window_size)

        # Level-specific tracking
        self.level_stats = []
        for i in range(len(curriculum_levels)):
            self.level_stats.append({
                "episodes": 0,
                "steps": 0,
                "successes": 0,
                "total_reward": 0.0,
                "completed": False,
            })

        if self.verbose:
            print(f"Curriculum wrapper initialized with {len(curriculum_levels)} levels")
            print(f"Success threshold: {success_threshold}")
            print(f"Window size: {window_size}")
            print(f"Min episodes per level: {min_episodes_per_level}")
            print(f"Steps per level: {steps_per_level}")
            print(f"Initial max episode steps: {self._max_episode_steps}")
            self._print_curriculum_levels()

    def _get_current_max_episode_steps(self) -> int:
        """Get the max episode steps for the current curriculum level."""
        current_config = self.curriculum_levels[self.current_level]
        return current_config.get("max_episode_steps", 50)  # Default to 50 if not specified

    def _print_curriculum_levels(self):
        """Print curriculum level descriptions."""
        print("\nCurriculum Levels:")
        for i, level in enumerate(self.curriculum_levels):
            print(f"  Level {i + 1}: {level.get('name', f'Level {i + 1}')}")
            if "description" in level:
                print(f"    {level['description']}")
            if "max_episode_steps" in level:
                print(f"    Max episode steps: {level['max_episode_steps']}")

    def reset(self, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Reset the environment with current curriculum level configuration."""
        # Apply current curriculum level settings
        current_config = self.curriculum_levels[self.current_level]

        # Merge curriculum config with any user-provided options
        options = kwargs.get("options", {})
        if "curriculum_level" not in options:
            options.update(current_config)
            kwargs["options"] = options

        # Reset the environment
        obs, info = self.env.reset(**kwargs)

        # Track episode start
        self.total_episodes += 1
        self.episodes_at_current_level += 1
        self.level_stats[self.current_level]["episodes"] += 1

        # Reset episode step counter and update max steps for current level
        self._episode_steps = 0
        self._max_episode_steps = self._get_current_max_episode_steps()

        return obs, info

    def step(self, action) -> Tuple[Any, Any, Any, Any, Dict[str, Any]]:
        """Step the environment and track curriculum progress."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Convert reward to a Python scalar for statistics tracking
        if isinstance(reward, torch.Tensor):
            reward_scalar = reward.item()
        elif isinstance(reward, np.ndarray):
            reward_scalar = reward.item()
        else:
            reward_scalar = reward

        # Use scalar reward for the remainder of the function to satisfy type hints
        reward = reward_scalar

        # Track steps
        self.total_steps += 1
        self.steps_at_current_level += 1
        self.level_stats[self.current_level]["steps"] += 1
        self.level_stats[self.current_level]["total_reward"] += reward_scalar

        # Track episode steps and apply curriculum-based time limit
        self._episode_steps += 1
        if self._episode_steps >= self._max_episode_steps:
            # Apply curriculum-based time limit
            if isinstance(truncated, torch.Tensor):
                truncated = torch.ones_like(truncated, dtype=torch.bool)
            elif isinstance(truncated, np.ndarray):
                truncated = np.ones_like(truncated, dtype=bool)
            else:
                truncated = True
            info["TimeLimit.truncated"] = True

        # Track success at episode end
        if terminated or truncated:
            success = info.get("success", False)
            if isinstance(success, torch.Tensor):
                success = success.any().item()  # Handle batched environments
            elif isinstance(success, np.ndarray):
                success = success.any()

            self.success_history.append(success)
            self.episode_rewards.append(reward_scalar)

            if success:
                self.level_stats[self.current_level]["successes"] += 1

            # Check for curriculum advancement
            self._check_curriculum_advancement()

        return obs, reward, terminated, truncated, info

    def _check_curriculum_advancement(self):
        """Check if we should advance to the next curriculum level."""
        # Don't advance if we're at the last level
        if self.current_level >= len(self.curriculum_levels) - 1:
            return

        # Check if we have enough episodes at this level
        if self.episodes_at_current_level < self.min_episodes_per_level:
            return

        # Check if we've completed the required steps for this level
        if self.steps_at_current_level < self.steps_per_level:
            return

        # Check success rate
        if len(self.success_history) >= self.window_size:
            success_rate = sum(self.success_history) / len(self.success_history)

            if success_rate >= self.success_threshold:
                self._advance_curriculum()

    def _advance_curriculum(self):
        """Advance to the next curriculum level."""
        # Mark current level as completed
        self.level_stats[self.current_level]["completed"] = True

        # Calculate final stats for current level
        current_stats = self.level_stats[self.current_level]
        success_rate = current_stats["successes"] / current_stats["episodes"] if current_stats["episodes"] > 0 else 0.0
        avg_reward = current_stats["total_reward"] / current_stats["episodes"] if current_stats["episodes"] > 0 else 0.0

        if self.verbose:
            print("\n🎓 CURRICULUM ADVANCEMENT 🎓")
            print(
                f"Completed Level {self.current_level + 1}: {self.curriculum_levels[self.current_level].get('name', '')}"
            )
            print(f"Episodes: {current_stats['episodes']}")
            print(f"Steps: {current_stats['steps']:,}")
            print(f"Success Rate: {success_rate:.3f}")
            print(f"Average Reward: {avg_reward:.3f}")

        # Advance to next level
        self.current_level += 1
        self.episodes_at_current_level = 0
        self.steps_at_current_level = 0

        # Clear success history for fresh start at new level
        self.success_history.clear()
        self.episode_rewards.clear()

        # Update max episode steps for new level
        self._max_episode_steps = self._get_current_max_episode_steps()

        if self.verbose:
            print(
                f"Advanced to Level {self.current_level + 1}: {self.curriculum_levels[self.current_level].get('name', '')}"
            )
            if "description" in self.curriculum_levels[self.current_level]:
                print(f"Description: {self.curriculum_levels[self.current_level]['description']}")
            print(f"New max episode steps: {self._max_episode_steps}")
            print()

    def get_curriculum_info(self) -> Dict[str, Any]:
        """Get current curriculum information."""
        current_success_rate = sum(self.success_history) / len(self.success_history) if self.success_history else 0.0
        current_avg_reward = sum(self.episode_rewards) / len(self.episode_rewards) if self.episode_rewards else 0.0

        return {
            "current_level": self.current_level,
            "total_levels": len(self.curriculum_levels),
            "current_level_name": self.curriculum_levels[self.current_level].get(
                "name", f"Level {self.current_level + 1}"
            ),
            "episodes_at_current_level": self.episodes_at_current_level,
            "steps_at_current_level": self.steps_at_current_level,
            "total_episodes": self.total_episodes,
            "total_steps": self.total_steps,
            "current_success_rate": current_success_rate,
            "current_avg_reward": current_avg_reward,
            "success_history_length": len(self.success_history),
            "level_stats": self.level_stats.copy(),
            "max_episode_steps": self._max_episode_steps,
            "progress_to_next_level": {
                "episodes_progress": min(1.0, self.episodes_at_current_level / self.min_episodes_per_level),
                "steps_progress": min(1.0, self.steps_at_current_level / self.steps_per_level),
                "success_rate_progress": min(1.0, current_success_rate / self.success_threshold),
            },
        }

    def print_curriculum_status(self):
        """Print current curriculum status."""
        info = self.get_curriculum_info()

        # Ensure scalar values for safe formatting
        def _scalar(val):
            if isinstance(val, torch.Tensor):
                return val.item()
            if isinstance(val, np.ndarray):
                return val.item()
            return val

        info["current_success_rate"] = _scalar(info["current_success_rate"])
        info["current_avg_reward"] = _scalar(info["current_avg_reward"])

        print("\n📚 CURRICULUM STATUS 📚")
        print(f"Current Level: {info['current_level'] + 1}/{info['total_levels']} - {info['current_level_name']}")
        print(f"Episodes at Level: {info['episodes_at_current_level']:,}")
        print(f"Steps at Level: {info['steps_at_current_level']:,}")
        print(f"Max Episode Steps: {info['max_episode_steps']}")
        print(f"Success Rate: {info['current_success_rate']:.3f} (need {self.success_threshold:.3f})")
        print(f"Average Reward: {info['current_avg_reward']:.3f}")

        progress = info["progress_to_next_level"]
        print("Progress to Next Level:")
        print(f"  Episodes: {progress['episodes_progress']:.1%}")
        print(f"  Steps: {progress['steps_progress']:.1%}")
        print(f"  Success Rate: {progress['success_rate_progress']:.1%}")

        print("\nLevel Summary:")
        for i, stats in enumerate(info["level_stats"]):
            status = (
                "✅ Completed" if stats["completed"] else ("🔄 Current" if i == info["current_level"] else "⏳ Pending")
            )
            success_rate = stats["successes"] / stats["episodes"] if stats["episodes"] > 0 else 0.0
            avg_reward = stats["total_reward"] / stats["episodes"] if stats["episodes"] > 0 else 0.0
            success_rate = _scalar(success_rate)
            avg_reward = _scalar(avg_reward)
            print(
                f"  Level {i + 1}: {status} - {stats['episodes']:,} episodes, {success_rate:.3f} success rate, {avg_reward:.3f} avg reward"
            )
        print()

    def force_advance_curriculum(self):
        """Force advancement to the next curriculum level (for debugging/testing)."""
        if self.current_level < len(self.curriculum_levels) - 1:
            if self.verbose:
                print("🔧 FORCED CURRICULUM ADVANCEMENT 🔧")
            self._advance_curriculum()
        elif self.verbose:
            print("Already at the highest curriculum level!")

    def set_curriculum_level(self, level: int):
        """Set the curriculum to a specific level (for debugging/testing)."""
        if 0 <= level < len(self.curriculum_levels):
            old_level = self.current_level
            self.current_level = level
            self.episodes_at_current_level = 0
            self.steps_at_current_level = 0
            self.success_history.clear()
            self.episode_rewards.clear()

            # Update max episode steps for new level
            self._max_episode_steps = self._get_current_max_episode_steps()

            if self.verbose:
                print(f"🔧 Curriculum level manually set from {old_level + 1} to {level + 1}")
                print(f"New max episode steps: {self._max_episode_steps}")
        else:
            raise ValueError(
                f"Invalid curriculum level {level}. Must be between 0 and {len(self.curriculum_levels) - 1}"
            )
