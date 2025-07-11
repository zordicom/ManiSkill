"""
Copyright 2025 Zordi, Inc. All rights reserved.

Rollout buffer management for SAC delta action training with two-bucket sampling.

This module implements:
1. Two-bucket sampling: base policy demos vs. residual rollouts
2. Grade-weighted sampling for residual rollouts
3. Age-based decay for residual rollouts
4. Episode indexing and metadata extraction
"""

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import orjson
from rl_configs import RolloutBufferConfig

logger = logging.getLogger(__name__)


class EpisodeMetadata:
    """Metadata for a single rollout episode."""

    def __init__(
        self,
        episode_path: Path,
        model_id: str,
        timestamp: datetime,
        grade: int,
        n_steps: int,
        mean_reward: float,
        age_step: int = 0,
    ):
        self.episode_path = episode_path
        self.model_id = model_id
        self.timestamp = timestamp
        self.grade = grade
        self.n_steps = n_steps
        self.mean_reward = mean_reward
        self.age_step = age_step  # Age step based on model directory ordering

    def __repr__(self) -> str:
        return (
            f"EpisodeMetadata(path={self.episode_path.name}, "
            f"model={self.model_id}, grade={self.grade}, "
            f"age_step={self.age_step})"
        )


class RolloutBuffer:
    """Manages rollout episodes with two-bucket sampling strategy."""

    def __init__(self, dataset_path: Path, config: RolloutBufferConfig):
        """Initialize rollout buffer.

        Args:
            dataset_path: Path to rollout dataset directory
            config: Rollout buffer configuration
        """
        self.dataset_path = Path(dataset_path)
        self.config = config
        self.rng = np.random.default_rng()

        # Episode collections
        self.demo_episodes: List[EpisodeMetadata] = []  # Bucket A: base policy demos
        self.residual_episodes: List[
            EpisodeMetadata
        ] = []  # Bucket B: residual rollouts
        self.validation_episodes: List[EpisodeMetadata] = []  # Held-out validation

        # Load and index episodes
        self._load_episodes()
        self._apply_filtering_and_sampling()

        logger.info(
            f"RolloutBuffer initialized: {len(self.demo_episodes)} demo episodes, "
            f"{len(self.residual_episodes)} residual episodes, "
            f"{len(self.validation_episodes)} validation episodes"
        )

    def _parse_episode_timestamp(self, episode_dir_name: str) -> datetime:
        """Parse timestamp from episode directory name (e.g., '20250707_133107')."""
        try:
            return datetime.strptime(episode_dir_name, "%Y%m%d_%H%M%S")
        except ValueError as e:
            logger.warning(f"Failed to parse timestamp from {episode_dir_name}: {e}")
            # Return a very old timestamp as fallback
            return datetime(2020, 1, 1)

    def _load_grade_from_file(self, episode_path: Path) -> int:
        """Load grade from inference_grade.json file."""
        grade_file = episode_path / "inference_grade.json"
        if not grade_file.exists():
            logger.warning(
                f"No grade file found for {episode_path.name}, using grade 0"
            )
            return 0

        try:
            with grade_file.open("r") as f:
                grade_data = json.load(f)
            grade = int(grade_data.get("feedback", 0))
            return grade
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse grade from {grade_file}: {e}")
            return 0

    def _assign_age_steps(
        self, residual_episodes: List[EpisodeMetadata]
    ) -> List[EpisodeMetadata]:
        """Assign age steps to residual episodes based on model directory ordering.

        Args:
            residual_episodes: List of residual episodes to assign age steps to

        Returns:
            List of episodes with updated age steps
        """
        if not residual_episodes:
            return []

        # Group episodes by model_id
        model_groups = {}
        for episode in residual_episodes:
            if episode.model_id not in model_groups:
                model_groups[episode.model_id] = []
            model_groups[episode.model_id].append(episode)

        # Sort model IDs by their timestamp (newer models get lower age steps)
        # Model IDs are in format "model_YYYYMMDD_HHMMSS"
        def extract_model_timestamp(model_id: str) -> datetime:
            try:
                # Extract timestamp from model_id (e.g., "model_20250707_133107")
                timestamp_str = model_id.replace("model_", "")
                return datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            except ValueError:
                # Fallback for malformed model IDs
                return datetime(2020, 1, 1)

        sorted_model_ids = sorted(
            model_groups.keys(),
            key=extract_model_timestamp,
            reverse=True,  # Most recent first (age_step=0)
        )

        logger.info(
            f"Found {len(sorted_model_ids)} model groups for age step assignment"
        )
        logger.info(f"Model ordering (newest to oldest): {sorted_model_ids}")

        # Assign age steps to episodes
        updated_episodes = []
        for age_step, model_id in enumerate(sorted_model_ids):
            episodes_in_group = model_groups[model_id]
            for episode in episodes_in_group:
                # Create new metadata with updated age step
                updated_episode = EpisodeMetadata(
                    episode_path=episode.episode_path,
                    model_id=episode.model_id,
                    timestamp=episode.timestamp,
                    grade=episode.grade,
                    n_steps=episode.n_steps,
                    mean_reward=episode.mean_reward,
                    age_step=age_step,
                )
                updated_episodes.append(updated_episode)

        logger.info(
            f"Assigned age steps to {len(updated_episodes)} residual episodes "
            f"(age_step range: 0-{len(sorted_model_ids) - 1})"
        )

        return updated_episodes

    def _extract_episode_metadata(
        self, episode_path: Path
    ) -> Optional[EpisodeMetadata]:
        """Extract metadata from an episode directory."""
        observations_file = episode_path / "observations.json"
        if not observations_file.exists():
            return None

        try:
            # Load episode data
            with observations_file.open("rb") as f:
                episode_data = orjson.loads(f.read())

            metadata = episode_data.get("metadata", {})
            observations = episode_data.get("observations", [])

            # Extract basic info
            n_steps = len(observations)
            if n_steps == 0:
                return None

            # Calculate mean reward
            rewards = [obs.get("reward", 0.0) for obs in observations]
            mean_reward = float(np.mean(rewards))

            # Parse timestamp from directory name
            timestamp = self._parse_episode_timestamp(episode_path.name)

            # Load grade
            grade = self._load_grade_from_file(episode_path)

            # Determine model ID from parent directory
            model_id = episode_path.parent.name

            return EpisodeMetadata(
                episode_path=episode_path,
                model_id=model_id,
                timestamp=timestamp,
                grade=grade,
                n_steps=n_steps,
                mean_reward=mean_reward,
            )

        except Exception as e:
            logger.warning(f"Failed to extract metadata from {episode_path}: {e}")
            return None

    def _load_episodes(self) -> None:
        """Load and categorize all episodes from the dataset directory."""
        if not self.dataset_path.exists():
            logger.warning(f"Dataset path does not exist: {self.dataset_path}")
            return

        # Find all episode directories
        episode_paths = []
        for observations_file in self.dataset_path.glob("**/observations.json"):
            episode_paths.append(observations_file.parent)

        logger.info(f"Found {len(episode_paths)} episode directories")

        # Extract metadata for all episodes
        all_episodes = []
        for episode_path in episode_paths:
            metadata = self._extract_episode_metadata(episode_path)
            if metadata is not None:
                all_episodes.append(metadata)

        logger.info(f"Successfully extracted metadata for {len(all_episodes)} episodes")

        # Categorize episodes
        demo_episodes = []
        residual_episodes = []
        for episode in all_episodes:
            if episode.model_id == "base_policy_only":
                demo_episodes.append(episode)
            else:
                # This is a residual rollout from a trained model
                residual_episodes.append(episode)

        # Apply age step logic to residual episodes
        residual_episodes = self._assign_age_steps(residual_episodes)

        self.demo_episodes = demo_episodes
        self.residual_episodes = residual_episodes

        logger.info(
            f"Categorized episodes: {len(self.demo_episodes)} demo, "
            f"{len(self.residual_episodes)} residual"
        )

    def _apply_filtering_and_sampling(self) -> None:
        """Apply filtering and sampling strategies to residual episodes."""
        if not self.residual_episodes:
            logger.info("No residual episodes to filter")
            return

        # Filter residual episodes by grade and age
        filtered_residual = []
        for episode in self.residual_episodes:
            # Grade filter
            if episode.grade < self.config.min_grade:
                continue

            # Age filter
            if episode.age_step > self.config.max_age_steps:
                continue

            filtered_residual.append(episode)

        logger.info(
            f"Filtered residual episodes: {len(filtered_residual)} / "
            f"{len(self.residual_episodes)} passed grade>={self.config.min_grade} "
            f"and age_step<={self.config.max_age_steps}"
        )

        # Hold out best episodes for validation
        if self.config.val_holdout_fraction > 0 and filtered_residual:
            # Sort by grade (descending), then by recency (ascending age_step)
            filtered_residual.sort(key=lambda ep: (-ep.grade, ep.age_step))

            n_holdout = max(
                1, int(len(filtered_residual) * self.config.val_holdout_fraction)
            )
            self.validation_episodes = filtered_residual[:n_holdout]
            self.residual_episodes = filtered_residual[n_holdout:]

            logger.info(
                f"Held out {len(self.validation_episodes)} best episodes for validation"
            )
        else:
            self.residual_episodes = filtered_residual

    def _compute_residual_sampling_weights(self) -> np.ndarray:
        """Compute sampling weights for residual episodes using grade and age."""
        if not self.residual_episodes:
            return np.array([])

        weights = []
        for episode in self.residual_episodes:
            # Grade weight: exp(β * grade)
            grade_weight = math.exp(self.config.grade_weight_beta * episode.grade)

            # Age weight: exp(-λ * age_step)
            age_weight = math.exp(-self.config.age_decay_lambda * episode.age_step)

            # Combined weight
            combined_weight = grade_weight * age_weight
            weights.append(combined_weight)

        weights = np.array(weights)
        # Normalize to probabilities
        weights = weights / weights.sum()

        logger.debug(
            f"Residual sampling weights: min={weights.min():.4f}, "
            f"max={weights.max():.4f}, mean={weights.mean():.4f}"
        )

        return weights

    def sample_episode_paths(self, n_samples: int) -> List[Path]:
        """Sample episode paths using two-bucket strategy.

        Args:
            n_samples: Total number of episodes to sample

        Returns:
            List of episode paths
        """
        if n_samples <= 0:
            return []

        # Determine bucket sizes
        n_demo = int(n_samples * self.config.demo_fraction)
        n_residual = n_samples - n_demo

        sampled_paths = []

        # Sample from demo episodes (bucket A) - uniform sampling
        if n_demo > 0 and self.demo_episodes:
            demo_indices = self.rng.choice(
                len(self.demo_episodes),
                size=min(n_demo, len(self.demo_episodes)),
                replace=n_demo > len(self.demo_episodes),
            )
            for idx in demo_indices:
                sampled_paths.append(self.demo_episodes[idx].episode_path)

        # Sample from residual episodes (bucket B) - weighted sampling
        if n_residual > 0 and self.residual_episodes:
            weights = self._compute_residual_sampling_weights()
            if len(weights) > 0:
                residual_indices = self.rng.choice(
                    len(self.residual_episodes),
                    size=min(n_residual, len(self.residual_episodes)),
                    replace=n_residual > len(self.residual_episodes),
                    p=weights,
                )
                for idx in residual_indices:
                    sampled_paths.append(self.residual_episodes[idx].episode_path)

        logger.debug(
            f"Sampled {len(sampled_paths)} episodes: "
            f"{min(n_demo, len(self.demo_episodes))} demo + "
            f"{min(n_residual, len(self.residual_episodes))} residual"
        )

        return sampled_paths

    def get_validation_episode_paths(self) -> List[Path]:
        """Get validation episode paths."""
        return [ep.episode_path for ep in self.validation_episodes]

    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics for logging."""
        stats = {
            "total_episodes": len(self.demo_episodes) + len(self.residual_episodes),
            "demo_episodes": len(self.demo_episodes),
            "residual_episodes": len(self.residual_episodes),
            "validation_episodes": len(self.validation_episodes),
            "config": {
                "demo_fraction": self.config.demo_fraction,
                "grade_weight_beta": self.config.grade_weight_beta,
                "min_grade": self.config.min_grade,
                "age_decay_lambda": self.config.age_decay_lambda,
                "max_age_steps": self.config.max_age_steps,
            },
        }

        if self.residual_episodes:
            grades = [ep.grade for ep in self.residual_episodes]
            age_steps = [ep.age_step for ep in self.residual_episodes]
            stats["residual_stats"] = {
                "grade_distribution": {
                    "min": min(grades),
                    "max": max(grades),
                    "mean": float(np.mean(grades)),
                },
                "age_step_distribution": {
                    "min": min(age_steps),
                    "max": max(age_steps),
                    "mean": float(np.mean(age_steps)),
                },
            }

        return stats
