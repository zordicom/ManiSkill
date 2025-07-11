"""
Copyright 2025 Zordi, Inc. All rights reserved.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import orjson
import torch
import yaml
from rl_configs import RLConfig
from rollout_buffer import RolloutBuffer
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter, RandomResizedCrop

from zordi_vla.common.normalizers import (
    DeltaActionNormalizer,
    DepthNormalizer,
    ImageNormalizer,
    MaskNormalizer,
    MeanStdNormalizer,
    MinMaxNormalizer,
)
from zordi_vla.configs.configs import (
    ActionConfig,
    DatasetConfig,
    ImageObsConfig,
    ImageType,
    NormalizerType,
    ShapeConfig,
    StateObsConfig,
)
from zordi_vla.utils.logging_utils import setup_logger


class RLDataset(Dataset):
    """Dataset for reinforcement learning rollouts with lazy image loading and rollout
    buffer management.
    """

    def __init__(
        self,
        cfg_rl: RLConfig,
        is_train_split: bool = True,
    ) -> None:
        super().__init__()

        self.logger = setup_logger(__name__)
        self.cfg_rl = cfg_rl
        self.dataset_path = Path(cfg_rl.dataset.path).expanduser().resolve()

        # Initialize rollout buffer for episode management
        self.rollout_buffer = RolloutBuffer(self.dataset_path, cfg_rl.rollout_buffer)

        # Log buffer statistics
        buffer_stats = self.rollout_buffer.get_statistics()
        self.logger.info(f"Rollout buffer statistics: {buffer_stats}")

        # Single file read to infer shape config and get metadata
        (
            self.cfg_shape,
            self.n_obs_steps,
            self.internal_shape_meta,
            self.image_keys,
            self.state_fields,
            self.action_fields,
            self.extra_obs_fields,
        ) = self._infer_configs_and_meta()

        # Extract dimensions
        self.state_dim = 0
        state_config = self.cfg_shape.obs.get("state")
        if isinstance(state_config, StateObsConfig):
            self.state_dim = state_config.dim
        self.action_dim = self.cfg_shape.action.dim

        # Store config values
        self.seed = cfg_rl.dataset.seed
        self.val_ratio = cfg_rl.dataset.val_ratio
        self.normalizer_type = cfg_rl.dataset.normalizer

        # Load statistics
        stat_file = self.dataset_path / cfg_rl.dataset.state_stats_path
        aggregated = orjson.loads(stat_file.read_bytes())

        state_stats_loaded = {
            k: np.array(v) for k, v in aggregated["observation_absolute"].items()
        }
        action_stats_loaded = {
            k: np.array(v) for k, v in aggregated["action_absolute"].items()
        }

        # Load episodes based on rollout buffer sampling
        self.episode_dict: Dict[str, List[Dict[str, Any]]] = {}

        if is_train_split:
            # Use rollout buffer to sample training episodes
            # Sample a reasonable number of episodes for training
            # This could be made configurable in the future
            n_episodes_to_sample = min(200, buffer_stats["total_episodes"])
            sampled_episode_paths = self.rollout_buffer.sample_episode_paths(
                n_episodes_to_sample
            )
            self.logger.info(
                f"Sampled {len(sampled_episode_paths)} episodes for training"
            )
        else:
            # For validation, use held-out episodes from rollout buffer
            sampled_episode_paths = self.rollout_buffer.get_validation_episode_paths()
            self.logger.info(
                f"Using {len(sampled_episode_paths)} held-out episodes for validation"
            )

        # Load selected episodes
        for episode_path in sampled_episode_paths:
            observations_file = episode_path / "observations.json"
            if observations_file.exists():
                try:
                    ep_data = orjson.loads(observations_file.read_bytes())
                    ep_name = episode_path.name
                    observations = ep_data["observations"]

                    # Convert relative image paths to absolute paths (lazy loading)
                    for item in observations:
                        for key in self.image_keys:
                            if key in item and item[key] is not None:
                                item[key] = (episode_path / item[key]).resolve()

                    self.episode_dict[ep_name] = observations
                except Exception as e:
                    self.logger.warning(f"Failed to load episode {episode_path}: {e}")

        # Build step references
        self.step_references: List[Tuple[str, int]] = []
        for ep_name, obs_list in self.episode_dict.items():
            for i in range(len(obs_list)):
                self.step_references.append((ep_name, i))

        # For train/validation split, we already have the right episodes
        # so we use all step references
        self.step_id_list = np.arange(len(self.step_references))
        self.is_train = is_train_split

        self.logger.info(
            f"Loaded {len(self.step_references)} steps from {len(self.episode_dict)} episodes"
        )

        # Initialize normalizers
        self._setup_normalizers(state_stats_loaded, action_stats_loaded)

        # Setup data augmentation
        self._setup_augmentation()

    def _infer_configs_and_meta(
        self,
    ) -> Tuple[
        ShapeConfig,
        int,
        Dict[str, Any],
        List[str],
        Dict[str, Tuple[int, int]],
        Dict[str, Tuple[int, int]],
        List[str],
    ]:
        """Infer all configs and metadata from the first episode in a single file read.

        Returns:
            - ShapeConfig
            - n_obs_steps
            - internal_shape_meta
            - image_keys
            - state_fields mapping
            - action_fields mapping
            - extra_obs_fields list
        """
        first_json = next(self.dataset_path.glob("**/observations.json"))
        with first_json.open("rb") as f:
            meta_json = orjson.loads(f.read())

        meta = meta_json["metadata"]
        shape_meta = meta["shape_meta"]

        # Extract n_obs_steps
        n_obs_steps = meta.get("n_obs_steps", 1)

        # Extract field definitions
        state_fields = meta.get("state_fields", {})
        action_fields = meta.get("action_fields", {})

        # Build ShapeConfig from field definitions
        if state_fields:
            # Calculate state dimension from field definitions
            state_dim = max(field_range[1] for field_range in state_fields.values())
        else:
            # Fallback to shape_meta if no field definitions
            state_dim = shape_meta["obs"]["state"]["dim"]

        action_dim = (
            max(field_range[1] for field_range in action_fields.values())
            if action_fields
            else shape_meta["action"]["dim"]
        )

        action_cfg = ActionConfig(dim=action_dim)

        obs_cfg: Dict[str, Any] = {}
        obs_cfg["state"] = StateObsConfig(dim=state_dim)

        # Process images and build image keys list simultaneously
        image_keys: List[str] = []
        for img_key, img_info in shape_meta["obs"]["images"].items():
            # Skip depth images for now as requested
            if "depth" in img_key.lower():
                continue

            channels = 3 if img_info["dtype"] == "rgb" else 1
            img_type = ImageType(img_info["dtype"])
            image_size = tuple(img_info["image_size"])  # [H, W]
            obs_cfg[img_key] = ImageObsConfig(
                channels=channels, image_size=image_size, image_type=img_type
            )
            # Add to image keys if it's a supported image type
            if img_type in {ImageType.RGB, ImageType.DEPTH, ImageType.MASK}:
                image_keys.append(img_key)

        cfg_shape = ShapeConfig(action=action_cfg, obs=obs_cfg)

        # Detect extra observation fields from first episode
        extra_obs_fields = []
        if meta_json.get("observations"):
            first_obs = meta_json["observations"][0]
            if "extra_obs" in first_obs and isinstance(first_obs["extra_obs"], dict):
                extra_obs_fields = list(first_obs["extra_obs"].keys())

        # Build internal shape meta efficiently
        obs_meta_for_model: Dict[str, Dict[str, Any]] = {}
        for key, mod_config in cfg_shape.obs.items():
            if isinstance(mod_config, StateObsConfig):
                obs_meta_for_model[key] = {
                    "dim": mod_config.dim,
                    "state_type": mod_config.state_type,
                    "shape": [mod_config.dim],
                }
            elif isinstance(mod_config, ImageObsConfig):
                obs_meta_for_model[key] = {
                    "channels": mod_config.channels,
                    "image_size": mod_config.image_size,
                    "image_type": mod_config.image_type,
                    "shape": [
                        mod_config.channels,
                        mod_config.image_size[0],
                        mod_config.image_size[1],
                    ],
                }

        # Add expert_action to obs_meta_for_model for SAC
        obs_meta_for_model["expert_action"] = {
            "shape": [action_dim],
        }

        # Add extra observation fields
        for field in extra_obs_fields:
            # Infer dimension from first observation
            if meta_json.get("observations"):
                first_obs = meta_json["observations"][0]
                if "extra_obs" in first_obs and field in first_obs["extra_obs"]:
                    field_data = first_obs["extra_obs"][field]
                    if isinstance(field_data, list):
                        obs_meta_for_model[field] = {
                            "shape": [len(field_data)],
                        }

        internal_shape_meta = {
            "obs": obs_meta_for_model,
            "action": {"shape": [action_dim]},
        }

        return (
            cfg_shape,
            n_obs_steps,
            internal_shape_meta,
            image_keys,
            state_fields,
            action_fields,
            extra_obs_fields,
        )

    def _convert_state_dict_to_vector(self, state_dict: Dict[str, Any]) -> np.ndarray:
        """Convert state dictionary to flattened vector using field definitions."""
        # Initialize vector with correct dimension
        state_vector = np.zeros(self.state_dim, dtype=np.float32)

        # Fill vector using field definitions
        for field_name, field_range in self.state_fields.items():
            if field_name in state_dict:
                start_idx, end_idx = field_range
                field_data = np.array(state_dict[field_name], dtype=np.float32)
                expected_len = end_idx - start_idx

                if len(field_data) != expected_len:
                    raise ValueError(
                        f"Field '{field_name}' has length {len(field_data)}, "
                        f"expected {expected_len}"
                    )

                state_vector[start_idx:end_idx] = field_data
            else:
                self.logger.warning(f"Missing field '{field_name}' in state dictionary")

        return state_vector

    def _convert_action_dict_to_vector(self, action_dict: Dict[str, Any]) -> np.ndarray:
        """Convert action dictionary to flattened vector using field definitions."""
        # Initialize vector with correct dimension
        action_vector = np.zeros(self.action_dim, dtype=np.float32)

        # Fill vector using field definitions
        for field_name, field_range in self.action_fields.items():
            if field_name in action_dict:
                start_idx, end_idx = field_range
                field_data = np.array(action_dict[field_name], dtype=np.float32)
                expected_len = end_idx - start_idx

                if len(field_data) != expected_len:
                    raise ValueError(
                        f"Field '{field_name}' has length {len(field_data)}, "
                        f"expected {expected_len}"
                    )

                action_vector[start_idx:end_idx] = field_data
            else:
                self.logger.warning(
                    f"Missing field '{field_name}' in action dictionary"
                )

        return action_vector

    def _setup_normalizers(
        self, state_stats: Dict[str, np.ndarray], action_stats: Dict[str, np.ndarray]
    ) -> None:
        """Setup all normalizers."""
        if self.normalizer_type == NormalizerType.MEANSTD:
            self.state_normalizer = MeanStdNormalizer(state_stats)
            # Expert actions use absolute action range (for input to SAC)
            self.expert_action_normalizer = MeanStdNormalizer(action_stats)
        else:
            self.state_normalizer = MinMaxNormalizer(state_stats)
            # Expert actions use absolute action range (for input to SAC)
            self.expert_action_normalizer = MinMaxNormalizer(action_stats)

        # For delta actions, use a specialized normalizer based on the delta_max range
        # instead of the absolute action range
        delta_max = self.cfg_rl.sac.delta_action_max_range
        self.action_normalizer = DeltaActionNormalizer(delta_max)

        self.rgb_normalizer = ImageNormalizer()
        self.mask_normalizer = MaskNormalizer(
            num_actual_classes=self.cfg_rl.dataset.mask_num_classes
        )
        self.depth_normalizer = DepthNormalizer(
            min_depth_m=self.cfg_rl.dataset.depth_clipping_range[0],
            max_depth_m=self.cfg_rl.dataset.depth_clipping_range[1],
            depth_unit_scale=self.cfg_rl.dataset.depth_unit_scale,
        )

    def _setup_augmentation(self) -> None:
        """Setup data augmentation transforms and parameters."""
        self.color_jitter = ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.03
        )

        # Data augmentation parameters
        self.noise_std_state = self.cfg_rl.dataset.noise_std_state
        self.noise_std_action = self.cfg_rl.dataset.noise_std_action
        self.random_resized_crop_scale = self.cfg_rl.dataset.random_resized_crop_scale
        self.random_resized_crop_ratio = self.cfg_rl.dataset.random_resized_crop_ratio

        # Selective noise augmentation
        self.selective_noise_state = self.cfg_rl.dataset.selective_noise_state
        if self.selective_noise_state:
            self.logger.info(
                "Selective noise augmentation enabled for state fields: %s "
                "(applied in physical units before normalization)",
                list(self.selective_noise_state.keys()),
            )

    def _apply_selective_noise(
        self,
        tensor: torch.Tensor,
        selective_noise_config: Optional[Dict[str, Any]],
        vector_fields: Optional[Dict[str, Tuple[int, int]]],
        global_noise_std: float,
    ) -> torch.Tensor:
        """
        Apply selective noise augmentation to specific vector fields.

        NOTE: This method is called BEFORE normalization, so noise values are applied
        in the original physical units (e.g., meters for poses, radians for joints).

        Args:
            tensor: Input tensor to apply noise to (shape: [T, D] or [D]) - RAW values
            selective_noise_config: Configuration for selective noise per field
            vector_fields: Dictionary mapping field names to (start, end) indices
            global_noise_std: Global noise standard deviation (fallback) - in physical units

        Returns:
            Tensor with selective noise applied (still in physical units)
        """
        if not self.is_train or tensor.numel() == 0:
            return tensor

        # If no selective noise config, apply global noise
        if selective_noise_config is None or vector_fields is None:
            if global_noise_std > 0:
                return tensor + torch.randn_like(tensor) * global_noise_std
            return tensor

        # Apply selective noise to specific fields
        result = tensor.clone()

        for field_name, noise_config in selective_noise_config.items():
            if field_name not in vector_fields:
                self.logger.warning(
                    f"Field '{field_name}' not found in vector_fields. "
                    f"Available fields: {list(vector_fields.keys())}"
                )
                continue

            start_idx, end_idx = vector_fields[field_name]

            # Generate noise based on configuration
            if isinstance(noise_config, (int, float)):
                # Simple Gaussian noise
                noise = torch.randn_like(result[..., start_idx:end_idx]) * noise_config
            elif isinstance(noise_config, dict):
                noise_type = noise_config.get("type", "gaussian")

                if noise_type == "uniform":
                    # Uniform noise in [-range, +range]
                    noise_range = noise_config.get("range", 0.05)
                    noise = (
                        torch.rand_like(result[..., start_idx:end_idx]) * 2 - 1
                    ) * noise_range
                elif noise_type == "gaussian":
                    # Gaussian noise with specified std
                    noise_std = noise_config.get("std", 0.01)
                    noise = torch.randn_like(result[..., start_idx:end_idx]) * noise_std
                else:
                    self.logger.warning(
                        f"Unknown noise type '{noise_type}', using Gaussian"
                    )
                    noise_std = noise_config.get("std", 0.01)
                    noise = torch.randn_like(result[..., start_idx:end_idx]) * noise_std
            else:
                self.logger.warning(
                    f"Invalid noise config for field '{field_name}': {noise_config}"
                )
                continue

            # Apply noise to the specific field
            result[..., start_idx:end_idx] += noise

        # Apply global noise to remaining fields if specified
        if global_noise_std > 0:
            # Create a mask for fields that don't have selective noise
            mask = torch.ones_like(result, dtype=torch.bool)
            for field_name in selective_noise_config.keys():
                if field_name in vector_fields:
                    start_idx, end_idx = vector_fields[field_name]
                    mask[..., start_idx:end_idx] = False

            # Apply global noise to masked regions
            if mask.any():
                global_noise = torch.randn_like(result) * global_noise_std
                result = torch.where(mask, result + global_noise, result)

        return result

    def get_shape_meta(self) -> Dict[str, Any]:
        """Return the shape meta dictionary used by models."""
        return self.internal_shape_meta

    def get_validation_dataset(self) -> Optional["RLDataset"]:
        """Return a view of this dataset representing the validation split."""
        if not self.is_train:
            return None

        # Create validation dataset with is_train_split=False
        val_ds = RLDataset(cfg_rl=self.cfg_rl, is_train_split=False)
        return val_ds

    def _load_and_process_image(
        self,
        img_path: Optional[Path],
        mod_meta: Dict[str, Any],
        mod_type_enum: ImageType,
        target_h: int,
        target_w: int,
    ) -> torch.Tensor:
        """Load and preprocess a single image sample."""
        if img_path is None:
            return torch.zeros(
                (mod_meta["shape"][0], target_h, target_w), dtype=torch.float32
            )

        img_np = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)

        # Color space conversion
        if mod_type_enum == ImageType.RGB:
            if img_np.ndim == 2:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            elif img_np.shape[2] == 4:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_BGRA2RGB)
            else:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        elif mod_type_enum in {ImageType.MASK, ImageType.DEPTH}:
            if img_np.ndim == 3:
                img_np = img_np[:, :, 0]
            if mod_meta["shape"][0] == 1 and img_np.ndim == 2:
                img_np = np.expand_dims(img_np, axis=-1)

        # Resize if necessary
        h_orig, w_orig = img_np.shape[:2]
        if h_orig != target_h or w_orig != target_w:
            interpolation = (
                cv2.INTER_NEAREST
                if mod_type_enum in {ImageType.MASK, ImageType.DEPTH}
                else cv2.INTER_LINEAR
            )
            img_np = cv2.resize(
                img_np, (target_w, target_h), interpolation=interpolation
            )

        # Normalize RGB to [0, 1]
        if mod_type_enum == ImageType.RGB and img_np.dtype != np.float32:
            img_np = img_np.astype(np.float32) / 255.0

        # Ensure channel dimension
        if (
            mod_type_enum in {ImageType.MASK, ImageType.DEPTH}
            and img_np.ndim == 2
            and mod_meta["shape"][0] == 1
        ):
            img_np = np.expand_dims(img_np, axis=-1)

        img_t = torch.from_numpy(np.moveaxis(img_np, -1, 0).copy())

        # Augmentation (train split only)
        if self.is_train and mod_type_enum == ImageType.RGB:
            img_t = RandomResizedCrop(
                size=(target_h, target_w),
                scale=self.random_resized_crop_scale,
                ratio=self.random_resized_crop_ratio,
            )(img_t)
            img_t = self.color_jitter(img_t)

        return img_t

    def __len__(self) -> int:
        return len(self.step_id_list)

    def _build_processed_obs(
        self, step_item: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """Convert a raw observation entry into a dictionary of **normalized** tensors.

        This utility centralizes the image + state loading logic so that it can be
        reused for both *current* and *next* observations.  All tensors are placed
        on CPU, leaving the `DataLoader` (or the training loop) free to move them
        to the appropriate device later on.
        """
        # Process state history ------------------------------------------------
        states_data = step_item["states"]

        # Convert structured states to flat vectors
        if self.state_fields and isinstance(states_data[0], dict):
            # New structured format - convert each state dict to vector
            state_vectors = []
            for state_dict in states_data:
                state_vector = self._convert_state_dict_to_vector(state_dict)
                state_vectors.append(state_vector)
            state_t = torch.from_numpy(np.stack(state_vectors, dtype=np.float32))
        else:
            # Legacy flat format
            state_t = torch.from_numpy(np.stack(states_data, dtype=np.float32))

        # Apply selective noise augmentation BEFORE normalization
        if self.is_train:
            state_t = self._apply_selective_noise(
                state_t,
                self.selective_noise_state,
                self.state_fields,
                self.noise_std_state,
            )

        norm_state = self.state_normalizer.normalize(state_t)

        # Optional noise augmentation for state history -----------------------
        if self.is_train and self.noise_std_state > 0 and norm_state.numel() > 0:
            # This global noise is now applied *after* selective noise and normalization
            # which is inconsistent with bc_dataset. Let's rely on _apply_selective_noise
            # to handle global noise on non-specified fields.
            # norm_state += torch.randn_like(norm_state) * self.noise_std_state
            pass

        # Process images -------------------------------------------------------
        norm_images: Dict[str, torch.Tensor] = {}
        for key in self.image_keys:
            mod_meta = self.internal_shape_meta["obs"][key]
            _, target_h, target_w = mod_meta["shape"]
            mod_type = mod_meta["image_type"]
            img_path = step_item.get(key)

            processed = self._load_and_process_image(
                img_path, mod_meta, mod_type, target_h, target_w
            )
            # Add batch dimension so image shape becomes [1, C, H, W]
            processed = processed.unsqueeze(0)

            # Normalize appropriately -----------------------------------------
            if mod_type == ImageType.RGB:
                norm_images[key] = self.rgb_normalizer.normalize(processed)
            elif mod_type == ImageType.MASK:
                norm_images[key] = self.mask_normalizer.normalize(processed.float())
            elif mod_type == ImageType.DEPTH:
                norm_images[key] = self.depth_normalizer.normalize(processed)
            else:
                norm_images[key] = processed

        # Expert action --------------------------------------------------------
        expert_action_raw = step_item["expert_action"]

        # Convert structured action to flat vector if needed
        if self.action_fields and isinstance(expert_action_raw, dict):
            expert_action_raw = self._convert_action_dict_to_vector(expert_action_raw)
        elif isinstance(expert_action_raw, list):
            expert_action_raw = np.array(expert_action_raw, dtype=np.float32)

        expert_action_t = torch.from_numpy(expert_action_raw).float()
        norm_expert_action = self.expert_action_normalizer.normalize(expert_action_t)

        # Extra observations ---------------------------------------------------
        extra_obs_tensors: Dict[str, torch.Tensor] = {}
        if "extra_obs" in step_item and isinstance(step_item["extra_obs"], dict):
            for field in self.extra_obs_fields:
                if field in step_item["extra_obs"]:
                    field_data = step_item["extra_obs"][field]
                    if isinstance(field_data, list):
                        field_tensor = torch.tensor(field_data, dtype=torch.float32)
                        extra_obs_tensors[field] = field_tensor

        # Assemble -------------------------------------------------------------
        final_obs: Dict[str, torch.Tensor] = {
            "state": norm_state,
            "expert_action": norm_expert_action,
        }
        final_obs.update(norm_images)
        final_obs.update(extra_obs_tensors)
        return final_obs

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        master_idx = self.step_id_list[idx]
        ep_name, step_idx = self.step_references[master_idx]
        step_item = self.episode_dict[ep_name][step_idx]

        # Build current and next observations ---------------------------------
        current_obs = self._build_processed_obs(step_item)

        # Determine *next* observation (bootstrap target)
        next_step_idx = step_idx + 1
        if next_step_idx < len(self.episode_dict[ep_name]):
            next_step_item = self.episode_dict[ep_name][next_step_idx]
            next_obs = self._build_processed_obs(next_step_item)
        else:
            # Terminal frame – create zero placeholders with correct shapes
            next_obs: Dict[str, torch.Tensor] = {}
            for key, tensor in current_obs.items():
                next_obs[key] = torch.zeros_like(tensor)

        # Load residual (agent) action ----------------------------------------
        residual_action_raw = step_item["residual_action"]

        # Convert structured action to flat vector if needed
        if self.action_fields and isinstance(residual_action_raw, dict):
            residual_action_raw = self._convert_action_dict_to_vector(
                residual_action_raw
            )
        elif isinstance(residual_action_raw, list):
            residual_action_raw = np.array(residual_action_raw, dtype=np.float32)

        action_t = torch.from_numpy(residual_action_raw).float()
        norm_action = self.action_normalizer.normalize(action_t)

        # Optional noise augmentation ----------------------------------------
        if self.is_train and self.noise_std_action > 0:
            norm_action += torch.randn_like(norm_action) * self.noise_std_action

        # Scalar values --------------------------------------------------------
        reward = torch.tensor(float(step_item["reward"]), dtype=torch.float32)
        done = torch.tensor(bool(step_item["done"]), dtype=torch.bool)
        terminated = torch.tensor(bool(step_item["terminated"]), dtype=torch.bool)

        return {
            "obs": current_obs,
            "next_obs": next_obs,
            "action": norm_action,
            "reward": reward,
            "done": done,
            "terminated": terminated,
        }


# =====================================================================
# Quick usage check ----------------------------------------------------
# =====================================================================


def main():  # noqa: D103
    logger = setup_logger("rl_dataset.main")

    parser = argparse.ArgumentParser(description="Sanity-check RLDataset loading")
    parser.add_argument(
        "--cfg",
        type=str,
        default="playground/rl/rl_galaxea_ppo.yaml",
        help="Path to RL dataset YAML configuration "
        "(default: playground/rl/rl_galaxea_ppo.yaml)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run initialization benchmark",
    )
    args = parser.parse_args()

    cfg_path = Path(args.cfg).expanduser().resolve()
    if not cfg_path.exists():
        logger.error("Config file not found: %s", cfg_path)
        sys.exit(1)

    # -----------------------------------------------------------------
    # Parse YAML into RLConfig / DatasetConfig -------------------------
    # -----------------------------------------------------------------
    with cfg_path.open("r", encoding="utf-8") as fp:
        raw_cfg = yaml.safe_load(fp)

    # Build DatasetConfig ------------------------------------------------
    ds_cfg_dict = raw_cfg["dataset"]
    dataset_cfg = DatasetConfig(**ds_cfg_dict)

    rl_cfg = RLConfig(
        bc_config_path=raw_cfg.get("bc_config_path", ""), dataset=dataset_cfg
    )

    # -----------------------------------------------------------------
    # Instantiate dataset & print sanity info --------------------------
    # -----------------------------------------------------------------
    if args.benchmark:
        logger.info("Running initialization benchmark...")

        # Warm up
        ds = RLDataset(cfg_rl=rl_cfg, is_train_split=True)

        # Benchmark multiple runs
        times = []
        for i in range(5):
            start_time = time.time()
            ds = RLDataset(cfg_rl=rl_cfg, is_train_split=True)
            end_time = time.time()
            times.append(end_time - start_time)
            logger.info(f"Run {i + 1}: {times[-1]:.3f}s")

        avg_time = sum(times) / len(times)
        logger.info(
            f"Average initialization time: {avg_time:.3f}s "
            f"± {max(times) - min(times):.3f}s"
        )
    else:
        ds = RLDataset(cfg_rl=rl_cfg, is_train_split=True)

    logger.info("Dataset instantiated with %d training samples", len(ds))
    sample = ds[0]
    logger.info(
        "Sample keys: %s | obs keys: %s | next_obs keys: %s",
        list(sample.keys()),
        list(sample["obs"].keys()),
        list(sample["next_obs"].keys()),
    )
    logger.info(
        "Action shape: %s | Reward tensor: %s", sample["action"].shape, sample["reward"]
    )

    # Log shape metadata and field definitions
    shape_meta = ds.get_shape_meta()
    logger.info("Shape metadata: %s", shape_meta)
    if hasattr(ds, "state_fields") and ds.state_fields:
        logger.info("State fields: %s", ds.state_fields)
    if hasattr(ds, "action_fields") and ds.action_fields:
        logger.info("Action fields: %s", ds.action_fields)
    if hasattr(ds, "extra_obs_fields") and ds.extra_obs_fields:
        logger.info("Extra observation fields: %s", ds.extra_obs_fields)

    # Test state/action conversion if field definitions exist
    if hasattr(ds, "state_fields") and ds.state_fields:
        logger.info("Testing state/action conversions...")
        # Test with sample data from first episode
        first_ep = next(iter(ds.episode_dict.values()))
        if first_ep and isinstance(first_ep[0]["states"][0], dict):
            logger.info("✓ Structured state format detected")
            test_state = first_ep[0]["states"][0]
            state_vector = ds._convert_state_dict_to_vector(test_state)
            logger.info(
                f"State conversion: {len(test_state)} fields → {len(state_vector)} dim vector"
            )

        if first_ep and isinstance(first_ep[0]["expert_action"], dict):
            logger.info("✓ Structured action format detected")
            test_action = first_ep[0]["expert_action"]
            action_vector = ds._convert_action_dict_to_vector(test_action)
            logger.info(
                f"Action conversion: {len(test_action)} fields → {len(action_vector)} dim vector"
            )

    logger.info("✓ RLDataset quick check successful.")


if __name__ == "__main__":
    main()
