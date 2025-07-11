#!/usr/bin/env python3
"""
Copyright 2025 Zordi, Inc. All rights reserved.

Light-weight WebSocket server that hosts a trained SAC Delta Action policy via the
zordi_policy_rpc.direct transport. The SAC model works in combination with
a UnifiedPolicyModel to provide fine-tuned actions.

The inference process:
1. Get base action from UnifiedPolicyModel
2. If SAC model is available:
   - Construct multimodal observation for SAC (including expert_action)
   - Get SAC delta prediction
   - Return final_action = base_action + sac_delta
3. If SAC model is not available:
   - Return final_action = base_action + zero_delta

This allows the same serving infrastructure to be used whether or not
a trained SAC delta action model is available.

Run with SAC model (deterministic, for production):
    python playground/rl/residual_rl/serve_delta_action.py \
        --expert-config config/galaxea/box_pnp_25hz/galaxea_act_25hz_modular.yaml \
        --sac-config playground/rl/residual_rl/rl_galaxea_sac_box_pnp.yaml \
        --sac-checkpoint path/to/sac_checkpoint.pt \
        [--host 0.0.0.0] [--port 10014]

Run with SAC model (stochastic, for data collection):
    python playground/rl/residual_rl/serve_delta_action.py \
        --expert-config config/galaxea/box_pnp_25hz/galaxea_act_25hz_modular.yaml \
        --sac-config playground/rl/residual_rl/rl_galaxea_sac_box_pnp.yaml \
        --sac-checkpoint path/to/sac_checkpoint.pt \
        --stochastic-rollouts \
        [--log-std-scale 1.0] \
        [--host 0.0.0.0] [--port 10014]

Run without SAC model (base policy only):
    python playground/rl/residual_rl/serve_delta_action.py \
        --expert-config config/galaxea/box_pnp_25hz/galaxea_act_25hz_modular.yaml \
        [--host 0.0.0.0] [--port 10014]

Stochastic rollout options:
    --stochastic-rollouts: Enable stochastic sampling for data collection
    --log-std-scale: Control exploration level (default: 1.0)
        - Use 0.5 for less exploration (more conservative)
        - Use 1.5 for more exploration (more diverse data)
        - Use 2.0 for maximum exploration (very diverse but potentially noisy)
"""

import argparse
import asyncio
import logging
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch
import yaml
from rl_configs import RLConfig
from rl_dataset import RLDataset
from try_sac_delta_action import SACDeltaAction
from zordi_policy_rpc.direct.metadata import Metadata
from zordi_policy_rpc.direct.server import DirectServer
from zordi_policy_rpc.direct.server.interface import Server

from zordi_vla.common.normalizers import (
    DeltaActionNormalizer,
    DepthNormalizer,
    ImageNormalizer,
    MaskNormalizer,
    MeanStdNormalizer,
    MinMaxNormalizer,
)
from zordi_vla.configs.configs import (
    ImageObsConfig,
    ImageType,
    StateObsConfig,
    VectorFieldConfig,
)
from zordi_vla.models.policies.unified_policy_model import UnifiedPolicyModel
from zordi_vla.utils.io_utils import load_modular_config

try:
    import cattrs
except ImportError as exc:
    logging.error("Missing dependency: %s", exc)
    raise

import math  # Added for log_std scaling computations

# -----------------------------------------------------------------------------
# SAC Delta Action policy handler
# -----------------------------------------------------------------------------


class SACDeltaPolicyServer(Server):
    """Request handler that wraps a SAC Delta Action policy with optional SAC
    components.
    """

    def __init__(
        self,
        expert_config_path: str,
        sac_config_path: Optional[str] = None,
        sac_checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        action_offset: int = 0,
        stochastic_rollouts: bool = False,
        log_std_scale: float = 1.0,
    ) -> None:
        """Initialize the SAC Delta Action policy server.

        Args:
            expert_config_path: Path to the expert model configuration
            sac_config_path: Optional path to the SAC YAML configuration
            sac_checkpoint_path: Optional path to the SAC checkpoint file
            device: Device to run inference on
            action_offset: Offset for selecting action from action chunk (default: 0)
            stochastic_rollouts: Whether to use stochastic sampling for data collection
            log_std_scale: Scale factor for log_std to control exploration level (default: 1.0)
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.has_sac = sac_config_path is not None and sac_checkpoint_path is not None
        self.action_offset = action_offset
        self.stochastic_rollouts = stochastic_rollouts
        self.log_std_scale = log_std_scale

        # Load expert configuration (always required)
        expert_config_path_obj = Path(expert_config_path).expanduser().resolve()
        if not expert_config_path_obj.exists():
            raise FileNotFoundError(
                f"Expert config not found: {expert_config_path_obj}"
            )

        self.expert_config = load_modular_config(str(expert_config_path_obj))

        # Extract field definitions from config
        state_obs_config = self.expert_config.shape_meta.obs["state"]
        assert isinstance(state_obs_config, StateObsConfig), (
            "obs['state'] must be a StateObsConfig"
        )
        assert state_obs_config.vector_fields is not None, "State vector_fields missing"
        self.state_fields = state_obs_config.vector_fields.fields

        assert self.expert_config.shape_meta.action.vector_fields is not None, (
            "Action vector_fields missing"
        )
        self.action_fields = self.expert_config.shape_meta.action.vector_fields.fields

        # Validate field definitions
        self._validate_field_definitions()

        # Load SAC configuration if provided
        if sac_config_path:
            sac_cfg_path = Path(sac_config_path).expanduser().resolve()
            if not sac_cfg_path.exists():
                raise FileNotFoundError(f"SAC config not found: {sac_cfg_path}")

            with sac_cfg_path.open("r", encoding="utf-8") as fp:
                raw_sac_cfg: Dict[str, Any] = yaml.safe_load(fp)

            self.rl_cfg = cattrs.structure(raw_sac_cfg, RLConfig)
            self.raw_sac_cfg = raw_sac_cfg
        else:
            self.rl_cfg = None
            self.raw_sac_cfg = None

        # Get observation shapes and dimensions
        image_keys = [
            k
            for k, v in self.expert_config.shape_meta.obs.items()
            if isinstance(v, ImageObsConfig)
            and v.image_type in {ImageType.RGB, ImageType.MASK, ImageType.DEPTH}
        ]
        self.camera_keys = image_keys

        # Initialize SAC agent if configuration is available
        if self.has_sac and self.rl_cfg and sac_checkpoint_path:
            # Create a dummy RLDataset to get shape metadata
            dummy_dataset = RLDataset(cfg_rl=self.rl_cfg, is_train_split=False)
            shape_meta = dummy_dataset.get_shape_meta()

            # Initialize SAC agent
            logging.info("Initializing SAC Delta Action agent...")
            self.sac_agent = SACDeltaAction(shape_meta, self.rl_cfg, self.device)

            # Load SAC checkpoint
            logging.info(f"Loading SAC checkpoint: {sac_checkpoint_path}")
            self._load_sac_checkpoint(sac_checkpoint_path)

            # Setup normalizers for SAC
            self._setup_sac_normalizers(dummy_dataset)

            self.model_id = self._extract_model_id(sac_checkpoint_path)
        else:
            self.sac_agent = None
            self.model_id = "base_policy_only"
            logging.info("No SAC model - running base policy only")

            # Still create delta action normalizer for consistency
            if self.rl_cfg:
                delta_max = self.rl_cfg.sac.delta_action_max_range
                self.sac_action_normalizer = DeltaActionNormalizer(delta_max)
                logging.info(
                    f"Created delta action normalizer with delta_max={delta_max}"
                )
            else:
                # Use default delta_max when no config is available
                default_delta_max = 0.05
                self.sac_action_normalizer = DeltaActionNormalizer(default_delta_max)
                logging.info(
                    f"Created default delta action normalizer with delta_max={default_delta_max}"
                )

        # Load UnifiedPolicyModel (expert) - always required
        logging.info("Loading UnifiedPolicyModel...")
        self.unified_policy = UnifiedPolicyModel(self.expert_config)

        # Initialize normalizers for the unified policy
        self._initialize_unified_policy_normalizers()

        # Try to load expert checkpoint if available
        expert_ckpt_path = getattr(
            self.expert_config.inference, "resolved_model_path", None
        )
        if expert_ckpt_path and Path(expert_ckpt_path).exists():
            logging.info(f"Loading expert checkpoint: {expert_ckpt_path}")
            checkpoint = torch.load(
                expert_ckpt_path, map_location=self.device, weights_only=False
            )

            # Extract state dict properly
            raw_sd = checkpoint.get(
                "model_state_dict", checkpoint.get("state_dict", checkpoint)
            )
            if isinstance(raw_sd, dict):
                state_dict = {
                    k.removeprefix("_orig_mod."): v for k, v in raw_sd.items()
                }
                self.unified_policy.load_state_dict(state_dict)
                logging.info("✅ Loaded UnifiedPolicyModel checkpoint")
            else:
                logging.warning(f"Unexpected checkpoint format: {type(raw_sd)}")

            # Initialize normalizers from checkpoint if available
            self._initialize_normalizers_from_checkpoint(checkpoint)
        else:
            logging.warning(
                f"Expert checkpoint not found or not specified: {expert_ckpt_path}"
            )

        self.unified_policy.to(self.device)
        self.unified_policy.eval()

        # Expose key temporal parameters
        self.horizon: int = self.expert_config.horizon
        self.n_obs_steps: int = self.expert_config.n_obs_steps
        self.n_action_steps: int = self.expert_config.n_action_steps

        logging.info(
            f"Model parameters: n_obs_steps={self.n_obs_steps}, horizon={self.horizon}"
        )

        # Get expected image size
        sample_img_config = next(
            (
                v
                for v in self.expert_config.shape_meta.obs.values()
                if isinstance(v, ImageObsConfig)
            ),
            None,
        )
        self.image_size = (
            list(sample_img_config.image_size) if sample_img_config else [224, 224]
        )

        # Log configuration
        mode = "SAC + Base Policy" if self.has_sac else "Base Policy Only"
        sampling_mode = "Stochastic" if self.stochastic_rollouts else "Deterministic"
        logging.info(f"🚀 Running in {mode} mode with {sampling_mode} sampling")
        if self.stochastic_rollouts and self.log_std_scale != 1.0:
            logging.info(f"📊 Log-std scaling factor: {self.log_std_scale}")
        logging.info(f"Model ID: {self.model_id}")
        logging.info(f"Expected camera keys: {self.camera_keys}")
        logging.info(f"Expected image size: {self.image_size}")

    def _initialize_unified_policy_normalizers(self) -> None:
        """Initialize normalizers for the unified policy model."""
        # Initialize basic normalizers
        self.rgb_normalizer = ImageNormalizer()
        self.mask_normalizer = MaskNormalizer(
            num_actual_classes=getattr(
                self.expert_config.dataset, "mask_num_classes", 256
            )
        )
        self.depth_normalizer = DepthNormalizer(
            min_depth_m=getattr(
                self.expert_config.dataset, "depth_clipping_range", [0.1, 10.0]
            )[0],
            max_depth_m=getattr(
                self.expert_config.dataset, "depth_clipping_range", [0.1, 10.0]
            )[1],
            depth_unit_scale=getattr(
                self.expert_config.dataset, "depth_unit_scale", 1000.0
            ),
        )

        # Initialize state and action normalizers
        # (will be set from checkpoint if available)
        self.state_normalizer = None
        self.action_normalizer = None

    def _initialize_normalizers_from_checkpoint(
        self, checkpoint: Dict[str, Any]
    ) -> None:
        """Initialize normalizers from checkpoint data."""
        stats = checkpoint.get("dataset_stats", {})

        # Initialize state normalizer
        state_stats = stats.get("state_normalizer")
        if state_stats and isinstance(state_stats, dict):
            if state_stats.get("type") == "meanstd":
                data = {
                    "mean": np.array(state_stats["mean"]),
                    "std": np.array(state_stats["std"]),
                }
                self.state_normalizer = MeanStdNormalizer(data)
            else:
                data = {
                    "min": np.array(state_stats["min"]),
                    "max": np.array(state_stats["max"]),
                }
                self.state_normalizer = MinMaxNormalizer(data)

        # Initialize action normalizer
        action_stats = stats.get("action_normalizer")
        if action_stats and isinstance(action_stats, dict):
            if action_stats.get("type") == "meanstd":
                data = {
                    "mean": np.array(action_stats["mean"]),
                    "std": np.array(action_stats["std"]),
                }
                self.action_normalizer = MeanStdNormalizer(data)
            else:
                data = {
                    "min": np.array(action_stats["min"]),
                    "max": np.array(action_stats["max"]),
                }
                self.action_normalizer = MinMaxNormalizer(data)

        # Set normalizers on the unified policy model
        self.unified_policy.set_normalizers(
            action_normalizer=self.action_normalizer,
            state_normalizer=self.state_normalizer,
            rgb_normalizer=self.rgb_normalizer,
            mask_normalizer=self.mask_normalizer,
            depth_normalizer=self.depth_normalizer,
        )

        logging.info("✅ Normalizers initialized from checkpoint")

    def _apply_log_std_scaling(self) -> None:
        """Apply log_std scaling to the SAC policy for controlled exploration."""
        if not self.has_sac or self.sac_agent is None or self.log_std_scale == 1.0:
            return

        # Scale the log_std layer weights and biases
        with torch.no_grad():
            if hasattr(self.sac_agent.policy, "log_std_layer"):
                # Compute additive offset in log space so that std -> std * scale
                offset = math.log(self.log_std_scale)

                # Adjust bias to increase/decrease log_std uniformly
                if self.sac_agent.policy.log_std_layer.bias is not None:
                    self.sac_agent.policy.log_std_layer.bias.data += offset

                logging.info(
                    "✅ Applied log_std scaling (multiplicative std factor): "
                    f"{self.log_std_scale} (offset={offset:+.3f})"
                )
            else:
                logging.warning(
                    "⚠️ log_std_layer not found in SAC policy - scaling skipped"
                )

    def _load_sac_checkpoint(self, checkpoint_path: str) -> None:
        """Load SAC checkpoint with proper error handling."""
        ckpt_path = Path(checkpoint_path).expanduser().resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"SAC checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=self.device)

        # Load state dicts with proper error handling
        required_keys = [
            "encoder_main_state",
            "encoder_target_state",
            "policy_state",
            "q1_state",
            "q2_state",
            "q1_target_state",
            "q2_target_state",
            "log_alpha",
        ]

        for key in required_keys:
            if key not in ckpt:
                raise KeyError(f"Missing key '{key}' in SAC checkpoint")

        # Load network states
        if self.sac_agent is not None:
            self.sac_agent.encoder_main.load_state_dict(ckpt["encoder_main_state"])
            self.sac_agent.encoder_target.load_state_dict(ckpt["encoder_target_state"])
            self.sac_agent.policy.load_state_dict(ckpt["policy_state"])
            self.sac_agent.q1.load_state_dict(ckpt["q1_state"])
            self.sac_agent.q2.load_state_dict(ckpt["q2_state"])
            self.sac_agent.q1_target.load_state_dict(ckpt["q1_target_state"])
            self.sac_agent.q2_target.load_state_dict(ckpt["q2_target_state"])
            self.sac_agent.log_alpha.data.copy_(torch.tensor(ckpt["log_alpha"]))

            # Apply log_std scaling after loading the checkpoint
            self._apply_log_std_scaling()

            # Set networks to eval mode
            self.sac_agent.policy.eval()
            self.sac_agent.q1.eval()
            self.sac_agent.q2.eval()
            self.sac_agent.q1_target.eval()
            self.sac_agent.q2_target.eval()
        else:
            raise RuntimeError(
                "SAC agent not initialized but checkpoint loading attempted"
            )

        logging.info("✅ SAC checkpoint loaded successfully")

    def _setup_sac_normalizers(self, dataset: RLDataset) -> None:
        """Setup normalizers for the SAC model."""
        # Store normalizers for SAC data processing
        self.sac_state_normalizer = dataset.state_normalizer
        # Use expert action normalizer for normalizing expert actions (absolute range)
        self.sac_expert_action_normalizer = dataset.expert_action_normalizer
        # Use delta action normalizer for normalizing/denormalizing delta actions
        self.sac_action_normalizer = dataset.action_normalizer
        self.sac_rgb_normalizer = dataset.rgb_normalizer
        self.sac_mask_normalizer = dataset.mask_normalizer
        self.sac_depth_normalizer = dataset.depth_normalizer

    @staticmethod
    def _extract_model_id(checkpoint_path: str) -> str:
        """Extract the model ID from the checkpoint path."""
        p = Path(checkpoint_path).resolve()
        for parent in p.parents:
            name = parent.name
            if name.startswith("model_") or name.startswith("sac_"):
                return name

        # Fallback to filename without extension
        return Path(checkpoint_path).stem

    def _validate_field_definitions(self) -> None:
        """Validate that field definitions are consistent."""
        # Check state fields
        state_obs_config = self.expert_config.shape_meta.obs["state"]
        assert isinstance(state_obs_config, StateObsConfig)
        state_dim = state_obs_config.dim
        state_end_idx = max(
            field_range[1] for field_range in self.state_fields.values()
        )
        if state_end_idx != state_dim:
            raise ValueError(
                f"State field definitions don't match state dimension. "
                f"Expected {state_dim}, got max index {state_end_idx}"
            )

        # Check action fields
        action_dim = self.expert_config.shape_meta.action.dim
        action_end_idx = max(
            field_range[1] for field_range in self.action_fields.values()
        )
        if action_end_idx != action_dim:
            raise ValueError(
                f"Action field definitions don't match action dimension. "
                f"Expected {action_dim}, got max index {action_end_idx}"
            )

    def _convert_state_dict_to_vector(self, state_dict: Dict[str, Any]) -> np.ndarray:
        """Convert state dictionary to flattened vector using field definitions."""
        # Initialize vector with correct dimension
        state_obs_config = self.expert_config.shape_meta.obs["state"]
        assert isinstance(state_obs_config, StateObsConfig)
        state_dim = state_obs_config.dim
        state_vector = np.zeros(state_dim, dtype=np.float32)

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
                logging.warning(f"Missing field '{field_name}' in state dictionary")

        return state_vector

    def _convert_action_vector_to_dict(
        self, action_vector: np.ndarray
    ) -> Dict[str, Any]:
        """Convert flattened action vector to dictionary using field definitions."""
        action_dict = {}

        for field_name, field_range in self.action_fields.items():
            start_idx, end_idx = field_range
            action_dict[field_name] = action_vector[start_idx:end_idx].tolist()

        return action_dict

    def get_metadata(self) -> Metadata:
        """Return metadata describing the server capabilities."""
        return {
            "server_info": {
                "server_name": "sac_delta_policy_server",
                "service_type": "action policy",
            },
            "service_metadata": {
                "horizon": self.horizon,
                "n_obs_steps": self.n_obs_steps,
                "n_action_steps": self.n_action_steps,
                "model_id": self.model_id,
                "has_sac": self.has_sac,
                "action_offset": self.action_offset,
                "stochastic_rollouts": self.stochastic_rollouts,
                "log_std_scale": self.log_std_scale,
                "state_fields": dict(self.state_fields),
                "action_fields": dict(self.action_fields),
                "state_dim": self.expert_config.shape_meta.obs["state"].dim,  # type: ignore
                "action_dim": self.expert_config.shape_meta.action.dim,
            },
            "request_format": {
                "predict": {
                    "observations": {
                        "state_sequence": "list of state dictionaries with field names",
                        "images": "JPEG-encoded camera images",
                    }
                }
            },
            "response_format": {
                "predict": {
                    "actions": {
                        "expert_action": "dictionary with field names",
                        "residual_action": "dictionary with field names",
                        "final_action": "dictionary with field names",
                    }
                }
            },
        }  # type: ignore

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Process incoming request and return combined action."""
        if request.get("type") != "predict":
            return {"error": "unknown request type"}

        obs_raw: dict[str, Any] = request.get("observations", {})

        try:
            # Process state sequence - expect list of dictionaries
            state_sequence = obs_raw.get("state_sequence", [])
            if not isinstance(state_sequence, list):
                return {"error": "state_sequence must be a list of dictionaries"}

            if len(state_sequence) == 0:
                return {"error": "state_sequence cannot be empty"}

            # Convert state dictionaries to vectors
            state_vectors = []
            for i, state_dict in enumerate(state_sequence):
                if not isinstance(state_dict, dict):
                    return {"error": f"state_sequence[{i}] must be a dictionary"}

                try:
                    state_vector = self._convert_state_dict_to_vector(state_dict)
                    state_vectors.append(state_vector)
                except ValueError as e:
                    return {"error": f"Error converting state_sequence[{i}]: {e!s}"}

            # Stack into array format expected by models
            joint_array = np.stack(state_vectors, axis=0)  # [n_obs_steps, state_dim]

            if joint_array.shape[0] != self.n_obs_steps:
                logging.warning(
                    f"Expected {self.n_obs_steps} obs steps, got {joint_array.shape[0]}"
                )

            # Build observation tensor dict for UnifiedPolicyModel
            obs_tensor: dict[str, torch.Tensor] = {}

            # Process images (JPEG bytes) - unchanged from original
            for camera_key in self.camera_keys:
                if camera_key in obs_raw:
                    img_bytes = obs_raw[camera_key]
                    if not isinstance(img_bytes, (bytes, bytearray)):
                        logging.warning(
                            f"Expected bytes for camera {camera_key}, "
                            f"got {type(img_bytes)}"
                        )
                        continue

                    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                    rgb_img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
                    if rgb_img is None:
                        logging.warning(f"Failed to decode JPEG for {camera_key}")
                        continue

                    img_tensor = (
                        torch.from_numpy(rgb_img).permute(2, 0, 1).float() / 255.0
                    ).unsqueeze(0)  # [1, C, H, W]
                    obs_tensor[camera_key] = img_tensor.to(self.device)
                else:
                    logging.warning(f"Missing camera observation: {camera_key}")

            # For UnifiedPolicyModel: use standard state format
            state_tensor = torch.from_numpy(joint_array).float()
            obs_tensor["state"] = state_tensor.unsqueeze(0).to(self.device)

            # Validate we have required observations
            if "state" not in obs_tensor:
                return {"error": "missing state_sequence in observation"}

            missing_cameras = [key for key in self.camera_keys if key not in obs_tensor]
            if missing_cameras:
                logging.warning(f"Missing camera observations: {missing_cameras}")

            # STEP 1: Get base action from UnifiedPolicyModel
            with torch.no_grad():
                # Prepare observations for UnifiedPolicyModel
                obs_for_expert = {}

                # Convert state tensor back to numpy for UnifiedPolicyModel
                obs_for_expert["state"] = joint_array  # Use original numpy array

                # Convert images back to numpy arrays for UnifiedPolicyModel
                for camera_key in self.camera_keys:
                    if camera_key in obs_raw:
                        img_bytes = obs_raw[camera_key]
                        if isinstance(img_bytes, (bytes, bytearray)):
                            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                            rgb_img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
                            if rgb_img is not None:
                                # Convert BGR to RGB
                                if rgb_img.shape[-1] == 3:
                                    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
                                obs_for_expert[camera_key] = rgb_img
                            else:
                                logging.warning(
                                    f"Failed to decode JPEG for {camera_key}, skipping"
                                )
                        else:
                            logging.warning(
                                f"Expected bytes for {camera_key}, "
                                f"got {type(img_bytes)}"
                            )

                logging.debug(
                    f"Calling UnifiedPolicyModel.generate_actions with obs keys: "
                    f"{list(obs_for_expert.keys())}"
                )

                # Check if we have the minimum required observations
                if "state" not in obs_for_expert:
                    return {"error": "missing state observation for UnifiedPolicyModel"}

                # Handle missing images gracefully by providing dummy images
                for camera_key in self.camera_keys:
                    if camera_key not in obs_for_expert:
                        logging.warning(
                            f"Creating dummy image for missing {camera_key}"
                        )
                        # Create a dummy 224x224 RGB image
                        dummy_img = np.full((224, 224, 3), 128, dtype=np.uint8)
                        obs_for_expert[camera_key] = dummy_img

                unified_actions = self.unified_policy.generate_actions(obs_for_expert)
                logging.debug(f"Unified actions type: {type(unified_actions)}")
                logging.debug(
                    f"Unified actions shape/form: {getattr(unified_actions, 'shape', getattr(unified_actions, 'size', len(unified_actions) if hasattr(unified_actions, '__len__') else 'unknown'))}"
                )

                # Handle action selection from action chunk
                if isinstance(unified_actions, torch.Tensor):
                    logging.debug(f"Tensor shape: {unified_actions.shape}")
                    if (
                        unified_actions.dim() == 3
                    ):  # [batch, n_action_steps, action_dim]
                        # Select action at specified offset
                        action_idx = min(
                            self.action_offset, unified_actions.shape[1] - 1
                        )
                        base_action = unified_actions[0, action_idx, :]
                        logging.debug(f"Using action chunk at index {action_idx}")
                    elif unified_actions.dim() == 2:  # [n_action_steps, action_dim]
                        # Select action at specified offset
                        action_idx = min(
                            self.action_offset, unified_actions.shape[0] - 1
                        )
                        base_action = unified_actions[action_idx, :]
                        logging.debug(f"Using action chunk at index {action_idx}")
                    else:  # [action_dim]
                        base_action = unified_actions
                        logging.debug("Using full tensor for 1D tensor")
                    base_action_np = base_action.cpu().numpy()
                else:
                    # Handle numpy array
                    logging.debug(f"Numpy array shape: {unified_actions.shape}")
                    if unified_actions.ndim == 2:
                        # Select action at specified offset
                        action_idx = min(
                            self.action_offset, unified_actions.shape[0] - 1
                        )
                        base_action_np = unified_actions[action_idx, :]
                        logging.debug(f"Using action chunk at index {action_idx}")
                    else:
                        base_action_np = unified_actions
                        logging.debug("Using full numpy array")

                logging.debug(f"Base action shape: {base_action_np.shape}")
                logging.debug(
                    f"Base action sample: "
                    f"{base_action_np[:5] if len(base_action_np) > 5 else base_action_np}"
                )

            # STEP 2: Get delta action (either from SAC or zeros)
            if self.has_sac and self.sac_agent:
                # Prepare observation for SAC Delta Action
                obs_for_sac = {}

                # State: temporal sequence as expected by the dataset
                # Apply SAC state normalization
                state_for_sac = self.sac_state_normalizer.normalize(state_tensor)
                obs_for_sac["state"] = state_for_sac.unsqueeze(0).to(self.device)

                # Expert action: the base action from UnifiedPolicyModel
                # Apply SAC action normalization
                expert_action_tensor = torch.from_numpy(base_action_np).float()
                expert_action_normalized = self.sac_expert_action_normalizer.normalize(
                    expert_action_tensor
                )
                obs_for_sac["expert_action"] = expert_action_normalized.unsqueeze(0).to(
                    self.device
                )

                # Images: process all available camera images with SAC normalization
                for camera_key in self.camera_keys:
                    if camera_key in obs_tensor:
                        # Apply SAC image normalization
                        img_for_sac = obs_tensor[camera_key]  # [1, C, H, W]
                        img_normalized = self.sac_rgb_normalizer.normalize(img_for_sac)
                        # SAC dataset expects images with an extra batch dimension
                        obs_for_sac[camera_key] = img_normalized.unsqueeze(
                            1
                        )  # [1, 1, C, H, W]

                # Extra observations: extract from extra_obs field and add to obs_for_sac
                extra_obs_raw = obs_raw.get("extra_obs", {})
                if isinstance(extra_obs_raw, dict):
                    for key, value in extra_obs_raw.items():
                        if isinstance(value, list):
                            # Convert to tensor and add batch dimension
                            extra_obs_tensor = torch.tensor(value, dtype=torch.float32)
                            obs_for_sac[key] = extra_obs_tensor.unsqueeze(0).to(
                                self.device
                            )
                        else:
                            logging.warning(
                                f"Unexpected extra_obs format for {key}: {type(value)}"
                            )
                else:
                    logging.warning(
                        f"extra_obs is not a dictionary: {type(extra_obs_raw)}"
                    )

                # Get SAC delta prediction
                with torch.no_grad():
                    delta_action, _ = self.sac_agent.policy.get_action(
                        obs_for_sac, deterministic=not self.stochastic_rollouts
                    )
                    # Denormalize the delta action
                    delta_action_denorm = self.sac_action_normalizer.denormalize(
                        delta_action
                    )
                    delta_action_np = delta_action_denorm.squeeze(0).cpu().numpy()

                final_delta_np = delta_action_np

                if self.stochastic_rollouts:
                    logging.debug(f"SAC delta (stochastic): {delta_action_np}")
                else:
                    logging.debug(f"SAC delta (deterministic): {delta_action_np}")

            else:
                # No SAC model - use zero delta
                final_delta_np = np.zeros_like(base_action_np)
                logging.debug("Using zero delta (no SAC model)")

            # STEP 3: Combine base action + delta
            final_action_np = base_action_np + final_delta_np

            # Log for debugging
            logging.debug(f"Base action: {base_action_np}")
            logging.debug(f"Final action: {final_action_np}")

            # Convert actions to dictionary format
            expert_action_dict = self._convert_action_vector_to_dict(base_action_np)
            residual_action_dict = self._convert_action_vector_to_dict(final_delta_np)
            final_action_dict = self._convert_action_vector_to_dict(final_action_np)

            return {
                "type": "actions",
                "actions": {
                    "expert_action": expert_action_dict,
                    "residual_action": residual_action_dict,
                    "final_action": final_action_dict,
                },
            }

        except Exception as e:
            logging.error(f"Error processing observation: {e}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            return {"error": f"observation processing failed: {e!s}"}


# -----------------------------------------------------------------------------
# CLI entry-point
# -----------------------------------------------------------------------------


async def _serve(
    expert_config_path: str,
    sac_config_path: Optional[str],
    sac_checkpoint_path: Optional[str],
    host: str,
    port: int,
    action_offset: int = 0,
    stochastic_rollouts: bool = False,
    log_std_scale: float = 1.0,
):
    """Start the SAC Delta Action policy server."""
    handler = SACDeltaPolicyServer(
        expert_config_path,
        sac_config_path,
        sac_checkpoint_path,
        action_offset=action_offset,
        stochastic_rollouts=stochastic_rollouts,
        log_std_scale=log_std_scale,
    )
    async with DirectServer(host, port, handler):
        mode = "SAC + Base Policy" if handler.has_sac else "Base Policy Only"
        sampling_mode = "Stochastic" if stochastic_rollouts else "Deterministic"
        logging.info(
            f"🚀 {mode} server ready at ws://{host}:{port} ({sampling_mode} sampling)"
        )
        # Keep the coroutine alive forever.
        await asyncio.Future()


def main() -> None:
    """Main entry point for the SAC Delta Action server."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(
        description="Run SAC Delta Action WebSocket server"
    )
    parser.add_argument(
        "--expert-config",
        required=True,
        help="Path to expert model configuration YAML (required)",
    )
    parser.add_argument(
        "--sac-config",
        help="Path to SAC training configuration YAML (optional)",
    )
    parser.add_argument(
        "--sac-checkpoint",
        help="Path to trained SAC Delta Action checkpoint .pt file (optional)",
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Server host (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=10012, help="Server port (default: 10012)"
    )
    parser.add_argument(
        "--action-offset",
        type=int,
        default=0,
        help="Offset for selecting action from action chunk (default: 0)",
    )
    parser.add_argument(
        "--stochastic-rollouts",
        action="store_true",
        help="Use stochastic sampling for data collection (default: deterministic)",
    )
    parser.add_argument(
        "--log-std-scale",
        type=float,
        default=1.0,
        help="Scale factor for log_std to control exploration level (default: 1.0, "
        "use 0.5 for less exploration, 1.5 for more exploration)",
    )
    args = parser.parse_args()

    # Validate SAC arguments
    if args.sac_config and not args.sac_checkpoint:
        parser.error("--sac-checkpoint is required when --sac-config is provided")
    if args.sac_checkpoint and not args.sac_config:
        parser.error("--sac-config is required when --sac-checkpoint is provided")

    # Validate log_std_scale
    if args.log_std_scale <= 0:
        parser.error("--log-std-scale must be positive")

    asyncio.run(
        _serve(
            args.expert_config,
            args.sac_config,
            args.sac_checkpoint,
            args.host,
            args.port,
            args.action_offset,
            args.stochastic_rollouts,
            args.log_std_scale,
        )
    )


if __name__ == "__main__":
    main()
