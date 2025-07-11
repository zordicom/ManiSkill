"""
Copyright 2025 Zordi, Inc. All rights reserved.
"""

from enum import Enum, unique
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import attr

from zordi_vla.utils.logging_utils import setup_logger


@unique
class PolicyType(str, Enum):
    """Type of policy."""

    DP = "diffusion"
    ACT = "act"
    ACTDP = "act+dp"
    ACTFLOW = "act+flow"


@unique
class StateType(str, Enum):
    """Type of state."""

    VECTOR = "vector"


@unique
class ActionType(str, Enum):
    """Type of action."""

    VECTOR = "vector"


@unique
class ImageType(str, Enum):
    """Type of image."""

    RGB = "rgb"
    DEPTH = "depth"
    MASK = "mask"


@unique
class EncoderType(str, Enum):
    """Type of encoder."""

    RESNET = "resnet"
    DINOV2 = "dinov2"
    EFFICIENTNET = "efficientnet"


@unique
class NormalizerType(str, Enum):
    """Type of normalizer."""

    MINMAX = "minmax"
    MEANSTD = "meanstd"


@unique
class NoiseScheduleType(str, Enum):
    """Type of noise schedule."""

    DDPM = "DDPM"
    DDIM = "DDIM"


@attr.s(auto_attribs=True, frozen=True)
class VectorFieldConfig:
    """Configuration for named fields within a vector."""

    fields: Dict[str, Tuple[int, int]]


@attr.s(auto_attribs=True, frozen=True)
class ActionConfig:
    """Configuration for action shape."""

    dim: int
    action_type: ActionType = ActionType.VECTOR
    vector_fields: Optional[VectorFieldConfig] = None


@attr.s(auto_attribs=True, frozen=True)
class StateObsConfig:
    """Configuration for a state observation modality."""

    dim: int
    state_type: StateType = StateType.VECTOR
    vector_fields: Optional[VectorFieldConfig] = None


@attr.s(auto_attribs=True, frozen=True)
class ImageObsConfig:
    """Configuration for an image observation modality."""

    channels: int
    image_size: Tuple[int, int]  # H, W
    image_type: ImageType


@attr.s(auto_attribs=True, frozen=True)
class ShapeConfig:
    """Defines the shapes and types of actions and observations."""

    action: ActionConfig
    obs: Dict[str, Union[StateObsConfig, ImageObsConfig]]


@attr.s(auto_attribs=True, frozen=True)
class DatasetConfig:
    """Dataset configuration (now minimal, shape info is in ShapeConfig)."""

    path: str
    seed: int = 42
    val_ratio: float = 0.05
    normalizer: NormalizerType = NormalizerType.MINMAX
    state_stats_path: str = "aggregated_stats.json"

    mask_num_classes: int = 1
    depth_clipping_range: Tuple[float, float] = (0.03, 2.0)
    depth_unit_scale: float = 1000.0

    # Global noise parameters (maintained for backward compatibility)
    noise_std_state: float = 0.005
    noise_std_action: float = 0.0

    # Selective noise augmentation parameters
    # Dict mapping vector field names to noise parameters
    # Each field can have either a single float (for Gaussian noise) or
    # a Dict with "type" and "range"
    selective_noise_state: Optional[Dict[str, Any]] = None
    selective_noise_action: Optional[Dict[str, Any]] = None

    random_resized_crop_scale: Tuple[float, float] = (0.98, 1.0)
    random_resized_crop_ratio: Tuple[float, float] = (0.98, 1.02)


@attr.s(auto_attribs=True, frozen=True)
class ResnetEncoderConfig:
    """ResNet image encoder configuration."""

    series: str = "resnet18"
    freeze: bool = False
    use_group_norm: bool = True


@attr.s(auto_attribs=True, frozen=True)
class DinoEncoderConfig:
    """DINOv2 image encoder configuration."""

    series: str = "facebook/dinov2-base"  # Default HuggingFace model ID
    freeze: bool = True
    use_lora: bool = False
    lora_r: int = 4
    lora_alpha: float = 1.0
    num_registers: int = 0

    def __attrs_post_init__(self):
        if self.num_registers == 0 and "with-registers" in self.series.lower():
            object.__setattr__(self, "num_registers", 4)
        elif self.num_registers > 0 and "with-registers" not in self.series.lower():
            logger = setup_logger(__name__ + ".DinoEncoderConfig")
            logger.warning(
                "DinoEncoderConfig for series '%s' has num_registers=%d but "
                "'with-registers' is not in the series name.",
                self.series,
                self.num_registers,
            )


@attr.s(auto_attribs=True, frozen=True)
class EfficientNetEncoderConfig:
    """EfficientNet image encoder configuration."""

    series: str = "efficientnet_b0"
    freeze: bool = True
    use_group_norm: bool = False
    # Pretrained weights identifier, e.g., 'DEFAULT' for ImageNet-1k
    weights: str = "DEFAULT"


@attr.s(auto_attribs=True, frozen=True)
class BaseModelConfig:
    """Base configuration for image encoders common to all policy models."""

    encoder_type: EncoderType = EncoderType.DINOV2
    resnet: ResnetEncoderConfig = attr.Factory(ResnetEncoderConfig)
    dino: DinoEncoderConfig = attr.Factory(DinoEncoderConfig)
    efficientnet: EfficientNetEncoderConfig = attr.Factory(EfficientNetEncoderConfig)
    share_image_encoder: bool = False
    image_encoder_grid_size: int = 5


@attr.s(auto_attribs=True, frozen=True)
class DPModelConfig(BaseModelConfig):
    """Model configuration for diffusion policy with advanced features."""

    model_dim: int = 256
    state_embed_dim: int = 128
    unet_depth: int = 3
    unet_kernel_size: int = 5
    time_embed_dim: int = 256
    film_hidden_dim: int = 1024

    diffusion_steps: int = 100
    num_inference_steps: int = 100
    noise_scheduler_type: NoiseScheduleType = NoiseScheduleType.DDIM
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "squaredcos_cap_v2"  # "linear", "squaredcos_cap_v2"
    prediction_type: str = "epsilon"  # "epsilon" or "sample"
    clip_sample: bool = False
    clip_sample_range: float = 1.0
    ddim_eta: float = 0.0

    # Advanced diffusion features (generalized from ACTDP)
    cfg_prob: float = 0.05  # Probability of dropping context for CFG during training (0.05 = light CFG)
    guidance_scale: float = (
        1.2  # Guidance scale for CFG during inference (1.2 = light guidance)
    )
    dynamic_thresholding_percentile: float = (
        0.99  # Percentile for dynamic thresholding (0.99 = light thresholding)
    )
    use_self_conditioning: bool = True  # Self-conditioning for diffusion
    dropout: float = 0.1


@attr.s(auto_attribs=True, frozen=True)
class ACTModelConfig(BaseModelConfig):
    """Model configuration for Extended ACT policy with advanced features."""

    model_dim: int = 256
    nhead: int = 4
    dim_feedforward: int = 512
    n_encoder_layers: int = 4
    n_decoder_layers: int = 4
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 2

    # Advanced features for future extensibility
    dropout: float = 0.1


@attr.s(auto_attribs=True, frozen=True)
class ACTDPModelConfig(BaseModelConfig):
    """Model configuration for Diffusion-Enhanced ACT policy."""

    model_dim: int = 512
    dim_feedforward_enc: int = 2048
    dim_feedforward_dec: int = 2048
    nheads_enc: int = 8
    nheads_dec: int = 8
    num_layers_enc: int = 6
    num_layers_dec: int = 6

    diffusion_steps: int = 100
    num_inference_steps: int = 100
    noise_scheduler_type: NoiseScheduleType = NoiseScheduleType.DDIM
    beta_start: float = 1.0e-4
    beta_end: float = 0.02
    beta_schedule: str = "squaredcos_cap_v2"  # "linear", "squaredcos_cap_v2"
    prediction_type: str = "epsilon"  # "epsilon" or "sample"
    clip_sample: bool = False
    clip_sample_range: float = 1.0
    ddim_eta: float = 0.0

    use_latent_variable: bool = True
    latent_dim: int = 32

    use_self_conditioning: bool = True
    use_hybrid_time_embedding: bool = True
    cfg_prob: float = (
        0.1  # Probability of dropping context for CFG during training (0.0 = disabled)
    )
    guidance_scale: float = (
        1.5  # Guidance scale for CFG during inference (1.0 = disabled)
    )
    dynamic_thresholding_percentile: float = (
        0.995  # Percentile for dynamic thresholding (1.0 = disabled)
    )
    dropout: float = 0.1


@attr.s(auto_attribs=True, frozen=True)
class ACTFlowModelConfig(BaseModelConfig):
    """Model configuration for ACT + Flow Matching policy."""

    model_dim: int = 512
    dim_feedforward_enc: int = 2048
    dim_feedforward_dec: int = 2048
    nheads_enc: int = 8
    nheads_dec: int = 8
    num_layers_enc: int = 6
    num_layers_dec: int = 6

    # Flow matching specific parameters
    num_flow_steps: int = 10  # Number of Euler integration steps for sampling
    flow_time_distribution: str = (
        "uniform"  # "uniform", "beta", or "mixed" for time sampling
    )
    beta_alpha: float = 1.0  # Alpha parameter for Beta distribution (if used)
    beta_beta: float = 1.0  # Beta parameter for Beta distribution (if used)

    # Boundary condition handling for flow matching
    endpoint_fraction: float = 0.2  # Fraction of batch to use for t=0 and t=1 training
    beta_upweight: Optional[Tuple[float, float]] = (
        None  # Beta params for upweighting endpoints
    )

    # CFG and conditioning
    cfg_prob: float = 0.1  # Probability of dropping context for CFG during training
    guidance_scale: float = 1.5  # Guidance scale for CFG during inference

    # Model architecture
    use_hybrid_time_embedding: bool = True  # Use sinusoidal time embedding
    dropout: float = 0.1
    use_latent_variable: bool = True
    latent_dim: int = 32


@attr.s(auto_attribs=True, frozen=True)
class TrainingConfig:
    """Training configuration for all policy types with advanced features."""

    batch_size: int = 128
    num_epochs: int = 1000
    num_workers: int = 8
    prefetch_factor: int = 6

    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-6
    kl_weight: float = 1.0
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 500

    device: str = "cuda"
    pin_memory: bool = False
    use_compile: bool = True
    use_amp: bool = True  # Mixed precision training (AMP)

    # Advanced training features (generalized from ACTDP)
    use_gradient_clipping: bool = True  # Enable gradient clipping for all models
    gradient_clip_norm: float = 1.0  # Max gradient norm for clipping

    # Temporal consistency (implemented but currently not used in training loop)
    temporal_consistency_weight: float = 0.0  # Weight for temporal smoothness loss

    use_wandb: bool = False
    monitor_metric: str = "train_total_loss"
    save_top_k: int = 3
    report_freq: int = 100
    attention_vis_freq: int = 20
    attention_vis_n_samples: int = 0
    preds_vis_freq: int = 10
    preds_vis_n_samples: int = 5

    output_dir: str = "outputs"

    ema_update_after_step: int = 100
    ema_inv_gamma: float = 1.0
    ema_power: float = 0.6667
    ema_min_value: float = 0.0
    ema_max_value: float = 0.9999


@attr.s(auto_attribs=True, frozen=True)
class InferenceConfig:
    """Inference configuration for diffusion policy."""

    model_path: str = "latest.ckpt"
    device: str = "cuda"

    # This is initialized in BasePolicyConfig.__attrs_post_init__.
    resolved_model_path: Path = attr.ib(init=False, default=None)


@attr.s(auto_attribs=True, frozen=True)
class BasePolicyConfig:
    """Base configuration common to all top-level policy configurations."""

    policy_type: PolicyType

    project_name: str
    exp_name: str
    date_str: str

    timedelta_sec: float

    horizon: int
    n_obs_steps: int
    n_action_steps: int

    shape_meta: ShapeConfig

    dataset: DatasetConfig

    training: TrainingConfig
    inference: InferenceConfig

    # This is initialized in __attrs_post_init__.
    checkpoint_dir: Path = attr.ib(init=False, default=None)

    # `model` should be explicitly set by each policy config subclass

    def __attrs_post_init__(self):
        """Resolves checkpoint path and directory after initialization."""
        orig_model_path_str = self.inference.model_path
        model_path_obj = Path(orig_model_path_str)

        resolved_sm_path_val: Path
        checkpoint_dir_val: Path

        current_file_path = Path(__file__).resolve()
        project_root = current_file_path.parent.parent.parent.parent

        if (
            not model_path_obj.is_absolute()
            and not model_path_obj.exists()
            and len(model_path_obj.parts) == 1
        ):
            exp_specific_path = f"{self.date_str}-{self.exp_name}"
            base_checkpoint_dir = (
                project_root
                / self.training.output_dir
                / self.project_name
                / exp_specific_path
                / "checkpoints"
            )
            resolved_sm_path_val = base_checkpoint_dir / model_path_obj.name
            checkpoint_dir_val = base_checkpoint_dir
        else:
            resolved_sm_path_val = model_path_obj.resolve()
            checkpoint_dir_val = resolved_sm_path_val.parent

        object.__setattr__(self.inference, "model_path", str(resolved_sm_path_val))
        object.__setattr__(self.inference, "resolved_model_path", resolved_sm_path_val)
        object.__setattr__(self, "checkpoint_dir", checkpoint_dir_val)


@attr.s(auto_attribs=True, frozen=True)
class DPPolicyConfig(BasePolicyConfig):
    """Top-level configuration for diffusion policy."""

    model: DPModelConfig

    # New modular configuration fields (optional for backward compatibility)
    components: Optional[Dict[str, Dict[str, Any]]] = None
    model_template: Optional[str] = None
    model_templates: Optional[Dict[str, Any]] = None
    model_overrides: Optional[Dict[str, Any]] = None
    compatibility_rules: Optional[Dict[str, Any]] = None


@attr.s(auto_attribs=True, frozen=True)
class ACTPolicyConfig(BasePolicyConfig):
    """Top-level configuration for Extended ACT policy."""

    model: ACTModelConfig

    # New modular configuration fields (optional for backward compatibility)
    components: Optional[Dict[str, Dict[str, Any]]] = None
    model_template: Optional[str] = None
    model_templates: Optional[Dict[str, Any]] = None
    model_overrides: Optional[Dict[str, Any]] = None
    compatibility_rules: Optional[Dict[str, Any]] = None


@attr.s(auto_attribs=True, frozen=True)
class ACTDPPolicyConfig(BasePolicyConfig):
    """Top-level configuration for Diffusion-Enhanced ACT policy."""

    model: ACTDPModelConfig

    # New modular configuration fields (optional for backward compatibility)
    components: Optional[Dict[str, Dict[str, Any]]] = None
    model_template: Optional[str] = None
    model_templates: Optional[Dict[str, Any]] = None
    model_overrides: Optional[Dict[str, Any]] = None
    compatibility_rules: Optional[Dict[str, Any]] = None


@attr.s(auto_attribs=True, frozen=True)
class ACTFlowPolicyConfig(BasePolicyConfig):
    """Top-level configuration for ACT + Flow Matching policy."""

    model: ACTFlowModelConfig

    # New modular configuration fields (optional for backward compatibility)
    components: Optional[Dict[str, Dict[str, Any]]] = None
    model_template: Optional[str] = None
    model_templates: Optional[Dict[str, Any]] = None
    model_overrides: Optional[Dict[str, Any]] = None
    compatibility_rules: Optional[Dict[str, Any]] = None
