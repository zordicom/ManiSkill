"""
Copyright 2025 Zordi, Inc. All rights reserved.

Factory for building vision encoders that output a flat feature vector of a specified
dimension.
"""

from typing import Any, Dict, Optional, Union

import torch
from torch import nn
from transformers.modeling_utils import PreTrainedModel

from zordi_vla.configs.configs import (
    ACTDPModelConfig,
    ACTFlowModelConfig,
    ACTModelConfig,
    DPModelConfig,
    ImageObsConfig,
    ImageType,
    ResnetEncoderConfig,
)
from zordi_vla.models.backbones.backbone_utils import load_image_encoder
from zordi_vla.utils.logging_utils import setup_logger

logger = setup_logger(__name__)


class VisionEncoderWithProjection(nn.Module):
    """Wraps a CNN-style backbone that outputs spatial features [B,C,H,W]
    and adds pooling + projection to a fixed output dimension.
    Now supports configurable grid_size for pooling.
    """

    def __init__(
        self,
        backbone: nn.Module,
        backbone_output_channels: int,
        projection_dim: int,
        grid_size: int = 1,  # defaults to 1 (global avg pooling)
    ):
        super().__init__()
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))
        project_in_dim = backbone_output_channels * grid_size * grid_size
        if project_in_dim == projection_dim:
            self.projection = nn.Identity()
        else:
            self.projection = nn.Linear(project_in_dim, projection_dim)
        logger.info(
            "VisionEncoderWithProjection: Backbone output %s, Grid size %dx%d, "
            "Projection from %s to %s",
            backbone_output_channels,
            grid_size,
            grid_size,
            project_in_dim,
            projection_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the VisionEncoderWithProjection.

        Args:
            x: Input tensor of shape [B, C, H, W]

        Returns:
            Output tensor of shape [B, projection_dim]
        """
        features = self.backbone(x)
        pooled_features = self.pool(features)
        flattened_features = torch.flatten(pooled_features, 1)
        projected_features = self.projection(flattened_features)
        return projected_features


class MaskEncoder(nn.Module):
    """Mask encoder that preserves coarse spatial layout via grid pooling."""

    def __init__(self, input_channels: int, projection_dim: int, grid_size: int = 1):
        super().__init__()
        # same CNN backbone as SimpleMaskCNN
        self.output_channels = 128
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.Mish(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(16, 64),
            nn.Mish(),
            nn.Conv2d(64, self.output_channels, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, self.output_channels),
            nn.Mish(),
        )
        # grid pooling to preserve spatial layout
        self.pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))
        # projection from flattened grid features to desired dimension
        project_in_dim = self.output_channels * grid_size * grid_size
        if project_in_dim == projection_dim:
            self.projection = nn.Identity()
        else:
            self.projection = nn.Linear(project_in_dim, projection_dim)

        logger.info(
            "SpatialMaskCNN initialized with input_channels=%s, "
            "CNN_output_channels=%s, grid_size=%dx%d, projection_dim=%s",
            input_channels,
            self.output_channels,
            grid_size,
            grid_size,
            projection_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of SpatialMaskCNN.

        Args:
            x: Input tensor of shape [B, C, H, W]

        Returns:
            Output tensor of shape [B, projection_dim] with spatial context.
        """
        features = self.cnn(x)
        pooled = self.pool(features)
        flattened = torch.flatten(pooled, 1)
        projected = self.projection(flattened)
        return projected


class DepthEncoder(nn.Module):
    """Encoder for depth modalities using ResNet18 and projection."""

    def __init__(
        self,
        input_channels: int,
        projection_dim: int,
        freeze_resnet18: bool = True,
        grid_size: int = 1,
    ):
        super().__init__()
        # Configure ResNet18 for depth
        depth_resnet_cfg = ResnetEncoderConfig(
            series="resnet18",
            freeze=freeze_resnet18,
            use_group_norm=True,  # GroupNorm is generally good
        )

        # load_image_encoder for ResNet returns
        # spatial features [B, C_feat, H_feat, W_feat]
        depth_backbone = load_image_encoder(
            name="resnet18",  # Explicitly resnet18
            in_channels=input_channels,
            resnet_cfg=depth_resnet_cfg,
        )

        # ResNet18's feature dimension before final FC (from its layer4) is 512
        self.encoder = VisionEncoderWithProjection(
            backbone=depth_backbone,
            backbone_output_channels=512,
            projection_dim=projection_dim,
            grid_size=grid_size,
        )
        logger.info(
            "DepthEncoder initialized with ResNet18 (frozen=%s, grid_size=%dx%d): "
            "projects to %s",
            freeze_resnet18,
            grid_size,
            grid_size,
            projection_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the DepthEncoder.

        Args:
            x: Input tensor of shape [B, C, H, W]

        Returns:
            Output tensor of shape [B, projection_dim]
        """
        return self.encoder(x)


class RGBEncoder(nn.Module):
    """Encoder for RGB modalities using configured ResNet/DINOv2."""

    def __init__(
        self,
        encoder_main_cfg: Union[
            DPModelConfig, ACTModelConfig, ACTDPModelConfig, ACTFlowModelConfig
        ],
        input_channels: int,
        projection_dim: int,
        grid_size: int = 1,
    ):
        super().__init__()
        encoder_type = encoder_main_cfg.encoder_type.lower()

        if encoder_type == "resnet":
            resnet_cfg = encoder_main_cfg.resnet
            current_series_name = resnet_cfg.series
            rgb_backbone = load_image_encoder(
                name="resnet",  # Generic resnet, series from resnet_cfg
                in_channels=input_channels,
                resnet_cfg=resnet_cfg,
            )
            backbone_out_c = _get_backbone_output_channels_after_loader(
                encoder_type, rgb_backbone, current_series_name
            )
            self.encoder = VisionEncoderWithProjection(
                backbone=rgb_backbone,
                backbone_output_channels=backbone_out_c,
                projection_dim=projection_dim,
                grid_size=grid_size,
            )
            logger.info(
                "RGBEncoder initialized with ResNet series '%s' (grid_size=%dx%d): "
                "projects to %s",
                current_series_name,
                grid_size,
                grid_size,
                projection_dim,
            )
        elif encoder_type == "dinov2":
            dino_cfg = encoder_main_cfg.dino
            current_series_name = dino_cfg.series
            # load_image_encoder for DINOv2 (from utils.py) returns a DINOEncoder
            # which itself handles CLS token extraction or spatial feature reshaping
            # and has its own projection if it's a ViT-style model focused on CLS.
            # The current DINOEncoder in utils.py is designed to output spatial features
            # [B,C,H,W] from patch tokens, which is suitable for
            # VisionEncoderWithProjection.
            dino_backbone_spatial = load_image_encoder(
                name="dinov2", in_channels=input_channels, dinov2_cfg=dino_cfg
            )
            backbone_out_c = _get_backbone_output_channels_after_loader(
                encoder_type,
                dino_backbone_spatial,
                current_series_name,  # Pass the spatial backbone to estimate channels
            )
            self.encoder = VisionEncoderWithProjection(
                backbone=dino_backbone_spatial,
                backbone_output_channels=backbone_out_c,
                projection_dim=projection_dim,
                grid_size=grid_size,
            )
            logger.info(
                "RGBEncoder initialized with DINOv2 series '%s' "
                "(spatial features, grid_size=%dx%d): projects to %s",
                current_series_name,
                grid_size,
                grid_size,
                projection_dim,
            )
        elif encoder_type == "efficientnet":
            eff_cfg = encoder_main_cfg.efficientnet
            current_series_name = eff_cfg.series
            eff_backbone = load_image_encoder(
                name=current_series_name,
                in_channels=input_channels,
                efficientnet_cfg=eff_cfg,
            )
            backbone_out_c = _get_backbone_output_channels_after_loader(
                encoder_type,
                eff_backbone,
                current_series_name,
            )
            self.encoder = VisionEncoderWithProjection(
                backbone=eff_backbone,
                backbone_output_channels=backbone_out_c,
                projection_dim=projection_dim,
                grid_size=grid_size,
            )
            logger.info(
                "RGBEncoder initialized with EfficientNet series '%s' "
                "(grid_size=%dx%d): projects to %s",
                current_series_name,
                grid_size,
                grid_size,
                projection_dim,
            )
        else:
            raise ValueError(f"Unsupported RGB encoder_type: '{encoder_type}'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the RGBEncoder.

        Args:
            x: Input tensor of shape [B, C, H, W]

        Returns:
            Output tensor of shape [B, projection_dim]
        """
        return self.encoder(x)


def _get_backbone_output_channels_after_loader(
    encoder_type: str,
    loaded_backbone_module: nn.Module,
    series_name: str,
) -> int:
    """Estimates output channels from the backbone module returned by
    load_image_encoder.

    - load_image_encoder for ResNet returns avgpooled features from layer4.
    - load_image_encoder for DINOv2 returns spatial patch features
        (DINOEncoder wrapper in utils.py).
    """
    encoder_type = encoder_type.lower()

    if encoder_type == "resnet":
        if series_name in {"resnet18", "resnet34"}:
            return 512
        elif series_name in {"resnet50", "resnet101", "resnet152"}:
            return 2048
        else:
            raise ValueError(
                f"Cannot determine output channels for ResNet series: {series_name}."
            )

    elif encoder_type == "dinov2":
        if (
            hasattr(loaded_backbone_module, "model")
            and isinstance(loaded_backbone_module.model, PreTrainedModel)
            and hasattr(loaded_backbone_module.model, "config")
            and hasattr(loaded_backbone_module.model.config, "hidden_size")
        ):
            potential_hs = loaded_backbone_module.model.config.hidden_size
            if isinstance(potential_hs, int):
                return potential_hs
            else:
                logger.warning(
                    "DINOv2 model.config.hidden_size not int for '%s'. Fallback.",
                    series_name,
                )
        logger.warning(
            "Estimating DINOv2 output channels for '%s' via series name (fallback).",
            series_name,
        )
        if "base" in series_name:
            return 768
        elif "small" in series_name:
            return 384
        elif "large" in series_name:
            return 1024
        elif "giant" in series_name:
            return 1536
        else:
            raise ValueError(
                f"Cannot estimate DINOv2 output channels from series name: "
                f"{series_name}."
            )

    elif encoder_type == "efficientnet":
        if series_name == "efficientnet_b0":
            return 1280

        raise ValueError(
            f"Cannot determine output channels for EfficientNet series: {series_name}."
        )

    else:
        raise ValueError(
            f"Unsupported encoder_type for channel estimation: {encoder_type}"
        )


def build_vision_encoders(
    obs_config: Dict[str, Any],
    encoder_cfg: Union[
        DPModelConfig, ACTModelConfig, ACTDPModelConfig, ACTFlowModelConfig
    ],
    output_dim: int,
) -> Optional[nn.ModuleDict]:
    """
    Builds a ModuleDict of vision encoders (RGBEncoder, DepthEncoder, SpatialMaskCNN)
    based on modality types in obs_config and overall encoder_cfg.
    Handles sharing of encoders per modality type if share_image_encoder is True.
    """
    image_encoders = nn.ModuleDict()

    # Identify all image modalities in obs_config
    image_keys_with_type = {
        k: mod_config.image_type.value
        for k, mod_config in obs_config.items()
        if isinstance(mod_config, ImageObsConfig)
        and mod_config.image_type in {ImageType.RGB, ImageType.DEPTH, ImageType.MASK}
    }

    if not image_keys_with_type:
        logger.info(
            "No image modalities (rgb, depth, mask) found in obs_config. "
            "No vision encoders built."
        )
        return None

    share_encoders = getattr(encoder_cfg, "share_image_encoder", False)
    # Get grid_size from the main model config
    encoder_grid_size = getattr(encoder_cfg, "image_encoder_grid_size", 1)

    logger.info(
        "Building vision encoders. Target output_dim: %s. Share encoders per type: %s. "
        "Grid size: %d.",
        output_dim,
        share_encoders,
        encoder_grid_size,
    )

    # Store shared encoders if share_encoders is True
    shared_encoder_instances: Dict[str, nn.Module] = {}

    for key, modality_type in image_keys_with_type.items():
        # Retrieve ImageObsConfig directly
        mod_config = obs_config[key]  # type: ImageObsConfig
        input_channels = mod_config.channels
        if input_channels is None:
            logger.warning(
                "Modality '%s' of type '%s' is missing 'channels'. Skipping.",
                key,
                modality_type,
            )
            continue

        current_encoder: Optional[nn.Module] = None

        if share_encoders and modality_type in shared_encoder_instances:
            assert isinstance(modality_type, str), (
                "modality_type is not a string: %s",
                type(modality_type),
            )
            current_encoder = shared_encoder_instances[modality_type]
            if key is not None:
                logger.info("Using shared '%s' encoder for '%s'.", modality_type, key)
        else:
            if key is not None:
                logger.info(
                    "Building new encoder for '%s' (type: '%s').", key, modality_type
                )
            if modality_type == "rgb":
                # RGBEncoder uses the main encoder_cfg for its type (resnet/dino)
                # and specific series
                current_encoder = RGBEncoder(
                    encoder_main_cfg=encoder_cfg,
                    input_channels=input_channels,
                    projection_dim=output_dim,
                    grid_size=encoder_grid_size,
                )
            elif modality_type == "depth":
                # DepthEncoder always uses ResNet18. Freeze option can be added to
                # ModelConfig if needed.
                # Default for freeze_resnet18 changed to True
                freeze_depth_enc = getattr(encoder_cfg, "freeze_depth_encoder", True)
                current_encoder = DepthEncoder(
                    input_channels=input_channels,
                    projection_dim=output_dim,
                    freeze_resnet18=freeze_depth_enc,
                    grid_size=encoder_grid_size,
                )
            elif modality_type == "mask":
                # Directly use SpatialMaskCNN for masks
                current_encoder = MaskEncoder(
                    input_channels=input_channels,
                    projection_dim=output_dim,
                    grid_size=encoder_grid_size,
                )
            else:
                logger.warning(
                    "Modality '%s' has unknown type '%s' for encoder creation. "
                    "Skipping.",
                    key,
                    modality_type,
                )
                continue

            if share_encoders and current_encoder is not None:
                shared_encoder_instances[modality_type] = current_encoder

        if current_encoder is not None:
            image_encoders[key] = current_encoder
        else:
            logger.error(
                "Failed to create or assign encoder for key '%s' of type '%s'.",
                key,
                modality_type,
            )

    if not image_encoders:
        logger.warning("No vision encoders were successfully created. Returning None.")
        return None

    return image_encoders
