"""
Copyright 2025 Zordi, Inc. All rights reserved.
"""

import math
from typing import Optional

import torch
from torch import nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from transformers.models.dinov2.modeling_dinov2 import Dinov2Model
from transformers.models.dinov2_with_registers.modeling_dinov2_with_registers import (
    Dinov2WithRegistersModel,
)

from zordi_vla.configs.configs import (
    DinoEncoderConfig,
    EfficientNetEncoderConfig,
    ResnetEncoderConfig,
)
from zordi_vla.utils.logging_utils import setup_logger

logger = setup_logger(__name__)


class LoRALinear(nn.Module):
    """LoRA adapter for nn.Linear modules."""

    def __init__(self, orig_linear: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.orig = orig_linear
        self.r = r
        self.alpha = alpha
        self.scaling = float(alpha) / float(r)
        in_features = orig_linear.in_features
        out_features = orig_linear.out_features
        self.lora_A = nn.Parameter(torch.zeros((r, in_features)))
        self.lora_B = nn.Parameter(torch.zeros((out_features, r)))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        orig_linear.weight.requires_grad = False
        if orig_linear.bias is not None:
            orig_linear.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA adapter."""
        return self.orig(x) + (x @ self.lora_A.t() @ self.lora_B.t()) * self.scaling


def inject_lora(module: nn.Module, r: int, alpha: float) -> None:
    """Recursively replace nn.Linear with LoRALinear in the module."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            lora_mod = LoRALinear(child, r, alpha)
            setattr(module, name, lora_mod)
        else:
            inject_lora(child, r, alpha)


def _convert_bn_to_groupnorm(module: nn.Module, groups_divisor: int = 16) -> None:
    """
    Recursively convert all nn.BatchNorm2d layers in module to GroupNorm.
    groups_divisor: Preferred divisor to calculate num_groups.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            if num_channels % 32 == 0:
                num_groups = 32
            elif num_channels % 16 == 0:
                num_groups = 16
            elif num_channels % 8 == 0:
                num_groups = 8
            elif num_channels % 4 == 0:
                num_groups = 4
            elif num_channels % 2 == 0:
                num_groups = 2
            else:
                current_num_groups = max(1, num_channels // groups_divisor)
                while num_channels % current_num_groups != 0 and current_num_groups > 1:
                    current_num_groups -= 1
                if num_channels % current_num_groups != 0:
                    num_groups = 1
                else:
                    num_groups = current_num_groups

            logger.info(
                "Replacing BatchNorm2d '%s' (%d channels) with GroupNorm (%d groups).",
                name,
                num_channels,
                num_groups,
            )
            gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)

            if child.affine:
                gn.weight.data = child.weight.data.clone().detach()
                gn.bias.data = child.bias.data.clone().detach()

            setattr(module, name, gn)
        else:
            _convert_bn_to_groupnorm(child, groups_divisor)


def load_image_encoder(
    name: str,
    in_channels: int,
    resnet_cfg: Optional[ResnetEncoderConfig] = None,
    dinov2_cfg: Optional[DinoEncoderConfig] = None,
    efficientnet_cfg: Optional[EfficientNetEncoderConfig] = None,
) -> nn.Module:
    """
    Load an image encoder backbone by name and adjust its first conv for in_channels.
    Supports ResNet variants and DINOv2 variants from HuggingFace Transformers.
    """
    name_lower = name.lower()

    if name_lower.startswith("resnet"):
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        orig_conv = model.conv1
        if in_channels != orig_conv.in_channels:
            k_tuple = orig_conv.kernel_size
            k_h, k_w = int(k_tuple[0]), int(k_tuple[1])
            if isinstance(orig_conv.stride, tuple):
                s_tuple = orig_conv.stride
                stride = (int(s_tuple[0]), int(s_tuple[1]))
            else:
                stride = int(orig_conv.stride)
            if isinstance(orig_conv.padding, str):
                padding = orig_conv.padding
            else:
                p_tuple = orig_conv.padding
                padding = (int(p_tuple[0]), int(p_tuple[1]))
            new_conv = nn.Conv2d(
                in_channels,
                orig_conv.out_channels,
                kernel_size=(k_h, k_w),
                stride=stride,
                padding=padding,
                bias=False,
            )
            with torch.no_grad():
                new_conv.weight[:, :3] = orig_conv.weight
                if in_channels > 3:
                    extra = orig_conv.weight.mean(dim=1, keepdim=True)
                    new_conv.weight[:, 3:] = extra.repeat(1, in_channels - 3, 1, 1)
            model.conv1 = new_conv
        features = nn.Sequential(*list(model.children())[:-1])
        if resnet_cfg:
            if resnet_cfg.use_group_norm:
                _convert_bn_to_groupnorm(features)
            if resnet_cfg.freeze:
                features.eval()
                for param in features.parameters():
                    param.requires_grad = False
        return features

    elif name_lower.startswith("efficientnet"):
        eff_cfg = efficientnet_cfg or EfficientNetEncoderConfig()
        # Determine weights enum, default to DEFAULT
        weight_enum = EfficientNet_B0_Weights.__members__.get(
            eff_cfg.weights, EfficientNet_B0_Weights.DEFAULT
        )
        model = efficientnet_b0(weights=weight_enum)
        # Remove classifier head to get feature extractor
        features = nn.Sequential(*list(model.children())[:-1])
        if eff_cfg.use_group_norm:
            _convert_bn_to_groupnorm(features)
        if eff_cfg.freeze:
            features.eval()
            for param in features.parameters():
                param.requires_grad = False
        return features

    elif name_lower.startswith("dinov2"):
        if dinov2_cfg is None:
            dinov2_cfg = DinoEncoderConfig()
        hf_model_id = dinov2_cfg.series
        logger.info("Loading DINOv2 model from HuggingFace: %s", hf_model_id)
        if dinov2_cfg.num_registers > 0:
            model = Dinov2WithRegistersModel.from_pretrained(hf_model_id)
            if (
                hasattr(model.config, "num_registers")
                and model.config.num_registers != dinov2_cfg.num_registers
            ):
                logger.warning(
                    "Loaded Dinov2WithRegistersModel '%s' has "
                    "model.config.num_registers=%s, but "
                    "DinoEncoderConfig.num_registers=%s. Using value from "
                    "DinoEncoderConfig.",
                    hf_model_id,
                    model.config.num_registers,
                    dinov2_cfg.num_registers,
                )
        else:
            model = Dinov2Model.from_pretrained(hf_model_id)

        orig_patch_proj = model.embeddings.patch_embeddings.projection
        if in_channels != orig_patch_proj.in_channels:
            new_patch_proj = nn.Conv2d(
                in_channels,
                orig_patch_proj.out_channels,
                kernel_size=orig_patch_proj.kernel_size,  # type: ignore
                stride=orig_patch_proj.stride,  # type: ignore
                padding=orig_patch_proj.padding,  # type: ignore
                bias=(orig_patch_proj.bias is not None),
            )
            with torch.no_grad():
                new_patch_proj.weight[:, : orig_patch_proj.in_channels] = (
                    orig_patch_proj.weight
                )
                if new_patch_proj.bias is not None and orig_patch_proj.bias is not None:
                    new_patch_proj.bias.data = orig_patch_proj.bias.data.clone()
                if in_channels > orig_patch_proj.in_channels:
                    mean_rgb_weights = orig_patch_proj.weight.mean(dim=1, keepdim=True)
                    num_new_channels = in_channels - orig_patch_proj.in_channels
                    new_channel_weights = mean_rgb_weights.repeat(
                        1, num_new_channels, 1, 1
                    )
                    new_patch_proj.weight[:, orig_patch_proj.in_channels :] = (
                        new_channel_weights
                    )
            model.embeddings.patch_embeddings.projection = new_patch_proj

        if dinov2_cfg.freeze:
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

        if dinov2_cfg.use_lora:
            if dinov2_cfg.freeze:
                pass
            inject_lora(model, dinov2_cfg.lora_r, dinov2_cfg.lora_alpha)
            if dinov2_cfg.freeze:
                for _name, param in model.named_parameters():
                    if "lora_A" not in _name and "lora_B" not in _name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True

        class DINOEncoder(nn.Module):
            def __init__(self, model_instance, num_model_registers: int):
                super().__init__()
                self.model = model_instance
                self.num_model_registers = num_model_registers

            def forward(
                self, x: torch.Tensor, output_attentions: bool = False
            ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
                model_output = self.model(x, output_attentions=output_attentions)
                feats = model_output.last_hidden_state
                num_tokens_before_patches = 1
                num_registers = self.num_model_registers
                total_sequence_length = feats.shape[1]
                num_patches = (
                    total_sequence_length - num_tokens_before_patches - num_registers
                )
                patch_tokens = feats[
                    :,
                    num_tokens_before_patches : num_tokens_before_patches + num_patches,
                ]
                h_w = int(math.sqrt(num_patches))
                if h_w * h_w != num_patches:
                    img_size = x.shape[-1]
                    patch_size = self.model.config.patch_size
                    h_w = img_size // patch_size
                    if h_w * h_w != num_patches:
                        raise ValueError(
                            f"Cannot reshape {num_patches} patch tokens into a square "
                            f"grid. H*W ({h_w * h_w}) does not match num_patches. "
                            f"Input image size: {img_size}, Patch size: {patch_size}"
                        )
                patch_tokens_spatial = patch_tokens.permute(0, 2, 1).reshape(
                    x.size(0), patch_tokens.size(2), h_w, h_w
                )
                if output_attentions:
                    return patch_tokens_spatial, model_output.attentions
                return patch_tokens_spatial

        return DINOEncoder(model, dinov2_cfg.num_registers)
    else:
        raise ValueError(f"Unsupported image encoder: {name}")
