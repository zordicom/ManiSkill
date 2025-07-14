"""
Copyright 2025 Zordi, Inc. All rights reserved.

Simplified multimodal encoders for SAC with DINOv2 vision backbone.
"""

from typing import Optional

import torch
from torch import nn
from torchvision import models
from transformers.models.dinov2.modeling_dinov2 import Dinov2Model


def _init_weights(module: nn.Module) -> None:
    """Initialize weights using orthogonal initialization for numerical stability."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=1)
        nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.Conv2d):
        nn.init.orthogonal_(module.weight, gain=1)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


class DINOv2Encoder(nn.Module):
    """Simplified DINOv2 encoder that outputs spatial features."""

    def __init__(self, in_channels: int = 3, output_dim: int = 256):
        super().__init__()

        # Load frozen DINOv2-small model
        self.model = Dinov2Model.from_pretrained("facebook/dinov2-small")

        # Adjust input channels if needed
        orig_patch_proj = self.model.embeddings.patch_embeddings.projection
        if in_channels != orig_patch_proj.in_channels:
            new_patch_proj = nn.Conv2d(
                in_channels,
                orig_patch_proj.out_channels,
                kernel_size=orig_patch_proj.kernel_size,
                stride=orig_patch_proj.stride,
                padding=orig_patch_proj.padding,
                bias=(orig_patch_proj.bias is not None),
            )
            with torch.no_grad():
                new_patch_proj.weight[:, : orig_patch_proj.in_channels] = orig_patch_proj.weight
                if new_patch_proj.bias is not None and orig_patch_proj.bias is not None:
                    new_patch_proj.bias.data = orig_patch_proj.bias.data.clone()
                if in_channels > orig_patch_proj.in_channels:
                    mean_rgb_weights = orig_patch_proj.weight.mean(dim=1, keepdim=True)
                    num_new_channels = in_channels - orig_patch_proj.in_channels
                    new_channel_weights = mean_rgb_weights.repeat(1, num_new_channels, 1, 1)
                    new_patch_proj.weight[:, orig_patch_proj.in_channels :] = new_channel_weights
            self.model.embeddings.patch_embeddings.projection = new_patch_proj

        # Freeze the model
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # Projection layer to output_dim
        self.projection = nn.Linear(384, output_dim)  # DINOv2-small has 384 hidden size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning spatial features [B, output_dim]."""
        with torch.no_grad():
            model_output = self.model(x)
            feats = model_output.last_hidden_state  # [B, num_tokens, hidden_size]

            # Extract patch tokens (skip CLS token)
            patch_tokens = feats[:, 1:]  # [B, num_patches, hidden_size]

            # Global average pooling over spatial patches
            pooled_features = patch_tokens.mean(dim=1)  # [B, hidden_size]

        # Project to desired output dimension
        return self.projection(pooled_features)


class EfficientNetB0Encoder(nn.Module):
    """EfficientNet-B0 encoder with pre-trained weights."""

    def __init__(self, in_channels: int = 3, output_dim: int = 256):
        super().__init__()

        # Load pre-trained EfficientNet-B0
        self.efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

        # Adjust first layer if needed for different input channels
        if in_channels != 3:
            # Replace first conv layer
            orig_conv = self.efficientnet.features[0][0]
            new_conv = nn.Conv2d(
                in_channels,
                orig_conv.out_channels,
                kernel_size=orig_conv.kernel_size,
                stride=orig_conv.stride,
                padding=orig_conv.padding,
                bias=orig_conv.bias is not None,
            )

            # Initialize with pre-trained weights for first 3 channels
            with torch.no_grad():
                if in_channels >= 3:
                    new_conv.weight[:, :3] = orig_conv.weight
                    if in_channels > 3:
                        # Initialize extra channels with mean of RGB weights
                        mean_weight = orig_conv.weight.mean(dim=1, keepdim=True)
                        for i in range(3, in_channels):
                            new_conv.weight[:, i : i + 1] = mean_weight
                else:
                    # If fewer than 3 channels, use subset of weights
                    new_conv.weight = orig_conv.weight[:, :in_channels]

                if new_conv.bias is not None and orig_conv.bias is not None:
                    new_conv.bias.data = orig_conv.bias.data.clone()

            self.efficientnet.features[0][0] = new_conv

        # Remove the classifier head (we'll add our own)
        self.efficientnet.classifier = nn.Identity()

        # Add custom projection layer
        # EfficientNet-B0 outputs 1280 features
        self.projection = nn.Sequential(nn.Linear(1280, 512), nn.ReLU(), nn.Linear(512, output_dim))

        # Apply weight initialization to projection layer
        self.projection.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning features [B, output_dim]."""
        x = x.float() / 255.0  # Normalize to [0, 1]
        features = self.efficientnet(x)  # [B, 1280]
        return self.projection(features)


class SimpleConvEncoder(nn.Module):
    """Simple CNN encoder for RGB images."""

    def __init__(self, in_channels: int = 3, output_dim: int = 256, image_size: tuple = (64, 64)):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )

        # Calculate flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, *image_size)
            cnn_output_size = self.cnn(dummy_input).shape[1]

        self.fc = nn.Sequential(nn.Linear(cnn_output_size, 512), nn.ReLU(), nn.Linear(512, output_dim))

        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning features [B, output_dim]."""
        x = x.float() / 255.0  # Normalize to [0, 1]
        features = self.cnn(x)
        return self.fc(features)


class MultimodalEncoder(nn.Module):
    """Multimodal encoder for state + RGB observations."""

    def __init__(
        self,
        state_dim: int,
        image_channels: int = 3,
        image_size: tuple = (64, 64),
        output_dim: int = 512,
        use_dinov2: bool = True,
        use_efficientnet: bool = False,
        image_encoder: Optional[nn.Module] = None,
        state_encoder_dims: list = [256, 256],
        fusion_dims: list = [512, 512],
    ):
        super().__init__()

        self.state_dim = state_dim
        self.output_dim = output_dim

        # State encoder
        state_layers = []
        prev_dim = state_dim
        for dim in state_encoder_dims:
            state_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
            ])
            prev_dim = dim
        self.state_encoder = nn.Sequential(*state_layers)

        # Image encoder (optionally shared across networks)
        image_output_dim = 256  # Both DINOv2Encoder and SimpleConvEncoder output 256-D features

        # Determine number of cameras based on image channels
        if (use_dinov2 or use_efficientnet) and image_channels % 3 == 0:
            self.num_cameras = image_channels // 3
        else:
            self.num_cameras = 1

        if image_encoder is not None:
            # Reuse the provided encoder to avoid redundant initialisation (e.g. heavy DINOv2 model)
            self.image_encoder = image_encoder
        elif use_efficientnet:
            if image_channels % 3 == 0:
                # Multi-camera setup: split channels and process each camera separately
                self.image_encoder = EfficientNetB0Encoder(3, image_output_dim)  # Process 3 channels at a time
                print(f"📷 Multi-camera setup detected: {self.num_cameras} cameras, using EfficientNet-B0 for each")
            else:
                print(f"⚠️ Warning: EfficientNet-B0 works best with 3 channels, but got {image_channels}. Using anyway.")
                self.image_encoder = EfficientNetB0Encoder(image_channels, image_output_dim)
        elif use_dinov2:
            if image_channels % 3 == 0:
                # Multi-camera setup: split channels and process each camera separately
                self.image_encoder = DINOv2Encoder(3, image_output_dim)  # Process 3 channels at a time
                print(f"📷 Multi-camera setup detected: {self.num_cameras} cameras, using DINOv2 for each")
            else:
                print(
                    f"⚠️ Warning: DINOv2 requires channels divisible by 3, but got {image_channels}. Falling back to SimpleConvEncoder"
                )
                self.image_encoder = SimpleConvEncoder(image_channels, image_output_dim, image_size)
        else:
            self.image_encoder = SimpleConvEncoder(image_channels, image_output_dim, image_size)

        # Fusion layer
        total_dim = state_encoder_dims[-1] + image_output_dim
        fusion_layers = []
        prev_dim = total_dim
        for dim in fusion_dims:
            fusion_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
            ])
            prev_dim = dim
        self.fusion = nn.Sequential(*fusion_layers)

        # Apply initialization
        self.apply(_init_weights)

    def forward(self, obs: dict) -> torch.Tensor:
        """Forward pass through multimodal encoder.

        Args:
            obs: Dictionary with 'state' and 'rgb' keys

        Returns:
            Fused features [B, output_dim]
        """
        features = []

        # Encode state
        state_feat = self.state_encoder(obs["state"])
        features.append(state_feat)

        # Encode RGB image
        rgb = obs["rgb"]
        if rgb.dim() == 4 and len(rgb.shape) == 4:
            if rgb.shape[1] == rgb.shape[2] and rgb.shape[1] > rgb.shape[3]:
                # Likely [B, H, W, C] format
                rgb = rgb.permute(0, 3, 1, 2)  # Convert to [B, C, H, W]

            # Handle multi-camera processing
            if self.num_cameras > 1:
                # Split channels into groups of 3 for each camera
                batch_size, channels, height, width = rgb.shape
                rgb_split = rgb.view(batch_size, self.num_cameras, 3, height, width)

                # Process each camera separately
                camera_features = []
                for i in range(self.num_cameras):
                    camera_rgb = rgb_split[:, i, :, :, :]  # [B, 3, H, W]
                    camera_feat = self.image_encoder(camera_rgb)
                    camera_features.append(camera_feat)

                # Combine features from all cameras (mean pooling)
                img_feat = torch.stack(camera_features, dim=1).mean(dim=1)  # [B, output_dim]
            else:
                # Single camera processing
                img_feat = self.image_encoder(rgb)
        else:
            raise ValueError(f"Unexpected RGB shape: {rgb.shape}. Expected 4D tensor.")
        features.append(img_feat)

        # Fuse features
        combined = torch.cat(features, dim=1)
        return self.fusion(combined)
