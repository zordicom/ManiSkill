# SAC Custom: Multimodal SAC for ManiSkill

This directory contains a custom SAC implementation that combines multimodal encoders with the ManiSkill environment setup and training pipeline. The implementation uses **MobileNet v3 small** as the default vision encoder for fast training and quick iterations.

## Key Features

- **Multimodal Encoder**: Processes both state vectors and RGB images
- **Vision Backbone Options**:
  - **MobileNet v3 small** (default, fast and lightweight)
  - EfficientNet-B0 (heavier but potentially more accurate)
  - Simple CNN (lightweight fallback)
- **Standard SAC**: Direct action learning with continuous control
- **ManiSkill Integration**: Full environment setup, video recording, checkpointing

## Files

- `sac_custom.py`: Main training script (similar interface to `sac_rgbd.py`)
- `multimodal_encoders.py`: Vision encoders (MobileNet v3 small + EfficientNet-B0 + Simple CNN)
- `sac_networks.py`: SAC policy and Q-networks (standalone)
- `test_sac_components.py`: Unit tests for all components

## Usage

### Basic Training (with MobileNet v3 small - Default)

```bash
python sac_custom.py --env_id="PickCube-v1" --obs_mode="rgb" \
  --num_envs=32 --utd=0.5 --buffer_size=300_000 \
  --control-mode="pd_ee_delta_pos" --camera_width=64 --camera_height=64 \
  --total_timesteps=1_000_000 --eval_freq=10_000
```

### Training with EfficientNet-B0

```bash
python sac_custom.py --env_id="PickCube-v1" --obs_mode="rgb" \
  --num_envs=32 --utd=0.5 --buffer_size=300_000 \
  --control-mode="pd_ee_delta_pos" --camera_width=64 --camera_height=64 \
  --total_timesteps=1_000_000 --eval_freq=10_000 \
  --use_mobilenet=False --use_efficientnet=True
```

### Training with Simple CNN (lightest option)

```bash
python sac_custom.py --env_id="PickCube-v1" --obs_mode="rgb" \
  --num_envs=32 --utd=0.5 --buffer_size=300_000 \
  --control-mode="pd_ee_delta_pos" --camera_width=64 --camera_height=64 \
  --total_timesteps=1_000_000 --eval_freq=10_000 \
  --use_mobilenet=False --use_efficientnet=False
```

### Run Tests

```bash
python test_sac_components.py
```

## Architecture

### MultimodalEncoder

- **State Encoder**: MLP with LayerNorm for proprioceptive state
- **Image Encoder**: MobileNet v3 small (default), EfficientNet-B0, or Simple CNN for RGB images
- **Fusion Layer**: Concatenates and processes combined features

### SAC Networks

- **Policy Network**: Outputs mean and log_std for Gaussian policy
- **Q-Networks**: Twin Q-networks for value estimation
- **Action Rescaling**: Automatically handles action space scaling

## Vision Encoder Comparison

| Encoder | Speed | Memory | Accuracy | Use Case |
|---------|-------|---------|----------|----------|
| MobileNet v3 small | ⚡⚡⚡ | 💾 | ⭐⭐⭐ | Default, fast iterations |
| EfficientNet-B0 | ⚡⚡ | 💾💾 | ⭐⭐⭐⭐ | Higher accuracy needed |
| Simple CNN | ⚡⚡⚡ | 💾 | ⭐⭐ | Minimal setup, debugging |

## Multi-Camera Support

The implementation automatically handles multi-camera setups:

- Detects when `image_channels` is divisible by 3
- Processes each camera separately through the same encoder
- Combines features using mean pooling
- Works with all encoder types

## Performance Tips

- **MobileNet v3 small**: Best for quick iterations and prototyping
- **EfficientNet-B0**: Use when you need higher accuracy and have more compute
- **Multi-camera**: Helps with robustness but increases training time
- **Buffer size**: Increase for better sample efficiency (300k-1M recommended)
- **UTD ratio**: 0.5 provides good balance between speed and performance

## Environment Compatibility

Tested with:

- PickCube-v1
- PickBox-v1
- Other ManiSkill manipulation tasks

## Dependencies

- PyTorch
- torchvision (for MobileNet and EfficientNet)
- ManiSkill environment
- Standard RL libraries (gymnasium, numpy, etc.)

## Citation

Based on the SAC algorithm with custom multimodal encoders optimized for robotic manipulation tasks.
