# SAC Custom: Multimodal SAC for ManiSkill

This directory contains a custom SAC implementation that combines multimodal encoders (inspired by `try_sac_delta_action.py`) with the ManiSkill environment setup and training pipeline (from `sac_rgbd.py`).

## Key Features

- **Multimodal Encoder**: Processes both state vectors and RGB images
- **Vision Backbone Options**:
  - DINOv2-small (frozen, requires `transformers` library)
  - Simple CNN (lightweight alternative)
- **Standard SAC**: Direct action learning without expert/delta concepts
- **ManiSkill Integration**: Full environment setup, video recording, checkpointing

## Files

- `sac_custom.py`: Main training script (similar interface to `sac_rgbd.py`)
- `multimodal_encoders.py`: Vision encoders (DINOv2 + Simple CNN)
- `sac_networks.py`: SAC policy and Q-networks (standalone)
- `test_sac_components.py`: Unit tests for all components

## Usage

### Basic Training (with Simple CNN)

```bash
python sac_custom.py --env_id="PickCube-v1" --obs_mode="rgb" \
  --num_envs=32 --utd=0.5 --buffer_size=300_000 \
  --control-mode="pd_ee_delta_pos" --camera_width=64 --camera_height=64 \
  --total_timesteps=1_000_000 --eval_freq=10_000 --use_dinov2=False
```

### Training with DINOv2 (requires transformers)

```bash
# First install transformers: pip install transformers
python sac_custom.py --env_id="PickCube-v1" --obs_mode="rgb" \
  --num_envs=32 --utd=0.5 --buffer_size=300_000 \
  --control-mode="pd_ee_delta_pos" --camera_width=64 --camera_height=64 \
  --total_timesteps=1_000_000 --eval_freq=10_000 --use_dinov2=True
```

### Run Tests

```bash
python test_sac_components.py
```

## Architecture

### MultimodalEncoder

- **State Encoder**: MLP with LayerNorm for proprioceptive state
- **Image Encoder**: DINOv2 (frozen) or Simple CNN for RGB images
- **Fusion Layer**: Concatenates and processes combined features

### SAC Networks

- **Policy Network**: Outputs mean and log_std for Gaussian policy
- **Q-Networks**: Dual critics with shared encoder for stability
- **Target Networks**: Polyak-averaged targets for stable learning

## Key Differences from Original Scripts

### From `try_sac_delta_action.py`

- ✅ Kept: Multimodal encoder architecture, LayerNorm, orthogonal init
- ❌ Removed: Expert action handling, delta action concepts, offline RL dataset
- ✅ Added: Direct action learning, online environment interaction

### From `sac_rgbd.py`

- ✅ Kept: ManiSkill environment setup, training loop, video recording
- ❌ Removed: Simple PlainConv encoder
- ✅ Added: Advanced multimodal encoder, DINOv2 support

## Performance Notes

- **Simple CNN**: Fast, lightweight, good for quick experiments
- **DINOv2**: Better feature quality, slower, requires more memory
- **Memory Usage**: Reduce `buffer_size` if running out of GPU memory
- **Training Speed**: Use smaller `camera_width/height` for faster training

## Troubleshooting

1. **Import Errors**: If `transformers` not available, set `--use_dinov2=False`
2. **GPU Memory**: Reduce `buffer_size`, `num_envs`, or image resolution
3. **Slow Training**: Try `--use_dinov2=False` for faster iterations

## Citation

Based on:

- SAC: Haarnoja et al. "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"
- ManiSkill: Gu et al. "ManiSkill: Generalizable Manipulation Skill Benchmark"
