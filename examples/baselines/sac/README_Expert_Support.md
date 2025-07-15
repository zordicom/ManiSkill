# SAC RGBD with Expert Support

This document explains the expert policy support added to `sac_rgbd.py`, which enables using pre-trained models (especially PPO RGB Fast) as expert policies for Expert+Residual learning.

## Overview

The updated `sac_rgbd.py` now supports Expert+Residual learning, where the final action is computed as:

```
final_action = expert_action + residual_action
```

This approach combines the knowledge from pre-trained expert policies with learned residual corrections, enabling more sample-efficient learning.

## New Features

### 1. Expert Types Supported

- **`none`**: Regular SAC RGBD (no expert) - baseline behavior
- **`zero`**: Zero expert (extended observation space for residual learning)
- **`ik`**: Inverse kinematics expert
- **`model`**: Generic model expert (loads PyTorch state_dict)
- **`ppo_rgb_fast`**: PPO RGB Fast expert (specifically designed for RGB observations)

### 2. PPO RGB Fast Expert

A new expert type specifically designed to use PPO RGB Fast models as experts:

- Loads PPO RGB Fast agent from checkpoint
- Handles RGB observations natively
- Uses deterministic action (actor mean) for expert guidance
- Automatically adapts to environment action dimensions

### 3. Expert Parameters

New command-line arguments for expert configuration:

```bash
--expert-type {none,zero,ik,model,ppo_rgb_fast}
--ppo-rgb-fast-path PATH_TO_CHECKPOINT
--model-path PATH_TO_MODEL
--residual-scale 0.5
--expert-action-noise 0.0
--track-action-stats
--ik-gain 2.0
```

## Usage Examples

### Basic SAC RGBD (No Expert)

```bash
python sac_rgbd.py \
    --env-id PickCube-v1 \
    --expert-type none \
    --num-envs 16 \
    --total-timesteps 1000000 \
    --include-state \
    --camera-width 128 \
    --camera-height 128
```

### SAC RGBD with Zero Expert

```bash
python sac_rgbd.py \
    --env-id PickCube-v1 \
    --expert-type zero \
    --residual-scale 1.0 \
    --num-envs 16 \
    --total-timesteps 1000000 \
    --include-state \
    --camera-width 128 \
    --camera-height 128 \
    --track-action-stats
```

### SAC RGBD with PPO RGB Fast Expert

```bash
python sac_rgbd.py \
    --env-id PickCube-v1 \
    --expert-type ppo_rgb_fast \
    --ppo-rgb-fast-path /path/to/ppo_rgb_fast_checkpoint.pt \
    --residual-scale 0.5 \
    --num-envs 16 \
    --total-timesteps 500000 \
    --include-state \
    --camera-width 128 \
    --camera-height 128 \
    --track-action-stats
```

### SAC RGBD with Model Expert

```bash
python sac_rgbd.py \
    --env-id PickCube-v1 \
    --expert-type model \
    --model-path /path/to/model_checkpoint.pt \
    --residual-scale 0.3 \
    --num-envs 16 \
    --total-timesteps 500000 \
    --include-state \
    --camera-width 128 \
    --camera-height 128 \
    --track-action-stats
```

### SAC RGBD with IK Expert

```bash
python sac_rgbd.py \
    --env-id PickCube-v1 \
    --expert-type ik \
    --ik-gain 2.0 \
    --residual-scale 0.5 \
    --num-envs 16 \
    --total-timesteps 500000 \
    --include-state \
    --camera-width 128 \
    --camera-height 128 \
    --track-action-stats
```

## Connection to Dataset Generator

The PPO RGB Fast expert is designed to work with models trained using the same configuration as the dataset generator:

```python
# From pick_cube_rgb_dataset_generator.py
env_config = {
    "id": "PickCube-v1",
    "robot_uids": "panda_wristcam",
    "control_mode": "pd_ee_delta_pos",
    "obs_mode": "rgb",
    "render_mode": "all",
    "sim_backend": "physx_cuda",
}
```

The expert policy expects the same observation format (RGB + state) and action space as the dataset generator.

## Implementation Details

### Expert Policy Creation

The `create_ppo_rgb_fast_expert` function:

1. Loads the PPO RGB Fast agent from checkpoint
2. Creates a wrapper function that provides deterministic actions
3. Handles device placement and observation format conversion
4. Returns a callable expert policy function

### Environment Wrapper Integration

The expert support integrates with ManiSkill's Expert+Residual wrapper system:

1. **Expert Registration**: New expert types are registered with the expert policy system
2. **Wrapper Creation**: Uses `create_expert_residual_envs` from `util.py`
3. **Observation Handling**: Maintains RGB observation format through the wrapper
4. **Action Space**: Automatically adapts to environment action dimensions

### Observation Format

The expert expects observations in the format:
```python
{
    "rgb": torch.Tensor,      # Shape: (batch, H, W, C)
    "state": torch.Tensor,    # Shape: (batch, state_dim)
}
```

This matches the output of `FlattenRGBDObservationWrapper` used in the SAC RGBD implementation.

## Performance Considerations

### Expert Type Recommendations

- **PPO RGB Fast**: Best for RGB-based manipulation tasks with pre-trained models
- **Zero Expert**: Good baseline for residual learning without specific expert knowledge
- **IK Expert**: Effective for reaching/manipulation tasks with known kinematics
- **Model Expert**: Flexible option for other pre-trained models

### Hyperparameter Tuning

- **`residual_scale`**: Start with 0.5 for PPO RGB Fast, 1.0 for zero expert
- **`expert_action_noise`**: Usually 0.0 for deterministic experts
- **`utd`**: Consider lower values (0.25-0.5) for expert-guided learning
- **`batch_size`**: May need larger batches for stable learning with experts

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure `util.py` is available in the PPO directory
2. **Checkpoint Loading**: Verify PPO RGB Fast checkpoint path and format
3. **Action Dimension Mismatch**: Check that expert and environment use same control mode
4. **Memory Issues**: Reduce batch size or number of environments for RGB observations

### Debug Tips

```bash
# Enable action statistics tracking
--track-action-stats

# Use smaller environments for testing
--num-envs 4 --num-eval-envs 2

# Reduce total timesteps for quick testing
--total-timesteps 10000 --eval-freq 10
```

## Testing

Use the provided test script to verify expert functionality:

```bash
python test_sac_rgbd_expert.py
```

This script shows command-line examples for all supported expert types and can be used to verify the implementation works correctly.

## Integration with Existing Workflows

The expert support is designed to be:

- **Backward Compatible**: Default behavior (`expert_type=none`) is unchanged
- **Modular**: Each expert type can be enabled/disabled independently
- **Configurable**: Extensive hyperparameter control for fine-tuning
- **Extensible**: Easy to add new expert types following the same pattern

## Future Enhancements

Potential improvements for expert support:

- **Adaptive Residual Scaling**: Dynamic scaling based on expert confidence
- **Multi-Expert Fusion**: Combining multiple expert policies
- **Curriculum Learning**: Progressive transition from expert to residual
- **Expert Evaluation**: Metrics to assess expert policy quality 