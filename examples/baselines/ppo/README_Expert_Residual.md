# Expert+Residual PPO for ManiSkill

High-performance Expert+Residual action decomposition with optimized PPO for ManiSkill environments.

## Overview

Expert+Residual combines expert policies with learned residual corrections:

```
final_action = expert_action + residual_action
```

## Quick Start

### State-based Observations
```bash
# PickBox with A1 Galaxea + Zero Expert (pure RL)
python ppo_fast_expert_residual.py \
    --env-id PickBox-v1 \
    --robot-uids a1_galaxea \
    --expert-type zero \
    --num-envs 1024 \
    --total-timesteps 100000
```

### RGB Observations
```bash
# PickBox with RGB + A1 Galaxea + Zero Expert (pure RL)
python ppo_rgb_fast_expert_residual.py \
    --env-id PickBox-v1 \
    --robot-uids a1_galaxea \
    --expert-type zero \
    --num-envs 8 \
    --total-timesteps 10000 \
    --compile
```

## Expert Types

### Zero Expert (Baseline)
No expert knowledge - pure RL learning:

**State-based:**
```bash
python ppo_fast_expert_residual.py \
    --env-id PickBox-v1 \
    --robot-uids a1_galaxea \
    --expert-type zero \
    --num-envs 1024 \
    --total-timesteps 100000
```

**RGB-based:**
```bash
python ppo_rgb_fast_expert_residual.py \
    --env-id PickBox-v1 \
    --robot-uids a1_galaxea \
    --expert-type zero \
    --num-envs 32 \
    --total-timesteps 10000 \
    --compile
```

### IK Expert (Coming Soon)
Uses inverse kinematics for manipulation guidance.

### Model Expert (Coming Soon)  
Uses pre-trained models as expert policies.

## Observation Types

### State vs RGB Observations

**State-based (`ppo_fast_expert_residual.py`):**
- Uses flattened state observations (joint positions, velocities, etc.)
- Faster training, lower memory usage
- Can handle more parallel environments (512-1024+)
- Better for tasks where spatial reasoning is less critical

**RGB-based (`ppo_rgb_fast_expert_residual.py`):**
- Uses camera images for visual learning
- Slower training, higher memory usage
- Fewer parallel environments recommended (16-64)
- Better for tasks requiring visual understanding
- Includes fast optimizations: `--compile`, CUDA graphs, TensorDict

**Memory Considerations:**
- RGB: Use fewer environments (16-64) due to camera buffer memory
- State: Can use many environments (512-1024+) for faster training

## Key Parameters

| Parameter | Description | Default | Examples |
|-----------|-------------|---------|----------|
| `--env-id` | ManiSkill environment | `PickBox-v1` | `PushCube-v1`, `StackCube-v1` |
| `--robot-uids` | Robot to use | `a1_galaxea` | `panda`, `franka` |
| `--expert-type` | Expert policy type | `zero` | `ik`, `model` |
| `--num-envs` | Parallel environments | `1024` | `64`, `256`, `512`, `2048` |
| `--total-timesteps` | Training steps | `100000` | `25000`, `50000` |
| `--residual-scale` | Scale residual actions | `1.0` | `0.1`, `0.5`, `2.0` |

## Environment Examples

### PickBox Task

**State-based Training:**
```bash
# A1 Galaxea robot with state observations
python ppo_fast_expert_residual.py \
    --env-id PickBox-v1 \
    --robot-uids a1_galaxea \
    --expert-type zero \
    --num-envs 1024 \
    --total-timesteps 100000
```

**RGB-based Training:**
```bash
# A1 Galaxea robot with RGB observations
python ppo_rgb_fast_expert_residual.py \
    --env-id PickBox-v1 \
    --robot-uids a1_galaxea \
    --expert-type zero \
    --num-envs 32 \
    --total-timesteps 10000 \
    --compile


## Files

- `ppo_fast_expert_residual.py` - State-based training script with optimizations
- `ppo_rgb_fast_expert_residual.py` - RGB-based training script with vision support
- `../../../mani_skill/envs/wrappers/expert_residual.py` - Expert+Residual wrapper
- `../../../mani_skill/envs/wrappers/README.md` - Wrapper documentation

## Future Work

- **IK Expert**: Inverse kinematics-based expert policies
- **Model Expert**: Pre-trained model integration  
- **Adaptive Scaling**: Dynamic residual scaling based on expert confidence
- **Multi-Expert**: Ensemble of different expert policies

Happy training! 🚀 