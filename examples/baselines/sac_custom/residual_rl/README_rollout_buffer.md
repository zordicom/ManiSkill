# Rollout Buffer Management for SAC Delta Action Training

This document describes the implementation of the rollout buffer management system for SAC delta action training, which implements a two-bucket sampling strategy with grade-weighted and age-based sampling.

## Overview

The rollout buffer management system provides:

1. **Two-bucket sampling**: Separates base policy demos from residual rollouts
2. **Grade-weighted sampling**: Prioritizes higher-quality episodes based on human feedback grades
3. **Age-based decay**: Gives preference to more recent episodes
4. **Validation holdout**: Automatically holds out the best episodes for validation
5. **Housekeeping utilities**: Tools for cleaning up old episodes and managing disk space

## Key Components

### 1. Configuration (`rl_configs.py`)

```python
@attr.s(auto_attribs=True, frozen=True)
class RolloutBufferConfig:
    # Two-bucket sampling configuration
    demo_fraction: float = 0.3  # Fraction from base policy demos

    # Grade-weighted sampling for residual rollouts
    grade_weight_beta: float = 0.5  # Exponential weight for grades
    min_grade: int = 2  # Minimum grade to include

    # Age-based decay
    age_decay_lambda: float = 0.1  # Decay rate for age
    max_age_days: float = 14.0  # Maximum age to include

    # Buffer management
    max_episodes_per_model: int = 50
    max_total_episodes: int = 500
    val_holdout_fraction: float = 0.05
```

### 2. Rollout Buffer (`rollout_buffer.py`)

The `RolloutBuffer` class manages episode sampling with the following features:

- **Episode Metadata Extraction**: Automatically extracts grades, timestamps, and other metadata
- **Two-Bucket Categorization**: Separates `base_policy_only` demos from model rollouts
- **Filtering**: Applies grade and age filters to residual episodes
- **Weighted Sampling**: Uses exponential weighting based on grade and age
- **Validation Holdout**: Automatically reserves best episodes for validation

### 3. Dataset Integration (`rl_dataset.py`)

The `RLDataset` class has been modified to:

- Initialize a `RolloutBuffer` instance
- Sample episodes using the two-bucket strategy for training
- Use held-out episodes for validation
- Maintain compatibility with existing data loading pipeline

## Usage

### Configuration

Add the rollout buffer configuration to your YAML config file:

```yaml
# rollout_buffer section in rl_galaxea_sac_box_pnp.yaml
rollout_buffer:
  demo_fraction: 0.3  # 30% from base policy demos
  grade_weight_beta: 0.5  # Exponential weight for grades
  min_grade: 2  # Only include episodes with grade >= 2
  age_decay_lambda: 0.1  # Decay rate for age
  max_age_days: 14.0  # Only include episodes from last 14 days
  max_episodes_per_model: 50
  max_total_episodes: 500
  val_holdout_fraction: 0.05
```

### Training

The rollout buffer is automatically used when creating the dataset:

```python
from rl_configs import RLConfig
from rl_dataset import RLDataset

# Load configuration
rl_cfg = RLConfig(...)

# Create training dataset (uses rollout buffer sampling)
train_dataset = RLDataset(cfg_rl=rl_cfg, is_train_split=True)

# Create validation dataset (uses held-out episodes)
val_dataset = train_dataset.get_validation_dataset()
```

### Testing

Test the rollout buffer functionality:

```bash
cd playground/rl/residual_rl
python test_rollout_buffer.py --config rl_galaxea_sac_box_pnp.yaml
```

### Housekeeping

Use the maintenance script for rollout pool management:

```bash
# Generate a detailed report
python scripts/maintain_rollout_pool.py --generate-report

# Clean up old episodes (dry run)
python scripts/maintain_rollout_pool.py --clean --dry-run

# Actually clean up old episodes
python scripts/maintain_rollout_pool.py --clean --max-age-days 30 --min-grade 2

# Recompute aggregated statistics
python scripts/maintain_rollout_pool.py --recompute-stats
```

## Data Structure

The system expects the following directory structure:

```
data/galaxea_rollouts/box_pnp/
├── aggregated_stats.json
├── base_policy_only/           # Bucket A: Base policy demos
│   ├── 20250706_160929/
│   │   ├── observations.json
│   │   ├── inference_grade.json
│   │   └── ...
│   └── ...
├── model_20250706_164917/      # Bucket B: Residual rollouts
│   ├── 20250706_174258/
│   │   ├── observations.json
│   │   ├── inference_grade.json
│   │   └── ...
│   └── ...
└── model_20250707_122934/      # More residual rollouts
    └── ...
```

Each episode directory must contain:

- `observations.json`: Episode data
- `inference_grade.json`: Grade feedback (optional, defaults to 0)

## Sampling Strategy

### Bucket A (Base Policy Demos)

- **Source**: `base_policy_only/` directory
- **Sampling**: Uniform random sampling
- **Fraction**: Configured by `demo_fraction` (default: 30%)

### Bucket B (Residual Rollouts)

- **Source**: `model_*/` directories
- **Filtering**:
  - Grade >= `min_grade` (default: 2)
  - Age <= `max_age_days` (default: 14 days)
- **Sampling**: Weighted by `exp(β * grade) * exp(-λ * age_days)`
- **Fraction**: 1 - `demo_fraction` (default: 70%)

### Validation Holdout

- **Source**: Best residual episodes (highest grade, most recent)
- **Fraction**: `val_holdout_fraction` (default: 5% of filtered residual episodes)

## Grade Levels

The system uses a 7-level grading system:

- **0**: Failed
- **1**: Stage 1 -- Approached target object
- **2**: Stage 2 -- Picked up target object
- **3**: Stage 3 -- Moved object to placement location
- **4**: Stage 4 -- Dropped object in placement location
- **5**: Stage 5 -- Finished successfully and returned to rest
- **6**: Stage 6 -- Finished gracefully and swiftly

## Benefits

1. **Balanced Training**: Maintains connection to base policy while learning from improvements
2. **Quality Focus**: Prioritizes high-quality episodes for learning
3. **Recency Bias**: Adapts to recent improvements in the policy
4. **Automatic Validation**: Provides clean validation episodes without manual curation
5. **Scalable**: Handles growing datasets with configurable limits
6. **Maintainable**: Includes tools for cleanup and statistics recomputation

## Implementation Notes

- **Thread-safe**: The rollout buffer can be safely used with multiple DataLoader workers
- **Lazy loading**: Episodes are loaded on-demand to minimize memory usage
- **Caching**: Metadata is cached for efficient repeated sampling
- **Robust**: Handles missing files and malformed data gracefully
- **Configurable**: All parameters can be tuned via the configuration file
