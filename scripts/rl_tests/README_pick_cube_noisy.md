# PickCube Noisy Environment - PPO Robustness Study

## Overview

This directory contains a comprehensive PPO robustness study for the PickCubeNoisy environment, designed to test how well PPO agents handle various types of noise in observations, rewards, and actions.

## Features

- **Live Output**: Real-time PPO training progress display
- **Noise Injection**: Configurable noise in observations, rewards, and actions
- **Curriculum Learning**: Progressive noise scheduling over training episodes
- **Multiple Robots**: Support for all robots from the original PickCube environment
- **Comprehensive Evaluation**: 12 different noise configurations
- **Video Recording**: Training and evaluation videos for analysis
- **CSV Export**: Automated export of results to multiple CSV formats
- **WandB Integration**: Experiment tracking and visualization with live metrics

## Files

- `pick_cube_noisy_ppo.py` - Main experiment runner (12 configurations, **live output**)
- `pick_cube_noisy_ppo_runner.py` - PPO wrapper script with noise parameter handling
- `test_pick_cube_noisy_single.py` - Single test script for verification (**live output**)
- `test_pick_cube_noisy_main_single.py` - Test single main experiment (**live output**)
- `CSV_Export_Documentation.md` - Detailed CSV export documentation

## Quick Start

### 1. Prerequisites

```bash
cd /home/gilwoo/workspace/ManiSkill
conda activate maniskill
pip install pandas  # Required for CSV export
```

### 2. Run Single Test

```bash
# Quick test with live PPO output
python scripts/rl_tests/test_pick_cube_noisy_single.py
```

### 3. Test Single Main Experiment

```bash
# Test one configuration from main suite with live output
python scripts/rl_tests/test_pick_cube_noisy_main_single.py
```

### 4. Run Full Experiment Suite

```bash
# Run all 12 experiments with live output (4-6 hours)
python scripts/rl_tests/pick_cube_noisy_ppo.py
```

## Experiment Configurations

The study includes 12 different noise configurations:

1. **No Noise** (Control) - Baseline performance
2. **Light Observation Noise** - σ = 0.005
3. **Light Reward Noise** - σ = 0.05
4. **Medium Observation Noise** - σ = 0.01
5. **Medium Reward Noise** - σ = 0.1
6. **Medium Combined Noise** - Obs + Reward
7. **Heavy Observation Noise** - σ = 0.02
8. **Heavy Reward Noise** - σ = 0.2
9. **Heavy Combined Noise** - Obs + Reward (heavy)
10. **Medium Action Noise** - σ = 0.05
11. **All Medium Noise** - Obs + Reward + Action
12. **Curriculum Learning** - Growing noise over episodes

## Technical Details

### PPO Training Parameters

- **Total Timesteps**: 300,000 per experiment
- **Training Environments**: 1024 parallel environments
- **Evaluation Environments**: 16 parallel environments
- **Evaluation Frequency**: Every 25 iterations
- **Control Mode**: `pd_joint_delta_pos`
- **Observation Mode**: `state`

### Noise Parameters

- **Observation Noise**: Applied to TCP pose, goal position, object pose, and relative positions
- **Reward Noise**: Applied to dense reward computation
- **Action Noise**: Applied to robot actions before execution
- **Curriculum Learning**: Progressive noise increase from 10% to 100% over episodes

### Live Output Display

All scripts now show **real-time PPO training output**:
- Environment setup progress
- Training iterations with loss values
- Success rate metrics (success_once, etc.)
- Noise parameter confirmations
- TensorBoard/WandB logging info

**Note**: After training completes, each model is automatically evaluated to determine accurate success rates for CSV export and analysis.

### CSV Export

The system automatically exports results to 4 CSV files:
- `interim_results.csv` - Real-time progress tracking
- `final_results.csv` - Complete raw results  
- `detailed_analysis.csv` - Enhanced analysis with categories
- `summary_statistics.csv` - Aggregated statistics

## Expected Results

The study evaluates PPO robustness across different noise conditions:

- **Control (No Noise)**: Baseline performance (~80-90% success rate)
- **Light Noise**: Minimal performance degradation (<10% drop)
- **Medium Noise**: Moderate performance impact (10-30% drop)
- **Heavy Noise**: Significant performance degradation (30-50% drop)
- **Curriculum Learning**: Improved robustness compared to fixed noise

## Troubleshooting

### Common Issues

1. **Argument Parsing Errors**
   - Fixed: Boolean arguments now use correct `tyro` format (`--flag` instead of `--flag True`)
   - The PPO runner script properly filters noise parameters

2. **Memory Issues**
   - Use fewer environments if running into memory constraints
   - Adjust batch size in PPO configuration

3. **CUDA Errors**
   - Ensure CUDA is available and compatible
   - Set `--no-cuda` to disable GPU usage if needed

### Fixed Issues

✅ **Boolean Argument Format**: Updated to use `tyro` format (`--flag` for True, `--no-flag` for False)
✅ **Noise Parameter Filtering**: PPO runner now properly filters noise arguments
✅ **CSV Export**: Comprehensive CSV export with detailed analysis
✅ **Environment Registration**: All robot variants properly registered

## Recent Updates

- **2025-01-09**: Fixed argument parsing issue with boolean flags
- **2025-01-09**: Added comprehensive CSV export functionality
- **2025-01-09**: Updated documentation with troubleshooting section
- **2025-01-09**: Verified all components working correctly

## Example Output

```
🎮 PickCubeNoisy PPO Robustness Study
==================================================
🚀 PickCubeNoisy PPO Experiment Manager initialized
📁 Output directory: runs/pick_cube_noisy_robustness_study
🎯 Training: 300,000 timesteps with 1024 envs
📊 Evaluation: 16 envs every 25 iterations
🎬 Videos: Enabled

🎯 Starting 12 PickCubeNoisy PPO experiments
================================================================================

📊 Experiment 1/12

🔬 Running experiment: Control (no noise)
📝 Experiment name: pick_cube_noisy_no_noise_1752055930
🚀 Command: /home/gilwoo/miniconda3/envs/maniskill/bin/python ...
```

## Support

For questions or issues, check:
1. The troubleshooting section above
2. The CSV export documentation
3. The individual test scripts for debugging

The system is now fully functional and ready for comprehensive PPO robustness analysis! 