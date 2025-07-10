# CSV Export Documentation - PickCubeNoisy PPO Experiments

## Overview

The `pick_cube_noisy_ppo.py` script automatically exports comprehensive experiment results to multiple CSV files, providing detailed analysis capabilities for PPO robustness studies with noise injection.

## Generated CSV Files

### 1. `interim_results.csv`
- **Purpose**: Real-time progress tracking during experiments
- **Updated**: After each individual experiment completion
- **Content**: Same as final_results.csv but updated incrementally

### 2. `final_results.csv`
- **Purpose**: Complete raw experimental results  
- **Updated**: After all experiments complete
- **Content**: Raw data from all experiments with full configuration details

**Key Columns:**
- `config_name`: Experiment configuration identifier
- `description`: Human-readable experiment description
- `success_rate`: Final success rate (success_once metric)
- `duration`: Experiment duration in seconds
- `experiment_name`: Unique experiment identifier
- `status`: success/failed/error
- `timestamp`: Experiment completion time
- `total_timesteps`: Training timesteps used
- `num_envs`: Number of parallel training environments
- `num_eval_envs`: Number of evaluation environments
- `eval_freq`: Evaluation frequency in iterations
- `obs_noise_type`: Observation noise type (none/gaussian/uniform)
- `obs_noise_std`: Observation noise standard deviation
- `reward_noise_type`: Reward noise type (none/gaussian/uniform)
- `reward_noise_std`: Reward noise standard deviation
- `action_noise_type`: Action noise type (none/gaussian/uniform)
- `action_noise_std`: Action noise standard deviation
- `pos_noise_std`: Position-specific noise standard deviation
- `noise_growth_rate`: Curriculum learning growth rate
- `min_noise_factor`: Minimum noise scaling factor
- `max_noise_factor`: Maximum noise scaling factor
- `error`: Error message if failed

### 3. `detailed_analysis.csv`
- **Purpose**: Enhanced analysis with derived metrics and categorizations
- **Updated**: After all experiments complete  
- **Content**: All raw data plus computed analysis metrics

**Additional Analysis Columns:**
- `performance_category`: Performance tier (Poor/Fair/Good/Excellent)
- `noise_level`: Noise intensity category (No Noise/Low Noise/Medium Noise/High Noise)
- `noise_combination`: Types of noise applied (none/obs/reward/action/obs+reward/etc.)
- `uses_curriculum`: Boolean indicating curriculum learning usage
- `relative_performance`: Performance relative to no-noise baseline
- `timesteps_per_second`: Training efficiency metric
- `success_rate_per_hour`: Success rate normalized by training time

**Performance Categories:**
- **Poor**: 0-30% success rate
- **Fair**: 30-60% success rate  
- **Good**: 60-90% success rate
- **Excellent**: 90-100% success rate

**Noise Level Categories:**
- **No Noise**: No noise injection
- **Low Noise**: Average noise std ≤ 0.01
- **Medium Noise**: Average noise std ≤ 0.1
- **High Noise**: Average noise std > 0.1

### 4. `summary_statistics.csv`
- **Purpose**: Aggregated statistics grouped by experiment characteristics
- **Updated**: After all experiments complete
- **Content**: Statistical summaries across different groupings

**Grouping Types:**
- `overall`: Statistics across all successful experiments
- `noise_level`: Grouped by noise intensity categories
- `noise_combination`: Grouped by types of noise applied
- `performance_category`: Grouped by performance tiers

**Statistical Metrics:**
- `success_rate_count`: Number of experiments in group
- `success_rate_mean`: Average success rate
- `success_rate_std`: Standard deviation of success rates
- `success_rate_min`: Minimum success rate
- `success_rate_max`: Maximum success rate
- `relative_performance_mean`: Average relative performance
- `relative_performance_std`: Standard deviation of relative performance
- `duration_mean`: Average experiment duration
- `duration_std`: Standard deviation of durations
- `timesteps_per_second_mean`: Average training efficiency
- `timesteps_per_second_std`: Standard deviation of training efficiency

## Usage Examples

### Loading and Analyzing Results

```python
import pandas as pd

# Load basic results
results = pd.read_csv("runs/pick_cube_noisy_robustness_study/final_results.csv")

# Load detailed analysis
analysis = pd.read_csv("runs/pick_cube_noisy_robustness_study/detailed_analysis.csv")

# Load summary statistics  
summary = pd.read_csv("runs/pick_cube_noisy_robustness_study/summary_statistics.csv")

# Filter successful experiments
successful = analysis[analysis["status"] == "success"]

# Compare performance by noise level
noise_comparison = successful.groupby("noise_level")["success_rate"].agg(["mean", "std"])
print(noise_comparison)

# Find best performing configurations
best_configs = successful.nlargest(5, "success_rate")[["config_name", "success_rate", "noise_level"]]
print(best_configs)
```

### Visualization Examples

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Success rate by noise level
plt.figure(figsize=(10, 6))
sns.boxplot(data=successful, x="noise_level", y="success_rate")
plt.title("Success Rate Distribution by Noise Level")
plt.show()

# Performance vs efficiency scatter
plt.figure(figsize=(10, 6))
sns.scatterplot(data=successful, x="timesteps_per_second", y="success_rate", 
                hue="noise_combination", size="relative_performance")
plt.title("Training Efficiency vs Success Rate")
plt.show()

# Curriculum learning comparison
curriculum_comparison = successful.groupby("uses_curriculum")["success_rate"].mean()
plt.figure(figsize=(8, 6))
curriculum_comparison.plot(kind="bar")
plt.title("Curriculum Learning Impact")
plt.show()
```

## File Locations

All CSV files are saved to the experiment output directory:
- Default: `runs/pick_cube_noisy_robustness_study/`
- Configurable via `base_output_dir` parameter

## Data Quality Notes

- Only successful experiments are included in aggregated statistics
- Failed experiments are recorded with error details in raw results
- Relative performance metrics are normalized to no-noise baseline
- All numerical values are rounded to 4 decimal places for consistency
- Timestamps are in local system time format: "YYYY-MM-DD HH:MM:SS"

## Research Applications

These CSV exports enable comprehensive analysis of:
- **Robustness Studies**: Impact of different noise types on PPO performance
- **Curriculum Learning**: Effectiveness of noise scheduling strategies  
- **Efficiency Analysis**: Training time vs performance trade-offs
- **Noise Tolerance**: Identification of optimal noise levels for sim-to-real transfer
- **Comparative Studies**: Performance across different noise combinations
- **Statistical Analysis**: Significance testing and confidence intervals

## Integration with Analysis Tools

The CSV format enables easy integration with:
- **Pandas**: Data manipulation and analysis
- **Matplotlib/Seaborn**: Visualization and plotting
- **Scikit-learn**: Statistical analysis and modeling
- **Jupyter Notebooks**: Interactive exploration
- **R**: Statistical computing and analysis
- **Excel**: Basic analysis and reporting
- **Tableau/PowerBI**: Business intelligence and dashboards 