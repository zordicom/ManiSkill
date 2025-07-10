#!/usr/bin/env python3
"""
PickCubeNoisy PPO Training and Evaluation Script

This script runs PPO training on PickCubeNoisy environments with varying noise levels
to study the robustness of reinforcement learning to observation, reward, and action noise
across different observation modalities.

Supported noise types:
- Gaussian noise: Normal distribution with specified standard deviation
- Uniform noise: Uniform distribution in range [-std, std] (clipped at std area)

Supported observation modes:
- State: Proprioceptive state observations (positions, velocities, poses)
- RGB: Color images from environment cameras
- RGB-Depth: Color + depth images from environment cameras
- RGB+Segmentation: Color + semantic segmentation from environment cameras

The script automatically saves comprehensive results to multiple CSV files:
- interim_results.csv: Updated after each experiment completion
- final_results.csv: Complete raw results with all experiment details
- detailed_analysis.csv: Enhanced analysis with performance categories and metrics
- summary_statistics.csv: Aggregated statistics grouped by noise characteristics

NOTE: This script shows live training output during training, then automatically evaluates
each trained model to get accurate success rates for CSV export and analysis.

Usage:
    python scripts/rl_tests/pick_cube_noisy_ppo.py [--3x]

Arguments:
    --3x    Include 3x noise magnitude configurations (doubles experiment count)
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Fix MKL threading conflicts for optimal performance
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["OMP_NUM_THREADS"] = "4"  # Reasonable thread count for training

import numpy as np
import pandas as pd


class PickCubeNoisyExperiment:
    """Manages PPO experiments with varying noise levels for PickCubeNoisy environment."""

    def __init__(
        self,
        base_output_dir: str = "runs/pick_cube_noisy_experiments",
        total_timesteps: int = 300000,
        num_envs: int = 1024,
        num_eval_envs: int = 16,
        eval_freq: int = 25,
        save_videos: bool = True,
        noise_mode: str = "standard",  # "standard", "3x", or "both"
    ):
        """
        Initialize the experiment manager.

        Args:
            base_output_dir: Base directory for all experiment outputs
            total_timesteps: Total training timesteps per experiment
            num_envs: Number of parallel training environments
            num_eval_envs: Number of parallel evaluation environments
            eval_freq: Evaluation frequency in iterations
            save_videos: Whether to save training and evaluation videos
            noise_mode: Which noise configurations to run ("standard", "3x", or "both")
        """
        self.base_output_dir = Path(base_output_dir)
        self.total_timesteps = total_timesteps
        self.num_envs = num_envs
        self.num_eval_envs = num_eval_envs
        self.eval_freq = eval_freq
        self.save_videos = save_videos
        self.noise_mode = noise_mode

        # Create output directory
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

        # Find the PPO runner script path
        script_dir = Path(__file__).parent
        self.ppo_script = script_dir / "pick_cube_noisy_ppo_runner.py"

        if not self.ppo_script.exists():
            raise FileNotFoundError(f"PPO runner script not found at {self.ppo_script}")

        print("🚀 PickCubeNoisy PPO Experiment Manager initialized")
        print(f"📁 Output directory: {self.base_output_dir}")
        print(f"🎯 Training: {total_timesteps:,} timesteps with {num_envs} envs")
        print(f"📊 Evaluation: {num_eval_envs} envs every {eval_freq} iterations")
        print(f"🎬 Videos: {'Enabled' if save_videos else 'Disabled'}")

    def scale_noise_config_3x(self, config: Dict) -> Dict:
        """
        Create a 3x scaled version of a noise configuration.

        Args:
            config: Original noise configuration

        Returns:
            3x scaled noise configuration
        """
        scaled_config = config.copy()

        # Scale noise standard deviations by 3x
        noise_std_params = [
            "obs_noise_std",
            "reward_noise_std",
            "action_noise_std",
            "pos_noise_std",
        ]
        for param in noise_std_params:
            if param in scaled_config and scaled_config[param] is not None:
                scaled_config[param] = scaled_config[param] * 3.0

        # Update name and description
        scaled_config["name"] = f"{config['name']}_3x"
        scaled_config["description"] = f"{config['description']} (3x magnitude)"

        return scaled_config

    def get_noise_configurations(self) -> List[Dict]:
        """
        Define noise configurations to test.

        Returns:
            List of noise configuration dictionaries
        """
        # Define base noise configurations (without observation mode)
        base_configs = [
            # Control: No noise
            {
                "name": "no_noise",
                "obs_noise_type": "none",
                "reward_noise_type": "none",
                "action_noise_type": "none",
                "description": "Control (no noise)",
            },
            # Light noise configurations - Gaussian
            {
                "name": "light_obs_noise",
                "obs_noise_type": "gaussian",
                "obs_noise_std": 0.005,
                "pos_noise_std": 0.002,
                "reward_noise_type": "none",
                "action_noise_type": "none",
                "description": "Light observation noise (Gaussian)",
            },
            {
                "name": "light_reward_noise",
                "obs_noise_type": "none",
                "reward_noise_type": "gaussian",
                "reward_noise_std": 0.05,
                "action_noise_type": "none",
                "description": "Light reward noise (Gaussian)",
            },
            # Light noise configurations - Uniform
            {
                "name": "light_obs_noise_uniform",
                "obs_noise_type": "uniform",
                "obs_noise_std": 0.005,
                "pos_noise_std": 0.002,
                "reward_noise_type": "none",
                "action_noise_type": "none",
                "description": "Light observation noise (Uniform)",
            },
            {
                "name": "light_reward_noise_uniform",
                "obs_noise_type": "none",
                "reward_noise_type": "uniform",
                "reward_noise_std": 0.05,
                "action_noise_type": "none",
                "description": "Light reward noise (Uniform)",
            },
            # Medium noise configurations - Gaussian
            {
                "name": "medium_obs_noise",
                "obs_noise_type": "gaussian",
                "obs_noise_std": 0.01,
                "pos_noise_std": 0.005,
                "reward_noise_type": "none",
                "action_noise_type": "none",
                "description": "Medium observation noise (Gaussian)",
            },
            {
                "name": "medium_reward_noise",
                "obs_noise_type": "none",
                "reward_noise_type": "gaussian",
                "reward_noise_std": 0.1,
                "action_noise_type": "none",
                "description": "Medium reward noise (Gaussian)",
            },
            {
                "name": "medium_combined_noise",
                "obs_noise_type": "gaussian",
                "obs_noise_std": 0.01,
                "pos_noise_std": 0.005,
                "reward_noise_type": "gaussian",
                "reward_noise_std": 0.1,
                "action_noise_type": "none",
                "description": "Medium obs + reward noise (Gaussian)",
            },
            # Medium noise configurations - Uniform
            {
                "name": "medium_obs_noise_uniform",
                "obs_noise_type": "uniform",
                "obs_noise_std": 0.01,
                "pos_noise_std": 0.005,
                "reward_noise_type": "none",
                "action_noise_type": "none",
                "description": "Medium observation noise (Uniform)",
            },
            {
                "name": "medium_reward_noise_uniform",
                "obs_noise_type": "none",
                "reward_noise_type": "uniform",
                "reward_noise_std": 0.1,
                "action_noise_type": "none",
                "description": "Medium reward noise (Uniform)",
            },
            {
                "name": "medium_combined_noise_uniform",
                "obs_noise_type": "uniform",
                "obs_noise_std": 0.01,
                "pos_noise_std": 0.005,
                "reward_noise_type": "uniform",
                "reward_noise_std": 0.1,
                "action_noise_type": "none",
                "description": "Medium obs + reward noise (Uniform)",
            },
            # Heavy noise configurations - Gaussian
            {
                "name": "heavy_obs_noise",
                "obs_noise_type": "gaussian",
                "obs_noise_std": 0.02,
                "pos_noise_std": 0.01,
                "reward_noise_type": "none",
                "action_noise_type": "none",
                "description": "Heavy observation noise (Gaussian)",
            },
            {
                "name": "heavy_reward_noise",
                "obs_noise_type": "none",
                "reward_noise_type": "gaussian",
                "reward_noise_std": 0.2,
                "action_noise_type": "none",
                "description": "Heavy reward noise (Gaussian)",
            },
            {
                "name": "heavy_combined_noise",
                "obs_noise_type": "gaussian",
                "obs_noise_std": 0.02,
                "pos_noise_std": 0.01,
                "reward_noise_type": "gaussian",
                "reward_noise_std": 0.2,
                "action_noise_type": "none",
                "description": "Heavy obs + reward noise (Gaussian)",
            },
            # Heavy noise configurations - Uniform
            {
                "name": "heavy_obs_noise_uniform",
                "obs_noise_type": "uniform",
                "obs_noise_std": 0.02,
                "pos_noise_std": 0.01,
                "reward_noise_type": "none",
                "action_noise_type": "none",
                "description": "Heavy observation noise (Uniform)",
            },
            {
                "name": "heavy_reward_noise_uniform",
                "obs_noise_type": "none",
                "reward_noise_type": "uniform",
                "reward_noise_std": 0.2,
                "action_noise_type": "none",
                "description": "Heavy reward noise (Uniform)",
            },
            {
                "name": "heavy_combined_noise_uniform",
                "obs_noise_type": "uniform",
                "obs_noise_std": 0.02,
                "pos_noise_std": 0.01,
                "reward_noise_type": "uniform",
                "reward_noise_std": 0.2,
                "action_noise_type": "none",
                "description": "Heavy obs + reward noise (Uniform)",
            },
            # Action noise configurations - Gaussian
            {
                "name": "medium_action_noise",
                "obs_noise_type": "none",
                "reward_noise_type": "none",
                "action_noise_type": "gaussian",
                "action_noise_std": 0.05,
                "description": "Medium action noise (Gaussian)",
            },
            {
                "name": "all_medium_noise",
                "obs_noise_type": "gaussian",
                "obs_noise_std": 0.01,
                "pos_noise_std": 0.005,
                "reward_noise_type": "gaussian",
                "reward_noise_std": 0.1,
                "action_noise_type": "gaussian",
                "action_noise_std": 0.05,
                "description": "All types medium noise (Gaussian)",
            },
            # Action noise configurations - Uniform
            {
                "name": "medium_action_noise_uniform",
                "obs_noise_type": "none",
                "reward_noise_type": "none",
                "action_noise_type": "uniform",
                "action_noise_std": 0.05,
                "description": "Medium action noise (Uniform)",
            },
            {
                "name": "all_medium_noise_uniform",
                "obs_noise_type": "uniform",
                "obs_noise_std": 0.01,
                "pos_noise_std": 0.005,
                "reward_noise_type": "uniform",
                "reward_noise_std": 0.1,
                "action_noise_type": "uniform",
                "action_noise_std": 0.05,
                "description": "All types medium noise (Uniform)",
            },
            # Curriculum learning - Gaussian
            {
                "name": "curriculum_noise",
                "obs_noise_type": "gaussian",
                "obs_noise_std": 0.02,
                "pos_noise_std": 0.01,
                "reward_noise_type": "gaussian",
                "reward_noise_std": 0.15,
                "noise_growth_rate": 0.02,
                "min_noise_factor": 0.1,
                "max_noise_factor": 1.0,
                "description": "Curriculum learning (growing Gaussian noise)",
            },
            # Curriculum learning - Uniform
            {
                "name": "curriculum_noise_uniform",
                "obs_noise_type": "uniform",
                "obs_noise_std": 0.02,
                "pos_noise_std": 0.01,
                "reward_noise_type": "uniform",
                "reward_noise_std": 0.15,
                "noise_growth_rate": 0.02,
                "min_noise_factor": 0.1,
                "max_noise_factor": 1.0,
                "description": "Curriculum learning (growing Uniform noise)",
            },
        ]

        # Define observation modes to test
        observation_modes = [
            ("state", "State"),
            ("rgb", "RGB"),
            ("rgbd", "RGB-Depth"),
            ("rgb+segmentation", "RGB+Segmentation"),
        ]

        # Generate configurations for each observation mode
        all_configs = []
        for obs_mode, obs_display_name in observation_modes:
            for base_config in base_configs:
                config = base_config.copy()
                config["name"] = f"{base_config['name']}_{obs_mode}"
                config["obs_mode"] = obs_mode
                config["description"] = (
                    f"{base_config['description']} ({obs_display_name})"
                )
                all_configs.append(config)

        if self.noise_mode == "3x":
            print("🔄 Using only 3x noise scale configurations...")
            # Create 3x versions of all configurations (replace originals)
            scaled_configs = []
            for config in all_configs:
                scaled_config = self.scale_noise_config_3x(config)
                scaled_configs.append(scaled_config)
            all_configs = scaled_configs
        elif self.noise_mode == "both":
            print("🔄 Adding 3x noise scale configurations to standard ones...")
            # Create 3x versions of all configurations (in addition to originals)
            scaled_configs = []
            for config in all_configs:
                scaled_config = self.scale_noise_config_3x(config)
                scaled_configs.append(scaled_config)
            # Add 3x configurations to the original ones
            all_configs.extend(scaled_configs)
        else:
            print("📊 Using standard noise scale configurations...")

        return all_configs

    def build_command(self, config: Dict, exp_name: str) -> List[str]:
        """
        Build PPO training command for a given noise configuration.

        Args:
            config: Noise configuration dictionary
            exp_name: Experiment name

        Returns:
            Command as list of strings
        """
        cmd = [
            sys.executable,
            str(self.ppo_script),
            "--env-id",
            "PickCubeNoisy-v1",
            "--exp-name",
            exp_name,
            "--total-timesteps",
            str(self.total_timesteps),
            "--num-envs",
            str(self.num_envs),
            "--num-eval-envs",
            str(self.num_eval_envs),
            "--eval-freq",
            str(self.eval_freq),
            "--obs-mode",
            config.get("obs_mode", "state"),
            "--control-mode",
            "pd_joint_delta_pos",
            "--partial-reset",  # This is True by default, so just use the flag
            "--cuda",  # This is True by default, so just use the flag
            "--save-model",  # This is True by default, so just use the flag
            "--track",  # Enable wandb tracking
            "--wandb-project-name",
            "PickCubeNoisy-Robustness",
            "--wandb-group",
            "PPO-Noise-Study",
        ]

        # Add video recording flags
        if self.save_videos:
            # Only enable evaluation videos to avoid training video resolution conflicts
            # Training videos can cause FFmpeg issues with large frame sizes
            cmd.extend(["--capture-video"])
            # Don't pass --save-train-video-freq at all to avoid ZeroDivisionError
            print(
                "📹 Evaluation videos enabled (training videos disabled to avoid FFmpeg issues)"
            )
            print(f"📁 Evaluation videos location: runs/{exp_name}/videos/")
        else:
            cmd.extend(["--no-capture-video"])

        # Add noise-specific parameters using the new format
        for key, value in config.items():
            if key in ["name", "description"]:
                continue

            # Convert parameter names to noise format for the runner script
            param_name = key.replace("_", "-")
            cmd.append(f"--noise-{param_name}={value}")

        return cmd

    def evaluate_trained_model(self, exp_name: str, config: Dict) -> float:
        """
        Evaluate a trained model to get the actual success rate.

        Args:
            exp_name: Experiment name (used to find the checkpoint)
            config: Noise configuration (to apply same noise during eval)

        Returns:
            Success rate from evaluation
        """
        # Find the checkpoint file - PPO saves to runs/{exp_name}/
        checkpoint_dir = Path("runs") / exp_name

        # Look for final checkpoint first, then fall back to other .pt files
        final_checkpoint = checkpoint_dir / "final_ckpt.pt"
        if final_checkpoint.exists():
            checkpoint_file = final_checkpoint
        else:
            # Look for other .pt checkpoint files
            checkpoint_files = list(checkpoint_dir.glob("ckpt_*.pt"))
            if not checkpoint_files:
                print(f"⚠️  No checkpoint found in {checkpoint_dir}")
                return 0.0
            # Use the most recent checkpoint
            checkpoint_file = max(checkpoint_files, key=lambda f: f.stat().st_mtime)
        print(f"📁 Using checkpoint: {checkpoint_file}")

        # Create a simple evaluation script that registers our environment and uses ppo_fast.py
        eval_script_content = f'''
import sys
from pathlib import Path

# Add ManiSkill root to path
maniskill_root = Path("{Path(__file__).parent.parent.parent}")
sys.path.insert(0, str(maniskill_root))

# Import and register the PickCubeNoisy environment
import mani_skill.envs.tasks.tabletop.pick_cube_noisy

# Set noise parameters as environment variables
import os
noise_config = {config}
for key, value in noise_config.items():
    if key not in ["name", "description"]:
        env_var_name = f"NOISE_{{key.upper()}}"
        os.environ[env_var_name] = str(value)

# Import and run the ppo_fast.py script
sys.argv = [
    "ppo_fast.py",
    "--env-id", "PickCubeNoisy-v1",
    "--exp-name", "{exp_name}_eval",
    "--evaluate",
    "--checkpoint", "{checkpoint_file}",
    "--num-eval-envs", "{self.num_eval_envs}",
    "--num-eval-steps", "{self.num_eval_envs * 10}",
    "--obs-mode", "{config.get("obs_mode", "state")}",
    "--control-mode", "pd_joint_delta_pos",
         "--cuda",
     "--capture-video",  # Enable evaluation videos
     "--no-track",
     "--no-save-model",
]

# Import the ppo_fast module and run it
ppo_fast_path = maniskill_root / "examples/baselines/ppo/ppo_fast.py"
with open(ppo_fast_path, "r") as f:
    ppo_code = f.read()

# Patch the environment creation to handle noise parameters
noise_patch = """
# Add noise parameters from environment variables for PickCubeNoisy
if "PickCubeNoisy" in args.env_id:
    noise_params = {{}}
    for key, value in os.environ.items():
        if key.startswith("NOISE_"):
            param_name = key[6:].lower()  # Remove "NOISE_" prefix
            # Convert string values to appropriate types
            if value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
            elif value.lower() == "none":
                value = "none"
            else:
                try:
                    # Try to convert to float
                    value = float(value)
                except ValueError:
                    # Keep as string
                    pass
            noise_params[param_name] = value

    if noise_params:
        env_kwargs.update(noise_params)
        print(f"Applied noise parameters: {{noise_params}}")
"""

# Insert the patch after the env_kwargs definition
insert_point = 'if args.robot_uids is not None:\\n        env_kwargs["robot_uids"] = args.robot_uids'
if insert_point in ppo_code:
    ppo_code = ppo_code.replace(insert_point, insert_point + "\\n    " + noise_patch)

# Execute the patched ppo_fast.py code
exec(ppo_code)
'''

        # Write evaluation script to temporary file
        eval_script_path = Path("temp_eval_script.py")
        eval_script_path.write_text(eval_script_content)

        print("🚀 Running evaluation with registered PickCubeNoisy environment")
        print(
            f"📹 Evaluation videos will be saved to: runs/{exp_name}_eval/test_videos/"
        )

        try:
            # Set up environment to avoid MKL threading conflicts
            eval_env = os.environ.copy()
            eval_env["MKL_THREADING_LAYER"] = "GNU"
            eval_env["OMP_NUM_THREADS"] = "2"  # Conservative for evaluation subprocess

            # Run evaluation script
            result = subprocess.run(
                [sys.executable, str(eval_script_path)],
                capture_output=True,
                text=True,
                env=eval_env,
                cwd=Path(__file__).parent.parent.parent,  # ManiSkill root
                timeout=300,  # 5 minute timeout for evaluation
                check=False,
            )

            if result.returncode == 0:
                # Parse success rate from ppo_fast.py evaluation output
                success_rate = self.parse_success_rate(result.stdout)
                return success_rate
            else:
                print(f"⚠️  Evaluation failed: {result.stderr}")
                return 0.0

        except subprocess.TimeoutExpired:
            print("⏰ Evaluation timed out")
            return 0.0
        except Exception as e:
            print(f"❌ Evaluation error: {e}")
            return 0.0
        finally:
            # Clean up temporary script
            if eval_script_path.exists():
                eval_script_path.unlink()

    def run_experiment(self, config: Dict) -> Tuple[bool, Dict]:
        """
        Run a single PPO experiment with the given noise configuration.

        Args:
            config: Noise configuration dictionary

        Returns:
            Tuple of (success, results_dict)
        """
        exp_name = f"pick_cube_noisy_{config['name']}_{int(time.time())}"

        print(f"\n🔬 Running experiment: {config['description']}")
        print(f"📝 Experiment name: {exp_name}")

        # Build and execute command
        cmd = self.build_command(config, exp_name)

        print(f"🚀 Command: {' '.join(cmd)}")
        print("\n" + "=" * 60)
        print("📺 TRAINING OUTPUT (live):")
        print("=" * 60)

        try:
            start_time = time.time()
            result = subprocess.run(
                cmd,
                cwd=Path(__file__).parent.parent.parent,  # ManiSkill root
                check=False,
            )
            duration = time.time() - start_time

            print("\n" + "=" * 60)
            print("📋 EXPERIMENT RESULTS:")
            print("=" * 60)

            if result.returncode == 0:
                print(f"✅ Experiment completed successfully in {duration:.1f}s")

                # Run evaluation with the trained model to get actual success rate
                print("🧪 Running post-training evaluation...")
                success_rate = self.evaluate_trained_model(exp_name, config)
                print(f"📊 Evaluated success rate: {success_rate:.3f}")

                results = {
                    "config_name": config["name"],
                    "description": config["description"],
                    "success_rate": success_rate,
                    "duration": duration,
                    "experiment_name": exp_name,
                    "status": "success",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "total_timesteps": self.total_timesteps,
                    "num_envs": self.num_envs,
                    "num_eval_envs": self.num_eval_envs,
                    "eval_freq": self.eval_freq,
                    **{
                        k: v
                        for k, v in config.items()
                        if k not in ["name", "description"]
                    },
                }

                return True, results
            else:
                print(f"❌ Experiment failed with return code {result.returncode}")
                print("💡 Check the training output above for error details")

                results = {
                    "config_name": config["name"],
                    "description": config["description"],
                    "success_rate": 0.0,
                    "duration": duration,
                    "experiment_name": exp_name,
                    "status": "failed",
                    "error": f"Process failed with return code {result.returncode}",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "total_timesteps": self.total_timesteps,
                    "num_envs": self.num_envs,
                    "num_eval_envs": self.num_eval_envs,
                    "eval_freq": self.eval_freq,
                    **{
                        k: v
                        for k, v in config.items()
                        if k not in ["name", "description"]
                    },
                }

                return False, results

        except Exception as e:
            print(f"❌ Experiment failed with exception: {e}")
            results = {
                "config_name": config["name"],
                "description": config["description"],
                "success_rate": 0.0,
                "duration": 0.0,
                "experiment_name": exp_name,
                "status": "error",
                "error": str(e),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_timesteps": self.total_timesteps,
                "num_envs": self.num_envs,
                "num_eval_envs": self.num_eval_envs,
                "eval_freq": self.eval_freq,
                **{k: v for k, v in config.items() if k not in ["name", "description"]},
            }
            return False, results

    def parse_success_rate(self, output: str) -> float:
        """
        Parse success rate from training output.

        Args:
            output: Training stdout output

        Returns:
            Final success rate (success_once metric)
        """
        lines = output.split("\n")
        success_rates = []

        for line in lines:
            if "success_once:" in line:
                try:
                    # Extract success rate from line like "success_once: 0.85, return: 2.34"
                    parts = line.split("success_once:")
                    if len(parts) > 1:
                        rate_str = parts[1].split(",")[0].strip()
                        success_rates.append(float(rate_str))
                except (ValueError, IndexError):
                    continue

        # Return the last (final) success rate, or 0.0 if none found
        return success_rates[-1] if success_rates else 0.0

    def run_all_experiments(self) -> pd.DataFrame:
        """
        Run all noise experiments and collect results.

        Returns:
            DataFrame with experiment results
        """
        configs = self.get_noise_configurations()
        all_results = []

        print(f"\n🎯 Starting {len(configs)} PickCubeNoisy PPO experiments")
        print("=" * 80)

        for i, config in enumerate(configs, 1):
            print(f"\n📊 Experiment {i}/{len(configs)}")

            success, results = self.run_experiment(config)
            all_results.append(results)

            # Print interim results
            if success:
                print(f"🎉 Final success rate: {results['success_rate']:.3f}")
            else:
                print("💥 Experiment failed")

            # Save interim results
            df = pd.DataFrame(all_results)
            results_file = self.base_output_dir / "interim_results.csv"
            df.to_csv(results_file, index=False)
            print(f"💾 Interim results saved to {results_file}")

        # Save final results
        final_df = pd.DataFrame(all_results)
        final_file = self.base_output_dir / "final_results.csv"
        final_df.to_csv(final_file, index=False)

        # Create detailed analysis CSV
        analysis_df = self.create_analysis_dataframe(final_df)
        analysis_file = self.base_output_dir / "detailed_analysis.csv"
        analysis_df.to_csv(analysis_file, index=False)

        # Create summary statistics CSV
        summary_df = self.create_summary_statistics(analysis_df)
        summary_file = self.base_output_dir / "summary_statistics.csv"
        summary_df.to_csv(summary_file, index=False)

        print("\n🏁 All experiments completed!")
        print(f"📊 Final results saved to {final_file}")
        print(f"📈 Detailed analysis saved to {analysis_file}")
        print(f"📈 Summary statistics saved to {summary_file}")

        return final_df

    def create_analysis_dataframe(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a detailed analysis DataFrame with additional metrics and insights.

        Args:
            results_df: DataFrame with raw experiment results

        Returns:
            Enhanced DataFrame with analysis metrics
        """
        analysis_df = results_df.copy()

        # Add observation mode category
        analysis_df["observation_mode"] = analysis_df["obs_mode"].fillna("state")

        # Add noise magnitude category
        analysis_df["noise_magnitude"] = analysis_df["config_name"].apply(
            lambda x: "3x" if "_3x_" in x else "standard"
        )

        # Add performance categories
        analysis_df["performance_category"] = pd.cut(
            analysis_df["success_rate"],
            bins=[0, 0.3, 0.6, 0.9, 1.0],
            labels=["Poor", "Fair", "Good", "Excellent"],
            include_lowest=True,
        )

        # Add noise level categories
        def categorize_noise_level(row):
            if (
                row.get("obs_noise_type", "none") == "none"
                and row.get("reward_noise_type", "none") == "none"
                and row.get("action_noise_type", "none") == "none"
            ):
                return "No Noise"

            # Calculate average noise level (works for both Gaussian and uniform)
            noise_levels = []
            if row.get("obs_noise_std", 0) > 0:
                noise_levels.append(row["obs_noise_std"])
            if row.get("reward_noise_std", 0) > 0:
                noise_levels.append(row["reward_noise_std"])
            if row.get("action_noise_std", 0) > 0:
                noise_levels.append(row["action_noise_std"])

            if not noise_levels:
                return "No Noise"

            avg_noise = sum(noise_levels) / len(noise_levels)
            if avg_noise <= 0.01:
                return "Low Noise"
            elif avg_noise <= 0.1:
                return "Medium Noise"
            else:
                return "High Noise"

        analysis_df["noise_level"] = analysis_df.apply(categorize_noise_level, axis=1)

        # Add efficiency metrics
        analysis_df["timesteps_per_second"] = (
            analysis_df["total_timesteps"] / analysis_df["duration"]
        )
        analysis_df["success_rate_per_hour"] = analysis_df["success_rate"] / (
            analysis_df["duration"] / 3600
        )

        # Add noise type combinations
        def get_noise_combination(row):
            types = []
            if row.get("obs_noise_type", "none") != "none":
                obs_type = row.get("obs_noise_type", "none")
                types.append(f"obs_{obs_type}")
            if row.get("reward_noise_type", "none") != "none":
                reward_type = row.get("reward_noise_type", "none")
                types.append(f"reward_{reward_type}")
            if row.get("action_noise_type", "none") != "none":
                action_type = row.get("action_noise_type", "none")
                types.append(f"action_{action_type}")

            if not types:
                return "none"
            return "+".join(types)

        analysis_df["noise_combination"] = analysis_df.apply(
            get_noise_combination, axis=1
        )

        # Add curriculum learning indicator
        analysis_df["uses_curriculum"] = analysis_df["noise_growth_rate"] > 0

        # Add relative performance (normalized to no-noise baseline)
        no_noise_success = analysis_df[analysis_df["noise_level"] == "No Noise"][
            "success_rate"
        ].mean()
        if no_noise_success > 0:
            analysis_df["relative_performance"] = (
                analysis_df["success_rate"] / no_noise_success
            )
        else:
            analysis_df["relative_performance"] = 0

        # Reorder columns for better readability
        column_order = [
            "config_name",
            "description",
            "observation_mode",
            "noise_magnitude",
            "success_rate",
            "performance_category",
            "noise_level",
            "noise_combination",
            "uses_curriculum",
            "relative_performance",
            "duration",
            "timesteps_per_second",
            "success_rate_per_hour",
            "status",
            "timestamp",
            "experiment_name",
            "total_timesteps",
            "num_envs",
            "num_eval_envs",
            "eval_freq",
            "obs_mode",
            "obs_noise_type",
            "obs_noise_std",
            "reward_noise_type",
            "reward_noise_std",
            "action_noise_type",
            "action_noise_std",
            "pos_noise_std",
            "noise_growth_rate",
            "min_noise_factor",
            "max_noise_factor",
            "error",
        ]

        # Only include columns that exist in the dataframe
        existing_columns = [col for col in column_order if col in analysis_df.columns]
        remaining_columns = [
            col for col in analysis_df.columns if col not in existing_columns
        ]
        final_column_order = existing_columns + remaining_columns

        return analysis_df[final_column_order]

    def create_summary_statistics(self, analysis_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create summary statistics grouped by noise characteristics.

        Args:
            analysis_df: DataFrame with analysis results

        Returns:
            DataFrame with summary statistics
        """
        successful_experiments = analysis_df[analysis_df["status"] == "success"]

        if len(successful_experiments) == 0:
            return pd.DataFrame({"message": ["No successful experiments to analyze"]})

        # Group by noise level
        noise_level_stats = (
            successful_experiments.groupby("noise_level")
            .agg({
                "success_rate": ["mean", "std", "min", "max", "count"],
                "duration": ["mean", "std"],
                "timesteps_per_second": ["mean", "std"],
                "relative_performance": ["mean", "std"],
            })
            .round(4)
        )

        # Flatten column names
        noise_level_stats.columns = [
            "_".join(col).strip() for col in noise_level_stats.columns
        ]
        noise_level_stats = noise_level_stats.reset_index()
        noise_level_stats["group_type"] = "noise_level"

        # Group by noise combination
        noise_combo_stats = (
            successful_experiments.groupby("noise_combination")
            .agg({
                "success_rate": ["mean", "std", "min", "max", "count"],
                "duration": ["mean", "std"],
                "timesteps_per_second": ["mean", "std"],
                "relative_performance": ["mean", "std"],
            })
            .round(4)
        )

        # Flatten column names
        noise_combo_stats.columns = [
            "_".join(col).strip() for col in noise_combo_stats.columns
        ]
        noise_combo_stats = noise_combo_stats.reset_index()
        noise_combo_stats["group_type"] = "noise_combination"
        noise_combo_stats.rename(
            columns={"noise_combination": "noise_level"}, inplace=True
        )

        # Group by performance category
        performance_stats = (
            successful_experiments.groupby("performance_category", observed=False)
            .agg({
                "success_rate": ["mean", "std", "min", "max", "count"],
                "duration": ["mean", "std"],
                "timesteps_per_second": ["mean", "std"],
                "relative_performance": ["mean", "std"],
            })
            .round(4)
        )

        # Flatten column names
        performance_stats.columns = [
            "_".join(col).strip() for col in performance_stats.columns
        ]
        performance_stats = performance_stats.reset_index()
        performance_stats["group_type"] = "performance_category"
        performance_stats.rename(
            columns={"performance_category": "noise_level"}, inplace=True
        )

        # Group by observation mode
        obs_mode_stats = (
            successful_experiments.groupby("obs_mode")
            .agg({
                "success_rate": ["mean", "std", "min", "max", "count"],
                "duration": ["mean", "std"],
                "timesteps_per_second": ["mean", "std"],
                "relative_performance": ["mean", "std"],
            })
            .round(4)
        )

        # Flatten column names
        obs_mode_stats.columns = [
            "_".join(col).strip() for col in obs_mode_stats.columns
        ]
        obs_mode_stats = obs_mode_stats.reset_index()
        obs_mode_stats["group_type"] = "observation_mode"
        obs_mode_stats.rename(columns={"obs_mode": "noise_level"}, inplace=True)

        # Group by noise magnitude
        noise_magnitude_stats = (
            successful_experiments.groupby("noise_magnitude")
            .agg({
                "success_rate": ["mean", "std", "min", "max", "count"],
                "duration": ["mean", "std"],
                "timesteps_per_second": ["mean", "std"],
                "relative_performance": ["mean", "std"],
            })
            .round(4)
        )

        # Flatten column names
        noise_magnitude_stats.columns = [
            "_".join(col).strip() for col in noise_magnitude_stats.columns
        ]
        noise_magnitude_stats = noise_magnitude_stats.reset_index()
        noise_magnitude_stats["group_type"] = "noise_magnitude"
        noise_magnitude_stats.rename(
            columns={"noise_magnitude": "noise_level"}, inplace=True
        )

        # Combine all statistics
        combined_stats = pd.concat(
            [
                noise_level_stats,
                noise_combo_stats,
                performance_stats,
                obs_mode_stats,
                noise_magnitude_stats,
            ],
            ignore_index=True,
        )

        # Add overall statistics
        overall_stats = pd.DataFrame({
            "noise_level": ["OVERALL"],
            "group_type": ["overall"],
            "success_rate_mean": [successful_experiments["success_rate"].mean()],
            "success_rate_std": [successful_experiments["success_rate"].std()],
            "success_rate_min": [successful_experiments["success_rate"].min()],
            "success_rate_max": [successful_experiments["success_rate"].max()],
            "success_rate_count": [len(successful_experiments)],
            "duration_mean": [successful_experiments["duration"].mean()],
            "duration_std": [successful_experiments["duration"].std()],
            "timesteps_per_second_mean": [
                successful_experiments["timesteps_per_second"].mean()
            ],
            "timesteps_per_second_std": [
                successful_experiments["timesteps_per_second"].std()
            ],
            "relative_performance_mean": [
                successful_experiments["relative_performance"].mean()
            ],
            "relative_performance_std": [
                successful_experiments["relative_performance"].std()
            ],
        }).round(4)

        final_stats = pd.concat([overall_stats, combined_stats], ignore_index=True)

        # Reorder columns for better readability
        column_order = [
            "group_type",
            "noise_level",
            "success_rate_count",
            "success_rate_mean",
            "success_rate_std",
            "success_rate_min",
            "success_rate_max",
            "relative_performance_mean",
            "relative_performance_std",
            "duration_mean",
            "duration_std",
            "timesteps_per_second_mean",
            "timesteps_per_second_std",
        ]

        return final_stats[column_order]

    def print_summary(self, results_df: pd.DataFrame):
        """
        Print a summary of experiment results.

        Args:
            results_df: DataFrame with experiment results
        """
        print("\n" + "=" * 80)
        print("📈 EXPERIMENT SUMMARY")
        print("=" * 80)

        # Sort by success rate
        sorted_df = results_df.sort_values("success_rate", ascending=False)

        print(
            f"{'Rank':<4} {'Config Name':<20} {'Success Rate':<12} {'Description':<30}"
        )
        print("-" * 80)

        for i, (_, row) in enumerate(sorted_df.iterrows(), 1):
            status_emoji = "✅" if row["status"] == "success" else "❌"
            print(
                f"{i:<4} {row['config_name']:<20} {row['success_rate']:<12.3f} {status_emoji} {row['description']}"
            )

        print("\n📊 Statistics:")
        successful_exps = results_df[results_df["status"] == "success"]
        if len(successful_exps) > 0:
            print(f"  • Best success rate: {successful_exps['success_rate'].max():.3f}")
            print(
                f"  • Worst success rate: {successful_exps['success_rate'].min():.3f}"
            )
            print(
                f"  • Average success rate: {successful_exps['success_rate'].mean():.3f}"
            )
            print(
                f"  • Successful experiments: {len(successful_exps)}/{len(results_df)}"
            )
        else:
            print("  • No successful experiments")

        print("\n💡 Key Insights:")

        # Find best and worst performing configs
        if len(successful_exps) > 0:
            best_config = successful_exps.loc[successful_exps["success_rate"].idxmax()]
            worst_config = successful_exps.loc[successful_exps["success_rate"].idxmin()]

            print(
                f"  • Best performing: {best_config['config_name']} ({best_config['success_rate']:.3f})"
            )
            print(
                f"  • Worst performing: {worst_config['config_name']} ({worst_config['success_rate']:.3f})"
            )

            # Compare noise types
            obs_gaussian_configs = successful_exps[
                successful_exps["obs_noise_type"] == "gaussian"
            ]
            obs_uniform_configs = successful_exps[
                successful_exps["obs_noise_type"] == "uniform"
            ]
            reward_gaussian_configs = successful_exps[
                successful_exps["reward_noise_type"] == "gaussian"
            ]
            reward_uniform_configs = successful_exps[
                successful_exps["reward_noise_type"] == "uniform"
            ]
            action_gaussian_configs = successful_exps[
                successful_exps["action_noise_type"] == "gaussian"
            ]
            action_uniform_configs = successful_exps[
                successful_exps["action_noise_type"] == "uniform"
            ]

            if len(obs_gaussian_configs) > 0:
                print(
                    f"  • Obs Gaussian noise avg success: {obs_gaussian_configs['success_rate'].mean():.3f}"
                )
            if len(obs_uniform_configs) > 0:
                print(
                    f"  • Obs Uniform noise avg success: {obs_uniform_configs['success_rate'].mean():.3f}"
                )
            if len(reward_gaussian_configs) > 0:
                print(
                    f"  • Reward Gaussian noise avg success: {reward_gaussian_configs['success_rate'].mean():.3f}"
                )
            if len(reward_uniform_configs) > 0:
                print(
                    f"  • Reward Uniform noise avg success: {reward_uniform_configs['success_rate'].mean():.3f}"
                )
            if len(action_gaussian_configs) > 0:
                print(
                    f"  • Action Gaussian noise avg success: {action_gaussian_configs['success_rate'].mean():.3f}"
                )
            if len(action_uniform_configs) > 0:
                print(
                    f"  • Action Uniform noise avg success: {action_uniform_configs['success_rate'].mean():.3f}"
                )

            # Compare observation modes
            state_configs = successful_exps[successful_exps["obs_mode"] == "state"]
            rgb_configs = successful_exps[successful_exps["obs_mode"] == "rgb"]
            rgbd_configs = successful_exps[successful_exps["obs_mode"] == "rgbd"]
            rgb_seg_configs = successful_exps[
                successful_exps["obs_mode"] == "rgb+segmentation"
            ]

            if len(state_configs) > 0:
                print(
                    f"  • State observations avg success: {state_configs['success_rate'].mean():.3f}"
                )
            if len(rgb_configs) > 0:
                print(
                    f"  • RGB observations avg success: {rgb_configs['success_rate'].mean():.3f}"
                )
            if len(rgbd_configs) > 0:
                print(
                    f"  • RGB-Depth observations avg success: {rgbd_configs['success_rate'].mean():.3f}"
                )
            if len(rgb_seg_configs) > 0:
                print(
                    f"  • RGB+Segmentation observations avg success: {rgb_seg_configs['success_rate'].mean():.3f}"
                )

            # Compare noise magnitudes
            standard_configs = successful_exps[
                successful_exps["noise_magnitude"] == "standard"
            ]
            scaled_3x_configs = successful_exps[
                successful_exps["noise_magnitude"] == "3x"
            ]

            if len(standard_configs) > 0:
                print(
                    f"  • Standard noise magnitude avg success: {standard_configs['success_rate'].mean():.3f}"
                )
            if len(scaled_3x_configs) > 0:
                print(
                    f"  • 3x noise magnitude avg success: {scaled_3x_configs['success_rate'].mean():.3f}"
                )
                if len(standard_configs) > 0:
                    performance_ratio = (
                        scaled_3x_configs["success_rate"].mean()
                        / standard_configs["success_rate"].mean()
                    )
                    print(
                        f"  • 3x vs standard performance ratio: {performance_ratio:.3f}"
                    )


def main():
    """Main function to run all experiments."""
    print("🎮 PickCubeNoisy PPO Robustness Study")
    print("=" * 50)

    # Initialize experiment manager
    parser = argparse.ArgumentParser(
        description="Run PickCubeNoisy PPO robustness study."
    )
    parser.add_argument(
        "--3x", action="store_true", help="Run only 3x noise magnitude configurations"
    )
    parser.add_argument(
        "--both",
        action="store_true",
        help="Run both standard and 3x noise configurations",
    )
    args = parser.parse_args()

    # Determine noise mode
    if getattr(args, "3x"):
        noise_mode = "3x"
    elif args.both:
        noise_mode = "both"
    else:
        noise_mode = "standard"

    experiment = PickCubeNoisyExperiment(
        base_output_dir="runs/pick_cube_noisy_robustness_study",
        total_timesteps=30_000_000,
        num_envs=2048,
        num_eval_envs=16,
        eval_freq=25,
        save_videos=True,
        noise_mode=noise_mode,
    )

    # Show total number of experiments
    total_experiments = len(experiment.get_noise_configurations())
    base_experiments = 92  # 23 base noise configs × 4 observation modes

    print(f"🔬 Total experiments planned: {total_experiments}")
    if noise_mode == "3x":
        print(f"📈 3x noise scaling only: {base_experiments} 3x scaled configurations")
    elif noise_mode == "both":
        print(
            f"📈 Both noise levels: {base_experiments} standard + {base_experiments} 3x scaled"
        )
    else:
        print("📊 Standard noise levels only")
    print("🎯 Noise types: Gaussian and Uniform configurations")
    print("📊 Includes light, medium, and heavy noise levels")
    print("🔄 Plus curriculum learning variants")
    print("👁️  Observation modes: State, RGB, RGB-Depth, RGB+Segmentation")
    print("🎮 Each noise config tested across all 4 observation modes")
    print("=" * 50)

    # Run all experiments
    results_df = experiment.run_all_experiments()

    # Print summary
    experiment.print_summary(results_df)

    print(
        "\n🎬 Training and evaluation videos saved in individual experiment directories"
    )
    print("📁 Check the runs/ directory for detailed results and videos")
    print("\n📊 CSV Output Files:")
    print("   • interim_results.csv - Updated after each experiment")
    print("   • final_results.csv - Complete raw results")
    print("   • detailed_analysis.csv - Enhanced analysis with categories and metrics")
    print("   • summary_statistics.csv - Aggregated statistics by noise type/level")
    print("\n🔍 Analysis includes:")
    print("   • Noise type comparisons (Gaussian vs Uniform)")
    print(
        "   • Observation mode comparisons (State vs RGB vs RGB-Depth vs RGB+Segmentation)"
    )
    print("   • Performance across different noise levels and combinations")
    print("   • Curriculum learning effectiveness analysis")
    if noise_mode == "3x":
        print("   • 3x noise magnitude scaling analysis")
    elif noise_mode == "both":
        print("   • 3x noise magnitude scaling analysis")
        print("   • Standard vs 3x noise robustness comparisons")


if __name__ == "__main__":
    main()
