#!/usr/bin/env python3
"""
Test script to run a single experiment from the main PickCubeNoisy PPO suite.
This allows testing the live output functionality with one configuration.
"""

import sys
from pathlib import Path

# Add the script directory to path to import our module
sys.path.insert(0, str(Path(__file__).parent))

from pick_cube_noisy_ppo import PickCubeNoisyExperiment


def test_single_main_experiment():
    """Run a single experiment from the main suite with live output."""
    print("🧪 Testing PickCubeNoisy Main Experiment (Live Output)")
    print("=" * 60)

    # Initialize experiment manager with shorter settings for testing
    experiment = PickCubeNoisyExperiment(
        base_output_dir="test_runs/single_main_experiment",
        total_timesteps=25000,  # Very short for testing evaluation
        num_envs=64,  # Fewer environments for testing
        num_eval_envs=4,  # Fewer eval envs for faster testing
        eval_freq=5,  # More frequent evaluation
        save_videos=True,  # Enable videos to test video functionality
    )

    # Get one noise configuration for testing
    configs = experiment.get_noise_configurations()
    test_config = configs[1]  # Light observation noise

    print(f"🔬 Running test with configuration: {test_config['name']}")
    print(f"📝 Description: {test_config['description']}")

    # Run the single experiment
    success, results = experiment.run_experiment(test_config)

    print("\n" + "=" * 60)
    print("📋 FINAL TEST RESULTS:")
    print("=" * 60)

    if success:
        print("✅ Test completed successfully!")
        print(f"⏱️  Duration: {results['duration']:.1f} seconds")
        print(f"📊 Success rate: {results['success_rate']}")
        if results["success_rate"] == -1.0:
            print("💡 Check WandB/TensorBoard for actual success metrics")
        print(f"🗂️  Results saved to: {experiment.base_output_dir}")
    else:
        print("❌ Test failed")
        print(f"🚨 Error: {results.get('error', 'Unknown error')}")

    return success


if __name__ == "__main__":
    success = test_single_main_experiment()
    sys.exit(0 if success else 1)
