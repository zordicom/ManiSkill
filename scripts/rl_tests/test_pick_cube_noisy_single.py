#!/usr/bin/env python3
"""
Test script to verify PickCubeNoisy PPO setup works correctly.
Runs a single short experiment to test the pipeline.
"""

import subprocess
import sys
from pathlib import Path


def test_single_experiment():
    """Run a single short experiment to test the setup."""
    print("🧪 Testing PickCubeNoisy PPO Pipeline")
    print("=" * 50)

    # Path to the runner script
    script_dir = Path(__file__).parent
    runner_script = script_dir / "pick_cube_noisy_ppo_runner.py"

    if not runner_script.exists():
        print(f"❌ Runner script not found at {runner_script}")
        return False

    # Build test command (short experiment)
    cmd = [
        sys.executable,
        str(runner_script),
        "--env-id",
        "PickCubeNoisy-v1",
        "--exp-name",
        "test_pick_cube_noisy",
        "--total-timesteps",
        "10000",  # Short test
        "--num-envs",
        "32",  # Small number for testing
        "--num-eval-envs",
        "4",
        "--eval-freq",
        "5",
        "--obs-mode",
        "state",
        "--control-mode",
        "pd_joint_delta_pos",
        "--partial-reset",
        "--cuda",
        "--no-capture-video",
        "--no-save-model",
        "--no-track",
        "--noise-obs-noise-type=gaussian",
        "--noise-obs-noise-std=0.01",
        "--noise-reward-noise-type=none",
        "--noise-action-noise-type=none",
    ]

    print("🚀 Running test command:")
    print(f"   {' '.join(cmd)}")
    print("\n" + "=" * 50)
    print("📺 TRAINING OUTPUT (live):")
    print("=" * 50)

    try:
        # Run without capturing output so we can see real-time prints
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent.parent.parent,  # ManiSkill root
            timeout=300,  # 5 minute timeout
            check=False,
        )

        print("\n" + "=" * 50)
        print("📋 TEST RESULTS:")
        print("=" * 50)

        if result.returncode == 0:
            print("✅ Test completed successfully!")
            print("✅ PPO training ran without errors")
            print("✅ Noise parameters were likely applied (check output above)")
            return True
        else:
            print(f"❌ Test failed with return code {result.returncode}")
            return False

    except subprocess.TimeoutExpired:
        print("\n" + "=" * 50)
        print("⏰ Test timed out after 5 minutes")
        print("💡 This might be normal - the environment setup can take time")
        print("💡 Try running a shorter test or increasing timeout")
        return False
    except KeyboardInterrupt:
        print("\n" + "=" * 50)
        print("🛑 Test interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        return False


if __name__ == "__main__":
    success = test_single_experiment()
    sys.exit(0 if success else 1)
