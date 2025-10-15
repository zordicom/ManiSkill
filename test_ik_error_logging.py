"""
Copyright 2025 Zordi, Inc. All rights reserved.
"""

"""
Test script to verify IK failure error logging for Panda robot with PD EE Pose Controller.
This script intentionally sends the robot to an unreachable pose to trigger IK failure.
"""

import gymnasium as gym
import numpy as np


# Create environment with Panda robot using PD EE Pose control
env = gym.make(
    "PickCube-v1",
    obs_mode="state",
    control_mode="pd_ee_pose",
    render_mode="human",
    num_envs=1,
)

obs, info = env.reset(seed=0)

print("\n" + "=" * 80)
print("Testing IK failure error logging for Panda robot")
print("=" * 80)

# First, let's do a valid action to show normal operation
print("\nStep 1: Valid action (small movement)")
action = np.array([0.01, 0.01, 0.01, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
obs, reward, terminated, truncated, info = env.step(action)
print("  ✓ Valid action executed successfully (no IK error)")

# Now trigger IK failure by trying to reach an impossible position
# Send very large movement that's likely unreachable
print("\nStep 2: Invalid action (trying to reach unreachable pose)")
print("  Sending large position delta to trigger IK failure...")
action = np.array([10.0, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
obs, reward, terminated, truncated, info = env.step(action)
print("  ✗ IK failure should have been logged above")

print("\n" + "=" * 80)
print("Test complete. Check above for IK FAILED error messages.")
print("=" * 80)

env.close()
