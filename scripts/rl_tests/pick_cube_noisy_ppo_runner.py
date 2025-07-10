#!/usr/bin/env python3
"""
Simple PPO Runner for PickCubeNoisy environments.
This script reads noise parameters from environment variables and runs training.
"""

import os
import sys
from pathlib import Path

# Filter out noise parameters and store them
noise_params = {}
filtered_args = []

for arg in sys.argv[1:]:
    if arg.startswith("--noise-"):
        # Convert --noise-param=value to environment variable
        if "=" in arg:
            key, value = arg.split("=", 1)
            key = key.replace("--noise-", "").replace("-", "_").upper()
            os.environ[f"NOISE_{key}"] = value
            noise_params[key] = value
    else:
        filtered_args.append(arg)

# Update sys.argv to only include non-noise arguments
sys.argv = [sys.argv[0]] + filtered_args

# Add ManiSkill root to path
maniskill_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(maniskill_root))

# Import the noisy environment to register it
import mani_skill.envs.tasks.tabletop.pick_cube_noisy

# Read the original ppo_fast.py script
original_ppo_path = maniskill_root / "examples/baselines/ppo/ppo_fast.py"

with open(original_ppo_path, "r") as f:
    ppo_code = f.read()

# Add noise parameter handling to environment creation
noise_patch = """
# Add noise parameters from environment variables for PickCubeNoisy
if "PickCubeNoisy" in args.env_id:
    noise_params = {}
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
        print(f"🔊 Applied noise parameters: {noise_params}")
"""

# Insert the patch after the env_kwargs definition
insert_point = 'if args.robot_uids is not None:\n        env_kwargs["robot_uids"] = args.robot_uids'
if insert_point in ppo_code:
    ppo_code = ppo_code.replace(insert_point, insert_point + "\n    " + noise_patch)
else:
    # Fallback: insert after env_kwargs = {}
    fallback_point = "env_kwargs = {}"
    if fallback_point in ppo_code:
        ppo_code = ppo_code.replace(
            fallback_point, fallback_point + "\n    " + noise_patch
        )

# Print debug information
if noise_params:
    print(f"🔊 Parsed noise parameters: {noise_params}")
    print(f"🔧 Filtered arguments: {filtered_args}")

# Execute the modified script
exec(ppo_code)
