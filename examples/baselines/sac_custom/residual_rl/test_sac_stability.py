#!/usr/bin/env python3
"""
Copyright 2025 Zordi, Inc. All rights reserved.

Simple test script to verify SAC training stability after fixes.
"""

from pathlib import Path

import cattrs
import torch
import yaml
from rl_configs import RLConfig
from rl_dataset import RLDataset
from torch.utils.data import DataLoader
from try_sac_delta_action import SACDeltaAction


def test_sac_stability():
    """Test SAC training for a few steps to check for NaN issues."""
    # Load config
    config_path = Path("playground/rl/simple_rl/rl_galaxea_sac.yaml")
    with config_path.open("r") as f:
        raw_cfg = yaml.safe_load(f)

    # Reduce epochs for quick test
    raw_cfg["training"]["num_epochs"] = 3
    raw_cfg["training"]["batch_size"] = 64

    rl_cfg = cattrs.structure(raw_cfg, RLConfig)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    try:
        # Create dataset
        dataset = RLDataset(cfg_rl=rl_cfg, is_train_split=True)
        dataloader = DataLoader(
            dataset,
            batch_size=rl_cfg.training.batch_size,
            shuffle=True,
            num_workers=0,  # Use 0 workers for testing
        )

        # Create agent
        shape_meta = dataset.get_shape_meta()
        agent = SACDeltaAction(shape_meta, rl_cfg, device)

        print("✅ Agent created successfully")

        # Test a few training steps
        for epoch in range(3):
            agent.current_epoch = epoch

            try:
                metrics = agent.update_on_the_fly(dataloader, 64)
                print(
                    f"✅ Epoch {epoch}: Policy Loss = {metrics['policy_loss']:.4f}, Q1 Loss = {metrics['q1_loss']:.4f}"
                )

                # Check for NaN in metrics
                for key, value in metrics.items():
                    if isinstance(value, float) and (value != value):  # NaN check
                        print(f"❌ NaN detected in metric {key}")
                        return False

            except Exception as e:
                print(f"❌ Error in epoch {epoch}: {e}")
                return False

        print("✅ All tests passed - SAC training is stable")
        return True

    except Exception as e:
        print(f"❌ Setup error: {e}")
        return False


if __name__ == "__main__":
    success = test_sac_stability()
    exit(0 if success else 1)
