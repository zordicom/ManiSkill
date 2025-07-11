#!/usr/bin/env python3
"""
Copyright 2025 Zordi, Inc. All rights reserved.

Simple benchmark script to test SAC performance optimizations.
"""

import time
from pathlib import Path

import cattrs
import torch
import yaml
from rl_configs import RLConfig
from rl_dataset import RLDataset
from torch.utils.data import DataLoader
from try_sac_delta_action import SACDeltaAction


def benchmark_sac_performance():
    """Benchmark SAC performance with different optimization settings."""
    # Load config
    config_path = Path("playground/rl/simple_rl/rl_galaxea_sac.yaml")
    with config_path.open("r") as f:
        raw_cfg = yaml.safe_load(f)

    # Test with minimal dataset for speed
    raw_cfg["training"]["num_epochs"] = 10
    raw_cfg["training"]["batch_size"] = 128

    # Create configurations for testing
    configs = {
        "baseline": {
            "use_amp": False,
            "torch_compile": False,
            "on_the_fly_sampling": False,
            "efficient_cql": False,
            "num_workers": 0,
        },
        "optimized": {
            "use_amp": True,
            "torch_compile": False,  # Keep disabled for stability
            "on_the_fly_sampling": True,
            "efficient_cql": True,
            "num_workers": 4,
        },
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking on device: {device}")

    for config_name, optimizations in configs.items():
        print(f"\n=== Testing {config_name.upper()} configuration ===")

        # Update config
        test_cfg = raw_cfg.copy()
        test_cfg["training"].update(optimizations)
        rl_cfg = cattrs.structure(test_cfg, RLConfig)

        try:
            # Create dataset and dataloader
            dataset = RLDataset(cfg_rl=rl_cfg, is_train_split=True)
            dataloader = DataLoader(
                dataset,
                batch_size=rl_cfg.training.batch_size,
                shuffle=True,
                num_workers=optimizations["num_workers"],
                pin_memory=optimizations.get("pin_memory", True),
            )

            # Create agent
            shape_meta = dataset.get_shape_meta()
            agent = SACDeltaAction(shape_meta, rl_cfg, device)

            # Benchmark training time
            start_time = time.time()

            # Run a few training steps
            for epoch in range(5):
                agent.current_epoch = epoch

                if optimizations["on_the_fly_sampling"]:
                    metrics = agent.update_on_the_fly(
                        dataloader, rl_cfg.training.batch_size
                    )
                else:
                    batch_data = agent.collect_batch_from_dataset(
                        dataloader, rl_cfg.training.batch_size
                    )
                    metrics = agent.update(batch_data)

                print(f"  Epoch {epoch}: Policy Loss = {metrics['policy_loss']:.4f}")

            end_time = time.time()
            total_time = end_time - start_time

            print(f"  Total time: {total_time:.2f}s")
            print(f"  Time per epoch: {total_time / 5:.2f}s")

            # Performance features enabled
            features = [k for k, v in optimizations.items() if v and k != "num_workers"]
            if optimizations["num_workers"] > 0:
                features.append(f"num_workers={optimizations['num_workers']}")
            print(f"  Features: {', '.join(features) if features else 'none'}")

        except Exception as e:
            print(f"  Error: {e}")
            continue


if __name__ == "__main__":
    benchmark_sac_performance()
