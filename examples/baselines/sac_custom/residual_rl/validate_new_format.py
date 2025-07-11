#!/usr/bin/env python3
"""
Copyright 2025 Zordi, Inc. All rights reserved.

Validation script for the updated SAC delta action pipeline with structured rollout data.

This script tests:
1. Loading rollout data with structured states/actions and extra observations
2. Converting structured data to flat vectors and back
3. SAC model initialization with the new data format
4. Forward passes through the complete pipeline

Usage:
    python playground/rl/residual_rl/validate_new_format.py \
        --config playground/rl/residual_rl/rl_galaxea_sac_box_pnp.yaml \
        --rollout-dir playground/rl/residual_rl/galaxea_rollouts/box_pnp
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml
from rl_configs import RLConfig
from rl_dataset import RLDataset
from torch.utils.data import DataLoader
from try_sac_delta_action import SACDeltaAction

try:
    import cattrs
except ImportError as exc:
    print(f"Missing dependency: {exc}")
    sys.exit(1)

from zordi_vla.utils.logging_utils import setup_logger

logger = setup_logger("validate_new_format")


def validate_dataset(rl_cfg: RLConfig) -> tuple[RLDataset, dict]:
    """Validate the dataset loading and processing."""
    logger.info("🔍 Validating dataset loading...")

    # Create dataset
    dataset = RLDataset(cfg_rl=rl_cfg, is_train_split=True)
    logger.info(f"✅ Dataset loaded with {len(dataset)} samples")

    # Get shape metadata
    shape_meta = dataset.get_shape_meta()
    logger.info(f"Shape metadata: {shape_meta}")

    # Test field definitions
    if hasattr(dataset, "state_fields") and dataset.state_fields:
        logger.info(f"State fields: {dataset.state_fields}")
    if hasattr(dataset, "action_fields") and dataset.action_fields:
        logger.info(f"Action fields: {dataset.action_fields}")
    if hasattr(dataset, "extra_obs_fields") and dataset.extra_obs_fields:
        logger.info(f"Extra observation fields: {dataset.extra_obs_fields}")

    # Test data sample
    sample = dataset[0]
    logger.info(f"Sample observation keys: {list(sample['obs'].keys())}")
    logger.info(f"State shape: {sample['obs']['state'].shape}")
    logger.info(f"Expert action shape: {sample['obs']['expert_action'].shape}")
    logger.info(f"Action shape: {sample['action'].shape}")

    # Check for extra observations
    extra_obs_count = 0
    for key in sample["obs"].keys():
        if key not in ["state", "expert_action"] and "rgb" not in key.lower():
            logger.info(f"Extra observation '{key}': {sample['obs'][key].shape}")
            extra_obs_count += 1

    logger.info(f"Found {extra_obs_count} extra observation fields")

    return dataset, shape_meta


def validate_sac_model(shape_meta: dict, rl_cfg: RLConfig) -> SACDeltaAction:
    """Validate SAC model initialization and forward passes."""
    logger.info("🤖 Validating SAC model initialization...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Initialize SAC agent
    agent = SACDeltaAction(shape_meta, rl_cfg, device)
    logger.info("✅ SAC agent initialized successfully")

    return agent


def validate_forward_passes(agent: SACDeltaAction, dataset: RLDataset) -> None:
    """Validate forward passes through the SAC model."""
    logger.info("🔄 Validating model forward passes...")

    # Create data loader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    batch = next(iter(dataloader))

    # Move observations to device
    obs = {key: tensor.to(agent.device) for key, tensor in batch["obs"].items()}
    actions = batch["action"].to(agent.device)

    logger.info("Batch observation shapes:")
    for key, tensor in obs.items():
        logger.info(f"  {key}: {tensor.shape}")
    logger.info(f"Actions shape: {actions.shape}")

    # Test policy forward pass
    with torch.no_grad():
        try:
            mean, log_std = agent.policy.forward(obs)
            logger.info(
                f"✅ Policy forward pass: mean={mean.shape}, log_std={log_std.shape}"
            )

            # Test policy sampling
            delta_action, log_prob = agent.policy.sample(obs, deterministic=False)
            logger.info(
                f"✅ Policy sampling: delta_action={delta_action.shape}, log_prob={log_prob.shape}"
            )

            # Test Q-network forward passes
            q1_vals = agent.q1(obs, delta_action)
            q2_vals = agent.q2(obs, delta_action)
            logger.info(
                f"✅ Q-network forward passes: Q1={q1_vals.shape}, Q2={q2_vals.shape}"
            )

            # Test target Q-networks
            q1_target_vals = agent.q1_target(obs, delta_action)
            q2_target_vals = agent.q2_target(obs, delta_action)
            logger.info(
                f"✅ Target Q-networks: Q1_target={q1_target_vals.shape}, Q2_target={q2_target_vals.shape}"
            )

        except Exception as e:
            logger.error(f"❌ Forward pass failed: {e}")
            raise


def validate_training_step(agent: SACDeltaAction, dataset: RLDataset) -> None:
    """Validate a single training step."""
    logger.info("🏋️ Validating training step...")

    # Create data loader
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

    try:
        # Test on-the-fly training method
        metrics = agent.update_on_the_fly(dataloader, num_steps=16)
        logger.info("✅ On-the-fly training step completed")
        logger.info(f"Sample metrics: {dict(list(metrics.items())[:5])}")

    except Exception as e:
        logger.error(f"❌ Training step failed: {e}")
        raise


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description="Validate SAC delta action pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="playground/rl/residual_rl/rl_galaxea_sac_box_pnp.yaml",
        help="Path to RL configuration file",
    )
    parser.add_argument(
        "--rollout-dir",
        type=str,
        help="Path to rollout directory (overrides config)",
    )
    args = parser.parse_args()

    logger.info("🚀 Starting validation of SAC delta action pipeline...")

    # Load configuration
    cfg_path = Path(args.config).expanduser().resolve()
    if not cfg_path.exists():
        logger.error(f"Config file not found: {cfg_path}")
        sys.exit(1)

    with cfg_path.open("r", encoding="utf-8") as fp:
        raw_cfg = yaml.safe_load(fp)

    rl_cfg = cattrs.structure(raw_cfg, RLConfig)

    # Override rollout directory if provided
    if args.rollout_dir:
        rollout_path = Path(args.rollout_dir).expanduser().resolve()
        if not rollout_path.exists():
            logger.error(f"Rollout directory not found: {rollout_path}")
            sys.exit(1)
        rl_cfg.dataset.path = str(rollout_path)

    logger.info(f"Configuration loaded from: {cfg_path}")
    logger.info(f"Dataset path: {rl_cfg.dataset.path}")

    try:
        # Step 1: Validate dataset
        dataset, shape_meta = validate_dataset(rl_cfg)

        # Step 2: Validate SAC model
        agent = validate_sac_model(shape_meta, rl_cfg)

        # Step 3: Validate forward passes
        validate_forward_passes(agent, dataset)

        # Step 4: Validate training step
        validate_training_step(agent, dataset)

        logger.info("🎉 All validations passed successfully!")
        logger.info(
            "✅ The SAC delta action pipeline is ready for training with the new data format"
        )

    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
