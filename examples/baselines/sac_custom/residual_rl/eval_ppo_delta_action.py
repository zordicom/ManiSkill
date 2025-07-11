#!/usr/bin/env python3
"""
Copyright 2025 Zordi, Inc. All rights reserved.

Visualize and evaluate a trained PPO-Δ (delta-action) policy.

For selected episodes we plot, per action-dimension, the ground-truth
trajectory (expert + stored residual) against the policy prediction
(expert + predicted residual).

Command-line synopsis:
    python playground/rl/eval_ppo_delta_action.py \
        --config playground/rl/rl_galaxea_ppo.yaml \
        --checkpoint outputs/ppo_towel_folding/model_id/checkpoint_best.pt \
        --n-episodes 3 --plot-dir plots/ppo_delta_action

The script relies on:
•  RLDataset - for loading & normalizing observations
•  DeltaPolicyNetwork - imported from try_ppo_delta_action.py

All functions stay <50 LOC and the file <500 LOC, in line with the
project code-style guidelines.
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.lines import Line2D
from rl_configs import RLConfig

# ---------------------------------------------------------------------------
# Local package imports (dynamically add current dir so sibling module loads)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))  # Enables `import try_ppo_delta_action`

from rl_dataset import RLDataset  # noqa: E402
from try_ppo_delta_action import DeltaPolicyNetwork  # noqa: E402

from zordi_vla.utils.logging_utils import setup_logger  # noqa: E402

logger = setup_logger("eval_ppo_delta_action")

try:
    import cattrs  # Heavy but already used elsewhere in repo
except ImportError as exc:  # pragma: no cover - should never happen in env
    logger.error("Missing dependency: %s", exc)
    raise

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Evaluate PPO Δ-action model")
    parser.add_argument(
        "--config",
        type=str,
        default="playground/rl/rl_galaxea_ppo.yaml",
        help="Path to RL YAML config used for training",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint_*.pt produced by training script",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=5,
        help="Number of random episodes to evaluate",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default="plots/ppo_delta_action",
        help="Directory where plots will be saved",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use mean action instead of sampling",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Computation device override",
    )
    return parser.parse_args()


def _resolve_device(arg: str) -> torch.device:
    """Resolve CUDA / CPU device from string option."""
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def _load_config(cfg_path: Path):
    """Load YAML → RLConfig (structured)."""
    with cfg_path.open("r", encoding="utf-8") as fp:
        raw_cfg: Dict = yaml.safe_load(fp)

    return cattrs.structure(raw_cfg, RLConfig), raw_cfg


def _load_policy(
    rl_cfg,
    shape_meta: Dict,
    checkpoint_path: Path,
    device: torch.device,
) -> DeltaPolicyNetwork:
    """Instantiate DeltaPolicyNetwork and load weights from checkpoint."""
    policy = DeltaPolicyNetwork(shape_meta, rl_cfg.network, device)
    ckpt = torch.load(checkpoint_path, map_location=device)

    if "policy_state" not in ckpt:
        raise KeyError("policy_state not found in checkpoint")

    missing, unexpected = policy.load_state_dict(ckpt["policy_state"], strict=False)
    if missing or unexpected:
        logger.warning(
            "State-dict mismatch → missing: %s | unexpected: %s", missing, unexpected
        )

    policy.eval()
    logger.info("Policy loaded from %s", checkpoint_path)
    return policy


def _prepare_single_obs(obs_norm: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Add outer batch dim to match training DataLoader collation."""
    return {key: tensor.unsqueeze(0) for key, tensor in obs_norm.items()}


def _evaluate_episode(
    dataset: RLDataset,
    policy: DeltaPolicyNetwork,
    ep_name: str,
    device: torch.device,
    deterministic: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run policy over *all* steps of one episode and collect trajectories."""
    steps: List[Dict] = dataset.episode_dict[ep_name]
    pred_list: List[np.ndarray] = []
    gt_list: List[np.ndarray] = []

    for step_item in steps:
        # ------------------------------------------------------------------
        # Build normalized observation dict --------------------------------
        # ------------------------------------------------------------------
        obs_norm = dataset._build_processed_obs(step_item)

        # Expert action (normalized → raw)
        expert_norm = obs_norm["expert_action"]  # shape [action_dim]
        expert_raw = dataset.expert_action_normalizer.denormalize(
            expert_norm
        ).numpy()  # ndarray

        # Ground-truth residual action (already raw in dataset)
        residual_raw_np = np.asarray(step_item["residual_action"], dtype=np.float32)

        # Final ground-truth action in *raw* scale
        gt_raw = expert_raw + residual_raw_np

        # ------------------------------------------------------------------
        # Policy prediction -------------------------------------------------
        # ------------------------------------------------------------------
        obs_batched = _prepare_single_obs(obs_norm)
        obs_device = {k: t.to(device) for k, t in obs_batched.items()}

        with torch.no_grad():
            # Obtain delta distribution (normalized space)
            dist = policy.get_distribution(obs_device)
            delta_norm = dist.mean if deterministic else dist.sample()

        # Remove batch dim and move to CPU
        delta_norm = delta_norm.squeeze(0).cpu()

        # Convert delta to *raw* scale using the delta action normalizer
        delta_raw = dataset.action_normalizer.denormalize(delta_norm).numpy()

        # Final predicted action in raw scale
        pred_raw = expert_raw + delta_raw

        # ------------------------------------------------------------------
        # Append to lists ---------------------------------------------------
        # ------------------------------------------------------------------
        gt_list.append(gt_raw)
        pred_list.append(pred_raw)

    return np.stack(pred_list), np.stack(gt_list)


def _plot_episode(
    pred: np.ndarray,
    gt: np.ndarray,
    ep_name: str,
    save_dir: Path,
    grade_name: str,
):
    """Generate matplotlib plot comparing gt vs prediction for each dim."""
    d_act = gt.shape[1]
    time_axis = np.arange(len(gt))

    fig, axes = plt.subplots(d_act, 1, figsize=(14, d_act * 2.5), sharex=True)
    axes = np.atleast_1d(axes)  # Handles d_act = 1 gracefully

    for dim in range(d_act):
        ax = axes[dim]
        ax.plot(time_axis, gt[:, dim], color="blue", linewidth=1.3)
        ax.plot(time_axis, pred[:, dim], color="red", linestyle="--", linewidth=1.1)
        ax.set_ylabel(f"Dim {dim}")
        ax.grid(True, linestyle=":", alpha=0.6)

    axes[0].legend(
        handles=[
            Line2D([0], [0], color="blue", lw=1.3, label="Ground Truth"),
            Line2D([0], [0], color="red", lw=1.1, linestyle="--", label="Prediction"),
        ],
        loc="best",
    )

    axes[-1].set_xlabel("Timestep")
    mse = float(np.mean((pred - gt) ** 2))
    fig.suptitle(f"Episode: {ep_name} | MSE: {mse:.6f}")

    save_dir.mkdir(parents=True, exist_ok=True)
    save_filename = f"grade_{grade_name}_ep_{ep_name}.jpg"
    save_path = save_dir / save_filename
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    logger.info("Saved plot → %s", save_path)

    return mse


# ---------------------------------------------------------------------------
# Episode grouping utilities -------------------------------------------------
# ---------------------------------------------------------------------------


def _group_episodes_by_grade(
    dataset: RLDataset,
) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """Return (groups, ep_to_grade) using `inference_grade.json` files.

    The JSON is expected to contain a key `feedback` representing the grade.
    Missing or malformed entries fall back to "unknown".
    """
    groups: Dict[str, List[str]] = {}
    ep_to_grade: Dict[str, str] = {}

    # Build mapping ep_name -> grade ------------------------------------
    for grade_json in dataset.dataset_path.glob("**/inference_grade.json"):
        try:
            grade_data = json.loads(grade_json.read_text())
            grade_val = str(grade_data.get("feedback", "unknown"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to parse %s: %s", grade_json, exc)
            grade_val = "unknown"

        ep_name = grade_json.parent.name
        if ep_name not in dataset.episode_dict:
            continue  # Skip directories not loaded into the dataset

        ep_to_grade[ep_name] = grade_val
        groups.setdefault(grade_val, []).append(ep_name)

    # Handle episodes without any grade file ------------------------------
    for ep_name in dataset.episode_dict.keys():
        if ep_name not in ep_to_grade:
            ep_to_grade[ep_name] = "unknown"
            groups.setdefault("unknown", []).append(ep_name)

    return groups, ep_to_grade


def _sample_episodes_by_group(
    groups: Dict[str, List[str]],
    total_samples: int,
    rng: random.Random,
) -> List[str]:
    """Return a list of episode names sampled evenly across grade groups."""
    if total_samples <= 0:
        return []

    grade_keys = list(groups.keys())
    n_grades = len(grade_keys)

    # Initial allocation (floor division)
    base_per_grade = total_samples // n_grades
    remainder = total_samples % n_grades

    selected: List[str] = []

    for grade in grade_keys:
        ep_pool = groups[grade]
        if not ep_pool:
            continue

        k = min(base_per_grade, len(ep_pool))
        selected.extend(rng.sample(ep_pool, k))

    # Distribute the remainder -------------------------------------------
    if remainder > 0:
        rng.shuffle(grade_keys)  # Random order for allocating extra slots
        for grade in grade_keys:
            if len(selected) >= total_samples:
                break
            ep_pool = [ep for ep in groups[grade] if ep not in selected]
            if ep_pool:
                selected.append(rng.choice(ep_pool))

    # Fallback: if still not enough (e.g., not enough distinct eps), sample randomly
    if len(selected) < total_samples:
        all_eps = [ep for eps in groups.values() for ep in eps]
        remaining = [ep for ep in all_eps if ep not in selected]
        if remaining:
            selected.extend(
                rng.sample(
                    remaining, min(len(remaining), total_samples - len(selected))
                )
            )

    return selected[:total_samples]


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI wrapper."""
    args = _parse_args()
    device = _resolve_device(args.device)
    logger.info("Using device: %s", device)

    # ------------------------------------------------------------------
    # Load config + dataset --------------------------------------------
    # ------------------------------------------------------------------
    cfg_path = Path(args.config).expanduser().resolve()
    rl_cfg, _ = _load_config(cfg_path)

    dataset = RLDataset(cfg_rl=rl_cfg, is_train_split=False)
    logger.info("Dataset loaded with %d episodes", len(dataset.episode_dict))

    shape_meta = dataset.get_shape_meta()

    # ------------------------------------------------------------------
    # Load policy -------------------------------------------------------
    # ------------------------------------------------------------------
    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    policy = _load_policy(rl_cfg, shape_meta, ckpt_path, device)

    # ------------------------------------------------------------------
    # Prepare plot directory -------------------------------------------
    # ------------------------------------------------------------------
    model_id_dir = ckpt_path.parent.name  # e.g., "model_20250625_201231"
    plot_root = Path(args.plot_dir).expanduser().resolve() / model_id_dir
    plot_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Select episodes ---------------------------------------------------
    # ------------------------------------------------------------------
    rng = random.Random(args.seed)

    groups, ep_to_grade = _group_episodes_by_grade(dataset)
    selected_eps = _sample_episodes_by_group(groups, args.n_episodes, rng)

    if not selected_eps:
        logger.error(
            "No episodes selected -- check dataset and inference_grade.json files"
        )
        sys.exit(1)

    logger.info("Episode groups: %s", {k: len(v) for k, v in groups.items()})
    logger.info("Evaluating episodes: %s", selected_eps)

    # ------------------------------------------------------------------
    # Evaluation loop ---------------------------------------------------
    # ------------------------------------------------------------------
    mse_list: List[float] = []

    for ep in selected_eps:
        grade_name = ep_to_grade.get(ep, "unknown")
        pred_traj, gt_traj = _evaluate_episode(
            dataset, policy, ep, device, args.deterministic
        )
        mse = _plot_episode(pred_traj, gt_traj, ep, plot_root, grade_name)
        mse_list.append(mse)

    # ------------------------------------------------------------------
    # Summary -----------------------------------------------------------
    # ------------------------------------------------------------------
    overall_mse = float(np.mean(mse_list)) if mse_list else float("nan")
    summary = ", ".join(f"{e}:{m:.6f}" for e, m in zip(selected_eps, mse_list))
    logger.info("Evaluation summary → %s", summary)
    logger.info("Overall average MSE: %.6f", overall_mse)


if __name__ == "__main__":  # pragma: no cover
    main()
