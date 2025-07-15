"""
Copyright 2025 Zordi, Inc. All rights reserved.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

# =============================================================================
# Configuration constants
# =============================================================================

# Path to the reference PPO training script. We deduce it relative to this file
PPO_SCRIPT = Path(__file__).resolve().parent / "ppo.py"
# Experiment parameters requested by the user
TOTAL_TIMESTEPS = 500_000  # shorter runs for quick sanity checks

# Default parallel environment counts (can be overridden per experiment)
DEFAULT_NUM_ENVS_STATE = 256
DEFAULT_NUM_ENVS_RGB = 32

# The evaluation scripts may print values as plain floats (e.g. ``eval_reward_mean=0.12``)
# or as PyTorch scalars (e.g. ``eval_reward_mean=tensor(0.12, device='cuda:0')``).
# The regex below captures *both* patterns by optionally matching the ``tensor(" prefix
# and grabbing the first numeric token that appears.
EVAL_LINE_RE = re.compile(
    r"eval_(\w+)_mean=(?:tensor\()?([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)

# Metrics we care about from stdout (the key between "eval_" and "_mean")
METRICS_OF_INTEREST = ["success_once", "reward"]


@dataclass(frozen=True)
class ExperimentConfig:
    """Immutable configuration for a single experiment run."""

    script: str  # e.g. "ppo.py", "ppo_fast.py", etc.
    env_id: str
    expert_type: str  # "none" or "zero"
    num_envs: int
    extra_args: List[str] | None = None  # Additional CLI flags

    @property
    def script_path(self) -> Path:
        return (Path(__file__).resolve().parent / self.script).resolve()

    @property
    def name(self) -> str:
        return f"{self.script.replace('.py', '')}-{self.env_id}-{self.expert_type}"


@dataclass
class ExperimentResult:
    """Captures wall-clock runtime and evaluation metrics for one experiment."""

    config: ExperimentConfig
    wall_time_s: float
    metrics: Dict[str, float]


# =============================================================================
# Helper functions
# =============================================================================


def _parse_metrics_from_line(line: str, metrics: Dict[str, float]) -> None:
    """Extracts evaluation metrics from a single stdout line.

    The function looks for patterns like ``eval_success_rate_mean=0.87`` and
    stores the *last* observed value for each metric key.
    """
    match = EVAL_LINE_RE.search(line)
    if match:
        key, value_str = match.group(1), match.group(2)
        if key in METRICS_OF_INTEREST:
            metrics[key] = float(value_str)


def _format_seconds(seconds: float) -> str:
    """Convert a float duration in seconds to *HH:MM:SS* string."""
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


# =============================================================================
# Core logic
# =============================================================================


def run_experiment(cfg: ExperimentConfig, dry_run: bool = False) -> ExperimentResult:
    """Execute one PPO training session and gather its metrics.

    Parameters
    ----------
    cfg: ExperimentConfig
        Experiment description (environment and expert type).
    cuda: bool
        Whether to append ``--cuda`` to the underlying `ppo.py` invocation.
    dry_run: bool, default False
        If *True* the command is printed but **not** executed. Useful for quick
        verification without spending GPU resources.

    Returns:
    -------
    ExperimentResult
        Summary containing wall-time and the last set of evaluation metrics.
    """
    cmd: List[str] = [
        sys.executable,
        str(cfg.script_path),
        f"--env-id={cfg.env_id}",
        f"--expert-type={cfg.expert_type}",
        f"--num-envs={cfg.num_envs}",
        f"--total-timesteps={TOTAL_TIMESTEPS}",
        "--eval-freq=2",  # Evaluate every iteration so we always capture metrics.
        "--no-capture-video",
        "--no-save-model",
        "--no-track",
    ]
    if cfg.extra_args:
        cmd.extend(cfg.extra_args)

    if dry_run:
        print("[Dry-run]", " ".join(cmd))
        return ExperimentResult(config=cfg, wall_time_s=0.0, metrics={})

    print(f"\n[Running] {cfg.name}")
    print("Command:", " ".join(cmd))

    start_t = time.perf_counter()
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    metrics: Dict[str, float] = {}
    assert process.stdout  # Guarantee types for static checkers.
    for line in process.stdout:
        line = line.rstrip("\n")
        print(line)
        _parse_metrics_from_line(line, metrics)

    process.wait()
    wall_time = time.perf_counter() - start_t

    if process.returncode != 0:
        # Let the error propagate naturally as preferred by the user.
        raise RuntimeError(
            f"Experiment '{cfg.name}' failed with exit code {process.returncode}"
        )

    return ExperimentResult(config=cfg, wall_time_s=wall_time, metrics=metrics)


def print_summary(results: List[ExperimentResult]) -> None:
    """Prints wall-time and selected evaluation metrics for each experiment."""
    header = f"\n{'Experiment':<28} | {'Wall-time':>9} | " + " | ".join(
        f"{m:<14}" for m in METRICS_OF_INTEREST
    )
    print(header)
    print("-" * len(header))
    for res in results:
        time_str = _format_seconds(res.wall_time_s)
        metric_values = [
            f"{res.metrics.get(m, float('nan')):>14.4f}"
            if m in res.metrics
            else f"{'--':>14}"
            for m in METRICS_OF_INTEREST
        ]
        print(f"{res.config.name:<28} | {time_str:>9} | " + " | ".join(metric_values))


# =============================================================================
# Entry-point
# =============================================================================


def main() -> None:  # noqa: D103
    parser = argparse.ArgumentParser(
        description="Compare ManiSkill PPO with and without zero-expert residual"
    )
    parser.add_argument(
        "--env-id",
        default="PickCube-v1",
        help="Target ManiSkill environment ID (default: PickCube-v1)",
    )
    parser.add_argument(
        "--control-mode",
        default="pd_ee_delta_pos",
        help="Control mode for the environment (default: pd_ee_delta_pos)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing the experiments.",
    )
    parser.add_argument(
        "--expert-type",
        default="ik",
        help="Expert type for the environment (default: ik)",
    )
    args = parser.parse_args()

    env_id = args.env_id
    control_mode = args.control_mode

    configs: List[ExperimentConfig] = []

    # Helper to append test configurations
    def add_cfg(
        script: str, expert: str, num_envs: int, extra: List[str] | None = None
    ):
        # Always include control_mode in extra_args
        control_mode_arg = [f"--control-mode={control_mode}"]
        if extra is None:
            extra = control_mode_arg
        else:
            extra = control_mode_arg + extra

        configs.append(
            ExperimentConfig(
                script=script,
                env_id=env_id,
                expert_type=expert,
                num_envs=num_envs,
                extra_args=extra,
            )
        )

    # Regular PPO (state)
    for expert in ("none", "zero"):
        add_cfg("ppo.py", expert, DEFAULT_NUM_ENVS_STATE)

    # PPO fast (state)
    for expert in ("none", "zero"):
        add_cfg("ppo_fast.py", expert, DEFAULT_NUM_ENVS_STATE, ["--compile"])

    # RGB variants
    for expert in ("none", "zero"):
        add_cfg("ppo_rgb.py", expert, DEFAULT_NUM_ENVS_RGB)
    for expert in ("none", "zero"):
        add_cfg("ppo_rgb_fast.py", expert, DEFAULT_NUM_ENVS_RGB, ["--compile"])

    results: List[ExperimentResult] = []
    for cfg in configs:
        result = run_experiment(cfg, dry_run=args.dry_run)
        results.append(result)

    print_summary(results)


if __name__ == "__main__":
    sys.exit(main())
