#!/usr/bin/env python3
"""
Task generator for AAMAS paper experiments.

Experiment 1A: Alpha=0 counterfactual (SAC with entropy disabled)
  - R4_sparse with fixed_alpha=0.0, 3 seeds
  - R4_sparse with fixed_alpha=0.1 (low entropy), 3 seeds
  - R4_sparse normal SAC (control), 3 seeds
  Total: 9 tasks (IDs 0-8)

Experiment 1B: Alpha dynamics across all presets
  - 8 presets x SAC selfplay x 3 seeds
  Total: 24 tasks (IDs 9-32)

All use Optuna-optimized SAC hyperparameters and the standard
game config (four_corners, HSM=1.15).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(__file__))
from reward_presets import PRESETS, get_preset_cli_args

SEEDS = [0, 1, 2]
TOTAL_TIMESTEPS = 5_000_000

BASE_OUTPUT = "experiments/results/paper_ablations"

# Optuna-optimized SAC hyperparameters (from sac_hpo_v3)
SAC_HPARAMS = {
    "lr": 0.000225,
    "gamma": 0.969,
    "tau": 0.00658,
    "init_alpha": 0.607,
    "buffer_size": 100000,
    "batch_size": 256,
    "updates_per_step": 4,
    "warmup_steps": 5000,
}

# --- Experiment 1A: Alpha counterfactual ---
# Three conditions on R4_sparse: no entropy, low entropy, normal (auto-tuned)
ALPHA_CONDITIONS = [
    {"name": "alpha0",   "fixed_alpha": 0.0},
    {"name": "alpha01",  "fixed_alpha": 0.1},
    {"name": "control",  "fixed_alpha": None},  # normal auto-tuned
]

# --- Experiment 1B: All presets with SAC selfplay ---
ALL_PRESETS = list(PRESETS.keys())


def build_task_table() -> List[Dict]:
    """Build the full task table."""
    tasks = []

    # Experiment 1A: Alpha counterfactual on R4_sparse (9 tasks)
    for cond in ALPHA_CONDITIONS:
        for seed in SEEDS:
            tasks.append({
                "experiment": "1A_counterfactual",
                "preset": "R4_sparse",
                "condition": cond["name"],
                "fixed_alpha": cond["fixed_alpha"],
                "seed": seed,
            })

    # Experiment 1B: Alpha dynamics across all presets (24 tasks)
    for preset in ALL_PRESETS:
        for seed in SEEDS:
            tasks.append({
                "experiment": "1B_alpha_dynamics",
                "preset": preset,
                "condition": "auto",
                "fixed_alpha": None,
                "seed": seed,
            })

    return tasks


def run_task(task_id: int):
    """Run a single task by its ID."""
    tasks = build_task_table()
    if task_id < 0 or task_id >= len(tasks):
        print(f"Invalid task_id {task_id}. Valid range: 0-{len(tasks)-1}")
        sys.exit(1)

    task = tasks[task_id]
    preset = task["preset"]
    condition = task["condition"]
    fixed_alpha = task["fixed_alpha"]
    seed = task["seed"]
    experiment = task["experiment"]

    # Build output directory
    output_dir = os.path.join(BASE_OUTPUT, experiment, preset,
                              condition, f"seed_{seed}")

    # Get preset-specific reward CLI args
    preset_args = get_preset_cli_args(preset)

    # SAC hyperparameter args
    hp = SAC_HPARAMS
    hp_args = [
        "--lr", str(hp["lr"]),
        "--batch-size", str(hp["batch_size"]),
        "--buffer-size", str(hp["buffer_size"]),
        "--updates-per-step", str(hp["updates_per_step"]),
        "--warmup-steps", str(hp["warmup_steps"]),
    ]

    # Fixed alpha args
    alpha_args = []
    if fixed_alpha is not None:
        alpha_args = ["--fixed-alpha", str(fixed_alpha)]
    alpha_args += ["--init-alpha", str(hp["init_alpha"])]

    # Build command
    cmd = [
        sys.executable, "trainer/train_selfplay_sac.py",
        "--timesteps", str(TOTAL_TIMESTEPS),
        "--num-envs", "64",
        "--seed", str(seed),
        "--output-dir", output_dir,
        "--log-interval", "5000",
        "--save-interval", "100000",
    ] + preset_args + hp_args + alpha_args

    print(f"Task {task_id}: {experiment} / {preset} / {condition} / seed={seed}")
    print(f"  fixed_alpha={fixed_alpha}")
    print(f"  Output: {output_dir}")
    print(f"  Command: {' '.join(cmd)}")
    print()

    os.makedirs(output_dir, exist_ok=True)
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="AAMAS paper ablation experiments")
    parser.add_argument("--task-id", type=int, default=None,
                        help="Run a single task by ID")
    parser.add_argument("--list", action="store_true",
                        help="List all tasks and exit")
    parser.add_argument("--experiment", type=str, default=None,
                        choices=["1A", "1B"],
                        help="Filter task list by experiment")
    args = parser.parse_args()

    tasks = build_task_table()

    if args.list:
        for i, t in enumerate(tasks):
            if args.experiment and not t["experiment"].startswith(f"{args.experiment}_"):
                continue
            alpha_str = f"fixed={t['fixed_alpha']}" if t['fixed_alpha'] is not None else "auto"
            print(f"{i:3d}: {t['experiment']:20s} {t['preset']:20s} "
                  f"{t['condition']:10s} seed={t['seed']} alpha={alpha_str}")
        total = len(tasks) if not args.experiment else sum(
            1 for t in tasks if t["experiment"].startswith(f"{args.experiment}_"))
        print(f"\nTotal: {total} tasks")
        return

    if args.task_id is not None:
        run_task(args.task_id)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
