#!/usr/bin/env python3
"""
Task generator for responsiveness intrinsic reward experiments.

Tests whether TE/KL responsiveness rewards can replace hand-crafted
anti-degenerate shaping (wall penalty, speed bonus) while maintaining
or improving agent strength.

Experimental design:
  Baseline conditions (no responsiveness reward):
    - R4_sparse (no shaping, no responsiveness) — control
    - R2_hider_active (hand-crafted anti-degenerate) — current best

  Responsiveness conditions on R4_sparse:
    - TE only, scale in {0.05, 0.1, 0.2}
    - KL only, scale in {0.05, 0.1, 0.2}
    - Both, scale in {0.05, 0.1, 0.2}

  3 seeds each = (2 + 9) x 3 = 33 tasks
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(__file__))
from reward_presets import get_preset_cli_args

SEEDS = [0, 1, 2]
TOTAL_TIMESTEPS = 5_000_000
BASE_OUTPUT = "experiments/results/responsiveness_sweep"

# Optuna-optimized SAC hyperparameters
SAC_HPARAMS = {
    "lr": 0.000225,
    "init_alpha": 0.607,
    "buffer_size": 100000,
    "batch_size": 256,
    "updates_per_step": 4,
    "warmup_steps": 5000,
}

CONDITIONS = [
    # Baselines
    {"name": "R4_sparse_baseline", "preset": "R4_sparse", "method": "none", "scale": 0.0},
    {"name": "R2_active_baseline", "preset": "R2_hider_active", "method": "none", "scale": 0.0},
    # TE only
    {"name": "R4_te_005", "preset": "R4_sparse", "method": "te", "scale": 0.05},
    {"name": "R4_te_01",  "preset": "R4_sparse", "method": "te", "scale": 0.1},
    {"name": "R4_te_02",  "preset": "R4_sparse", "method": "te", "scale": 0.2},
    # KL only
    {"name": "R4_kl_005", "preset": "R4_sparse", "method": "kl", "scale": 0.05},
    {"name": "R4_kl_01",  "preset": "R4_sparse", "method": "kl", "scale": 0.1},
    {"name": "R4_kl_02",  "preset": "R4_sparse", "method": "kl", "scale": 0.2},
    # Both
    {"name": "R4_both_005", "preset": "R4_sparse", "method": "both", "scale": 0.05},
    {"name": "R4_both_01",  "preset": "R4_sparse", "method": "both", "scale": 0.1},
    {"name": "R4_both_02",  "preset": "R4_sparse", "method": "both", "scale": 0.2},
]


def build_task_table() -> List[Dict]:
    tasks = []
    for cond in CONDITIONS:
        for seed in SEEDS:
            tasks.append({**cond, "seed": seed})
    return tasks


def run_task(task_id: int):
    tasks = build_task_table()
    if task_id < 0 or task_id >= len(tasks):
        print(f"Invalid task_id {task_id}. Valid range: 0-{len(tasks)-1}")
        sys.exit(1)

    task = tasks[task_id]
    name = task["name"]
    preset = task["preset"]
    method = task["method"]
    scale = task["scale"]
    seed = task["seed"]

    output_dir = os.path.join(BASE_OUTPUT, name, f"seed_{seed}")
    preset_args = get_preset_cli_args(preset)

    hp = SAC_HPARAMS
    hp_args = [
        "--lr", str(hp["lr"]),
        "--init-alpha", str(hp["init_alpha"]),
        "--buffer-size", str(hp["buffer_size"]),
        "--batch-size", str(hp["batch_size"]),
        "--updates-per-step", str(hp["updates_per_step"]),
        "--warmup-steps", str(hp["warmup_steps"]),
    ]

    resp_args = ["--responsiveness", method]
    if method != "none":
        resp_args += ["--responsiveness-scale", str(scale)]

    cmd = [
        sys.executable, "trainer/train_selfplay_sac.py",
        "--timesteps", str(TOTAL_TIMESTEPS),
        "--num-envs", "64",
        "--seed", str(seed),
        "--output-dir", output_dir,
        "--log-interval", "5000",
        "--save-interval", "100000",
    ] + preset_args + hp_args + resp_args

    print(f"Task {task_id}: {name} / seed={seed}")
    print(f"  method={method}, scale={scale}, preset={preset}")
    print(f"  Output: {output_dir}")
    print(f"  Command: {' '.join(cmd)}")
    print()

    os.makedirs(output_dir, exist_ok=True)
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Responsiveness intrinsic reward experiments")
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    tasks = build_task_table()

    if args.list:
        for i, t in enumerate(tasks):
            print(f"{i:3d}: {t['name']:<25s} method={t['method']:<5s} "
                  f"scale={t['scale']:.2f} seed={t['seed']}")
        print(f"\nTotal: {len(tasks)} tasks")
        return

    if args.task_id is not None:
        run_task(args.task_id)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
