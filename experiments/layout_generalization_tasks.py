#!/usr/bin/env python3
"""
Task generator for layout generalization experiments.

Tests whether KL responsiveness transfers across arena layouts
without re-tuning the scale parameter.

Conditions:
  3 layouts (empty, central_cross, playground) x
  2 methods (baseline, kl scale=0.2) x
  3 seeds = 18 tasks

Uses R4_sparse (no shaping) for all conditions.
The key question: does KL responsiveness at scale=0.2 (the sweet
spot from four_corners) also reduce corner-camping on different layouts?
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
BASE_OUTPUT = "experiments/results/layout_generalization"

# Layouts to test (four_corners already tested in responsiveness sweep)
LAYOUTS = ["empty", "central_cross", "playground"]

# Optuna-optimized SAC hyperparameters
SAC_HPARAMS = {
    "lr": 0.000225,
    "init_alpha": 0.607,
    "buffer_size": 100000,
    "batch_size": 256,
    "updates_per_step": 4,
    "warmup_steps": 5000,
}

CONDITIONS = []
for layout in LAYOUTS:
    CONDITIONS.append({
        "name": f"{layout}_baseline",
        "layout": layout,
        "method": "none",
        "scale": 0.0,
    })
    CONDITIONS.append({
        "name": f"{layout}_kl_02",
        "layout": layout,
        "method": "kl",
        "scale": 0.2,
    })


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
    output_dir = os.path.join(BASE_OUTPUT, task["name"], f"seed_{task['seed']}")

    # R4_sparse preset args, but override layout
    preset_args = get_preset_cli_args("R4_sparse")
    # Remove the --layout from preset args (it defaults to four_corners)
    filtered_args = []
    skip_next = False
    for arg in preset_args:
        if skip_next:
            skip_next = False
            continue
        if arg == "--layout":
            skip_next = True
            continue
        filtered_args.append(arg)

    hp = SAC_HPARAMS
    hp_args = [
        "--lr", str(hp["lr"]),
        "--init-alpha", str(hp["init_alpha"]),
        "--buffer-size", str(hp["buffer_size"]),
        "--batch-size", str(hp["batch_size"]),
        "--updates-per-step", str(hp["updates_per_step"]),
        "--warmup-steps", str(hp["warmup_steps"]),
    ]

    resp_args = ["--responsiveness", task["method"]]
    if task["method"] != "none":
        resp_args += ["--responsiveness-scale", str(task["scale"])]

    cmd = [
        sys.executable, "trainer/train_selfplay_sac.py",
        "--timesteps", str(TOTAL_TIMESTEPS),
        "--num-envs", "64",
        "--layout", task["layout"],
        "--hider-speed-mult", "1.15",
        "--seed", str(task["seed"]),
        "--output-dir", output_dir,
        "--log-interval", "5000",
        "--save-interval", "100000",
    ] + filtered_args + hp_args + resp_args

    print(f"Task {task_id}: {task['name']} / seed={task['seed']}")
    print(f"  layout={task['layout']}, method={task['method']}, scale={task['scale']}")
    print(f"  Output: {output_dir}")
    print(f"  Command: {' '.join(cmd)}")
    print()

    os.makedirs(output_dir, exist_ok=True)
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Layout generalization experiments")
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        tasks = build_task_table()
        for i, t in enumerate(tasks):
            print(f"{i:3d}: {t['name']:<30s} layout={t['layout']:<15s} "
                  f"method={t['method']:<5s} scale={t['scale']:.1f} seed={t['seed']}")
        print(f"\nTotal: {len(tasks)} tasks")
        return

    if args.task_id is not None:
        run_task(args.task_id)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
