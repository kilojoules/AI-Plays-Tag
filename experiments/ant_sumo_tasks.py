#!/usr/bin/env python3
"""
Task generator for Ant Sumo generalization experiments.

Tests whether entropy bootstrapping findings transfer from 2D tag
to 3D MuJoCo competitive wrestling.

Conditions:
  - Baseline: auto-tuned alpha, init=0.2 (default SAC)
  - Optimal init: auto-tuned, init=0.607 (tag-optimal)
  - Low init: auto-tuned, init=0.05
  - High init: auto-tuned, init=2.0
  - Fixed alpha=0.1 (should be catastrophic if finding transfers)
  - No entropy (alpha=0)

3 seeds each = 6 x 3 = 18 tasks, 500K steps each.
"""
from __future__ import annotations
import argparse, os, subprocess, sys
from typing import List, Dict

SEEDS = [0, 1, 2]
TOTAL_TIMESTEPS = 5_000_000
BASE_OUTPUT = "experiments/results/ant_sumo"

CONDITIONS = [
    {"name": "baseline_02",   "init_alpha": 0.2,   "fixed_alpha": None},
    {"name": "optimal_0607",  "init_alpha": 0.607, "fixed_alpha": None},
    {"name": "low_005",       "init_alpha": 0.05,  "fixed_alpha": None},
    {"name": "high_20",       "init_alpha": 2.0,   "fixed_alpha": None},
    {"name": "fixed_01",      "init_alpha": 0.1,   "fixed_alpha": 0.1},
    {"name": "no_entropy",    "init_alpha": 0.001, "fixed_alpha": 0.0},
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
        print(f"Invalid task_id {task_id}. Valid: 0-{len(tasks)-1}")
        sys.exit(1)

    task = tasks[task_id]
    output_dir = os.path.join(BASE_OUTPUT, task["name"], f"seed_{task['seed']}")

    cmd = [
        sys.executable, "experiments/train_ant_sumo.py",
        "--timesteps", str(TOTAL_TIMESTEPS),
        "--num-envs", "16",
        "--init-alpha", str(task["init_alpha"]),
        "--seed", str(task["seed"]),
        "--output-dir", output_dir,
    ]
    if task["fixed_alpha"] is not None:
        cmd += ["--fixed-alpha", str(task["fixed_alpha"])]

    print(f"Task {task_id}: {task['name']} seed={task['seed']}")
    print(f"  init_alpha={task['init_alpha']}, fixed_alpha={task['fixed_alpha']}")
    print(f"  Command: {' '.join(cmd)}")
    print()

    os.makedirs(output_dir, exist_ok=True)
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        for i, t in enumerate(build_task_table()):
            fa = f"fixed={t['fixed_alpha']}" if t['fixed_alpha'] is not None else "auto"
            print(f"{i:3d}: {t['name']:<20s} init={t['init_alpha']:<6} {fa:<12s} seed={t['seed']}")
        print(f"\nTotal: {len(build_task_table())} tasks")
        return

    if args.task_id is not None:
        run_task(args.task_id)


if __name__ == "__main__":
    main()
