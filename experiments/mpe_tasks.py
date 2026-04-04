#!/usr/bin/env python3
"""
Task generator for MPE simple_tag generalization experiments.

Tests whether KL responsiveness transfers from our custom tag env
to a standard benchmark (PettingZoo MPE simple_tag).

Conditions:
  - baseline (no responsiveness)
  - kl scale=0.1, 0.2, 0.3
  3 seeds each = 4 x 3 = 12 tasks

MPE is slower than our vectorized env, so we use 500K steps
(vs 5M for tag) with 32 envs.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Dict, List

SEEDS = [0, 1, 2]
TOTAL_TIMESTEPS = 500_000
BASE_OUTPUT = "experiments/results/mpe_tag"

CONDITIONS = [
    {"name": "baseline",  "method": "none", "scale": 0.0},
    {"name": "kl_01",     "method": "kl",   "scale": 0.1},
    {"name": "kl_02",     "method": "kl",   "scale": 0.2},
    {"name": "kl_03",     "method": "kl",   "scale": 0.3},
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
    output_dir = os.path.join(BASE_OUTPUT, task["name"], f"seed_{task['seed']}")

    cmd = [
        sys.executable, "experiments/train_mpe_tag.py",
        "--timesteps", str(TOTAL_TIMESTEPS),
        "--num-envs", "32",
        "--seed", str(task["seed"]),
        "--output-dir", output_dir,
        "--responsiveness", task["method"],
        "--log-interval", "2000",
        "--save-interval", "50000",
    ]
    if task["method"] != "none":
        cmd += ["--responsiveness-scale", str(task["scale"])]

    print(f"Task {task_id}: {task['name']} / seed={task['seed']}")
    print(f"  Command: {' '.join(cmd)}")
    print()

    os.makedirs(output_dir, exist_ok=True)
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="MPE simple_tag experiments")
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        tasks = build_task_table()
        for i, t in enumerate(tasks):
            print(f"{i:3d}: {t['name']:<15s} method={t['method']:<5s} "
                  f"scale={t['scale']:.1f} seed={t['seed']}")
        print(f"\nTotal: {len(tasks)} tasks")
        return

    if args.task_id is not None:
        run_task(args.task_id)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
