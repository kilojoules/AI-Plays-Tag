#!/usr/bin/env python3
"""
Init-alpha replication on empty + central_cross layouts.

Tests whether the inverted-U pattern (0.607 best) holds across layouts
or is four_corners-specific.

4 init_alpha values x 2 layouts x 3 seeds = 24 tasks.
"""
from __future__ import annotations
import argparse, os, subprocess, sys
from typing import List, Dict

sys.path.insert(0, os.path.dirname(__file__))
from reward_presets import get_preset_cli_args

SEEDS = [0, 1, 2]
TOTAL_TIMESTEPS = 5_000_000
BASE_OUTPUT = "experiments/results/init_alpha_layouts"
INIT_ALPHAS = [0.05, 0.2, 0.607, 2.0]
LAYOUTS = ["empty", "central_cross"]

SAC_HPARAMS = {
    "lr": 0.000225, "buffer_size": 100000, "batch_size": 256,
    "updates_per_step": 4, "warmup_steps": 5000,
}


def build_task_table() -> List[Dict]:
    tasks = []
    for layout in LAYOUTS:
        for ia in INIT_ALPHAS:
            for seed in SEEDS:
                tasks.append({
                    "name": f"{layout}_ia{ia}",
                    "layout": layout,
                    "init_alpha": ia,
                    "seed": seed,
                })
    return tasks


def run_task(task_id: int):
    tasks = build_task_table()
    if task_id < 0 or task_id >= len(tasks):
        print(f"Invalid task_id {task_id}. Valid: 0-{len(tasks)-1}")
        sys.exit(1)

    task = tasks[task_id]
    output_dir = os.path.join(BASE_OUTPUT, task["name"], f"seed_{task['seed']}")

    # R4_sparse preset, override layout
    preset_args = get_preset_cli_args("R4_sparse")
    filtered = []
    skip_next = False
    for arg in preset_args:
        if skip_next:
            skip_next = False
            continue
        if arg in ("--layout", "--hider-speed-mult"):
            skip_next = True
            continue
        filtered.append(arg)

    hp = SAC_HPARAMS
    cmd = [
        sys.executable, "trainer/train_selfplay_sac.py",
        "--timesteps", str(TOTAL_TIMESTEPS),
        "--num-envs", "64",
        "--layout", task["layout"],
        "--hider-speed-mult", "1.15",
        "--seed", str(task["seed"]),
        "--output-dir", output_dir,
        "--log-interval", "5000",
        "--save-interval", "500000",
        "--init-alpha", str(task["init_alpha"]),
        "--lr", str(hp["lr"]),
        "--buffer-size", str(hp["buffer_size"]),
        "--batch-size", str(hp["batch_size"]),
        "--updates-per-step", str(hp["updates_per_step"]),
        "--warmup-steps", str(hp["warmup_steps"]),
    ] + filtered

    print(f"Task {task_id}: {task['name']} seed={task['seed']}")
    print(f"  layout={task['layout']}, init_alpha={task['init_alpha']}")
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
            print(f"{i:3d}: {t['name']:<25s} layout={t['layout']:<15s} "
                  f"ia={t['init_alpha']:<6} seed={t['seed']}")
        print(f"\nTotal: {len(build_task_table())} tasks")
        return

    if args.task_id is not None:
        run_task(args.task_id)


if __name__ == "__main__":
    main()
