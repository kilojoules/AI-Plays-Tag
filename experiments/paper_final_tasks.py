#!/usr/bin/env python3
"""
Task generator for final paper experiments.

Experiment 1: Geometry study (more layouts)
  9 layouts x 3 seeds = 27 tasks (IDs 0-26)
  Tests corner camping vs obstacle proximity to corners.

Experiment 2: Buffer diversity (already logged — uses same runs as Exp 1)
  No extra tasks needed. Buffer diversity metrics are now logged in
  every SAC training run via metrics.csv columns.

Experiment 3: Init-alpha ablation
  4 init_alpha values x 3 seeds = 12 tasks (IDs 27-38)
  Tests whether the bootstrapping window has a minimum duration.

All use R4_sparse, SAC selfplay, Optuna-optimized hyperparameters,
HSM=1.15, 5M steps.

Total: 39 tasks.
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
BASE_OUTPUT = "experiments/results/paper_final"

SAC_HPARAMS = {
    "lr": 0.000225,
    "init_alpha": 0.607,
    "buffer_size": 100000,
    "batch_size": 256,
    "updates_per_step": 4,
    "warmup_steps": 5000,
}

# Experiment 1: All layouts
# Existing layouts we already have data for: four_corners (from ablation study)
# New layouts to test:
GEOMETRY_LAYOUTS = [
    "empty", "one_corner", "two_corners", "central_cross",
    "wall_midpoints", "four_corners", "corner_tight",
    "center_cluster", "playground",
]

# Experiment 3: Init-alpha values
INIT_ALPHAS = [0.05, 0.2, 0.607, 2.0]


def build_task_table() -> List[Dict]:
    tasks = []

    # Experiment 1: Geometry study (27 tasks)
    for layout in GEOMETRY_LAYOUTS:
        for seed in SEEDS:
            tasks.append({
                "experiment": "geometry",
                "name": f"geo_{layout}",
                "layout": layout,
                "init_alpha": SAC_HPARAMS["init_alpha"],
                "fixed_alpha": None,
                "seed": seed,
            })

    # Experiment 3: Init-alpha ablation (12 tasks)
    for init_alpha in INIT_ALPHAS:
        for seed in SEEDS:
            tasks.append({
                "experiment": "init_alpha",
                "name": f"initalpha_{init_alpha}",
                "layout": "four_corners",
                "init_alpha": init_alpha,
                "fixed_alpha": None,
                "seed": seed,
            })

    return tasks


def run_task(task_id: int):
    tasks = build_task_table()
    if task_id < 0 or task_id >= len(tasks):
        print(f"Invalid task_id {task_id}. Valid range: 0-{len(tasks)-1}")
        sys.exit(1)

    task = tasks[task_id]
    experiment = task["experiment"]
    name = task["name"]
    layout = task["layout"]
    init_alpha = task["init_alpha"]
    fixed_alpha = task["fixed_alpha"]
    seed = task["seed"]

    output_dir = os.path.join(BASE_OUTPUT, experiment, name, f"seed_{seed}")

    # R4_sparse args, but override layout
    preset_args = get_preset_cli_args("R4_sparse")
    # Remove --layout and --hider-speed-mult from preset (we set them explicitly)
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
    hp_args = [
        "--lr", str(hp["lr"]),
        "--buffer-size", str(hp["buffer_size"]),
        "--batch-size", str(hp["batch_size"]),
        "--updates-per-step", str(hp["updates_per_step"]),
        "--warmup-steps", str(hp["warmup_steps"]),
    ]

    alpha_args = ["--init-alpha", str(init_alpha)]
    if fixed_alpha is not None:
        alpha_args += ["--fixed-alpha", str(fixed_alpha)]

    cmd = [
        sys.executable, "trainer/train_selfplay_sac.py",
        "--timesteps", str(TOTAL_TIMESTEPS),
        "--num-envs", "64",
        "--layout", layout,
        "--hider-speed-mult", "1.15",
        "--seed", str(seed),
        "--output-dir", output_dir,
        "--log-interval", "5000",
        "--save-interval", "500000",
    ] + filtered + hp_args + alpha_args

    print(f"Task {task_id}: {experiment}/{name} / seed={seed}")
    print(f"  layout={layout}, init_alpha={init_alpha}, fixed_alpha={fixed_alpha}")
    print(f"  Output: {output_dir}")
    print(f"  Command: {' '.join(cmd)}")
    print()

    os.makedirs(output_dir, exist_ok=True)
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Final paper experiments")
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--experiment", type=str, default=None,
                        choices=["geometry", "init_alpha"])
    args = parser.parse_args()

    tasks = build_task_table()

    if args.list:
        for i, t in enumerate(tasks):
            if args.experiment and t["experiment"] != args.experiment:
                continue
            print(f"{i:3d}: {t['experiment']:<12s} {t['name']:<25s} "
                  f"layout={t['layout']:<15s} init_a={t['init_alpha']:.3f} seed={t['seed']}")
        total = len(tasks) if not args.experiment else sum(
            1 for t in tasks if t["experiment"] == args.experiment)
        print(f"\nTotal: {total} tasks")
        return

    if args.task_id is not None:
        run_task(args.task_id)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
