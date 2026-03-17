#!/usr/bin/env python3
"""
Task generator for FR (Forgetting Regret) vs A (zoo mixing rate) sweep.

Experiment design:
  5 reward presets x 2 algorithms (PPO, SAC) x 5 A values x 3 seeds = 150 tasks
  Single game config: STP=0.01, HSM=1.15, layout=four_corners

Each task trains a zoo agent with a specific (preset, algorithm, A, seed) combo.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Dict, List

# Reward presets to use (subset of 8 — chosen for diversity)
REWARD_PRESETS = ["R0_baseline", "R1_seeker_pursuit", "R3_both_shaped",
                  "R4_sparse", "R5_escalating"]

ALGORITHMS = ["ppo", "sac"]

A_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]

SEEDS = [0, 1, 2]

# Single game config
GAME_CONFIG = {
    "layout": "four_corners",
    "hider_speed_mult": 1.15,
    "seeker_time_penalty": 0.01,  # STP=0.01 (note: will be overridden by preset)
}

TOTAL_TIMESTEPS = 5_000_000
BASE_OUTPUT = "experiments/results/fr_sweep_v2"

# Optuna-optimized hyperparameters (from ppo_hpo_v5 / sac_hpo_v3)
PPO_HPARAMS = {
    "lr": 0.000672,
    "gamma": 0.957,
    "gae_lambda": 0.902,
    "clip_ratio": 0.162,
    "train_iters": 20,
    "batch_size": 4096,
    "entropy_coef": 0.014,
    "target_kl": 0.022,
}
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


def build_task_table() -> List[Dict]:
    """Build the full task table (150 tasks)."""
    tasks = []
    for preset in REWARD_PRESETS:
        for algo in ALGORITHMS:
            for a_val in A_VALUES:
                for seed in SEEDS:
                    task = {
                        "preset": preset,
                        "algorithm": algo,
                        "a_value": a_val,
                        "seed": seed,
                    }
                    tasks.append(task)
    return tasks


def get_preset_reward_args(preset_name: str) -> list:
    """Get CLI args for reward shaping from a preset name."""
    # Import here to avoid path issues when called as script
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from reward_presets import PRESETS

    cfg = PRESETS[preset_name]
    args = []

    mapping = {
        "seeker_time_penalty": "--seeker-time-penalty",
        "distance_reward_scale": "--distance-reward-scale",
        "hider_dist_reward_scale": "--hider-dist-reward",
        "hider_abs_dist_reward_scale": "--hider-abs-dist-reward",
        "hider_wall_prox_penalty": "--hider-wall-prox-penalty",
        "hider_min_speed_reward": "--hider-min-speed-reward",
        "area_coverage_bonus": "--area-coverage-bonus",
        "runner_survival_bonus": "--runner-survival-bonus",
        "seeker_escalating_urgency": "--seeker-escalating-urgency",
    }

    for key, cli_flag in mapping.items():
        if key in cfg:
            val = cfg[key]
            if isinstance(val, bool):
                if val:
                    args.append(cli_flag)
            else:
                args += [cli_flag, str(val)]

    return args


def run_task(task_id: int):
    """Run a single task by its ID."""
    tasks = build_task_table()
    if task_id < 0 or task_id >= len(tasks):
        print(f"Invalid task_id {task_id}. Valid range: 0-{len(tasks)-1}")
        sys.exit(1)

    task = tasks[task_id]
    preset = task["preset"]
    algo = task["algorithm"]
    a_val = task["a_value"]
    seed = task["seed"]

    # Build output directory
    output_dir = os.path.join(BASE_OUTPUT, preset, f"A{a_val:.2f}_{algo}",
                              f"seed_{seed}")

    # Choose trainer script and optimized hyperparameters
    if algo == "ppo":
        script = "trainer/train_zoo.py"
        a_flag = ["--latest-prob", str(1.0 - a_val)]  # PPO uses latest_prob = 1 - A
        hp = PPO_HPARAMS
        hp_args = [
            "--lr", str(hp["lr"]),
            "--gamma", str(hp["gamma"]),
            "--gae-lambda", str(hp["gae_lambda"]),
            "--clip-ratio", str(hp["clip_ratio"]),
            "--train-iters", str(hp["train_iters"]),
            "--batch-size", str(hp["batch_size"]),
            "--entropy-coef", str(hp["entropy_coef"]),
            "--target-kl", str(hp["target_kl"]),
        ]
    else:
        script = "trainer/train_zoo_sac.py"
        a_flag = ["--zoo-prob", str(a_val)]  # SAC uses zoo_prob = A
        hp = SAC_HPARAMS
        hp_args = [
            "--lr", str(hp["lr"]),
            "--gamma", str(hp["gamma"]),
            "--tau", str(hp["tau"]),
            "--init-alpha", str(hp["init_alpha"]),
            "--buffer-size", str(hp["buffer_size"]),
            "--batch-size", str(hp["batch_size"]),
            "--updates-per-step", str(hp["updates_per_step"]),
            "--warmup-steps", str(hp["warmup_steps"]),
        ]

    # Build command
    cmd = [
        sys.executable, script,
        "--timesteps", str(TOTAL_TIMESTEPS),
        "--num-envs", "64",
        "--layout", GAME_CONFIG["layout"],
        "--hider-speed-mult", str(GAME_CONFIG["hider_speed_mult"]),
        "--seed", str(seed),
        "--output-dir", output_dir,
    ] + a_flag + hp_args + get_preset_reward_args(preset)

    print(f"Task {task_id}: {preset} / {algo} / A={a_val} / seed={seed}")
    print(f"  Output: {output_dir}")
    print(f"  Command: {' '.join(cmd)}")
    print()

    os.makedirs(output_dir, exist_ok=True)
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--list", action="store_true",
                        help="List all tasks and exit")
    args = parser.parse_args()

    if args.list:
        tasks = build_task_table()
        for i, t in enumerate(tasks):
            print(f"{i:3d}: {t['preset']:20s} {t['algorithm']:3s} A={t['a_value']:.2f} seed={t['seed']}")
        print(f"\nTotal: {len(tasks)} tasks")
        return

    run_task(args.task_id)


if __name__ == "__main__":
    main()
