#!/usr/bin/env python3
"""
Design C main-grid task table.

12 PPO runs: 2 rewards × 2 A × 3 seeds. See pre-reg v2 §A3.

Usage:
  python experiments/design_c_grid_tasks.py --list
  python experiments/design_c_grid_tasks.py --task-id $SLURM_ARRAY_TASK_ID
  python experiments/design_c_grid_tasks.py --count
"""
import argparse
import itertools
import subprocess
import sys
from pathlib import Path

# ── Sweep axes (locked by pre-registration v2) ──────────────────────
REWARDS = ["R4_sparse", "R7_kitchen_sink"]
A_VALUES = [0.0, 0.5]
SEEDS = [0, 1, 2]

TIMESTEPS = 5_000_000
HSM = 1.15
LAYOUT = "four_corners"
OUTPUT_BASE = "experiments/results/design_c/grid"

# Reward CLI args, written explicitly (not via reward_presets import) so the
# task table is fully reproducible from this file alone — protects against
# silent drift in reward_presets.py.
REWARD_ARGS = {
    "R4_sparse": [
        "--seeker-time-penalty", "0.0",
        "--distance-reward-scale", "0.0",
        "--hider-dist-reward", "0.0",
        "--hider-abs-dist-reward", "0.0",
        "--runner-survival-bonus", "0.0",
    ],
    "R7_kitchen_sink": [
        "--seeker-time-penalty", "-0.015",
        "--seeker-escalating-urgency",
        "--distance-reward-scale", "0.2",
        "--hider-dist-reward", "0.14",
        "--hider-abs-dist-reward", "0.1",
        "--hider-wall-prox-penalty", "-0.02",
        "--hider-min-speed-reward", "0.005",
        "--area-coverage-bonus", "0.05",
        "--runner-survival-bonus", "0.01",
    ],
}


def build_task_table():
    """One dict per task, indexed by SLURM_ARRAY_TASK_ID."""
    tasks = []
    for reward, A, seed in itertools.product(REWARDS, A_VALUES, SEEDS):
        a_str = f"A{int(A * 100):02d}"
        task_name = f"{reward}/{a_str}/seed_{seed}"
        tasks.append(dict(
            task_name=task_name,
            reward=reward,
            A=A,
            latest_prob=round(1.0 - A, 2),  # PPO trainer uses (1 - A) as the live-partner probability
            seed=seed,
        ))
    return tasks


def run_task(task, dry_run=False):
    output_dir = f"{OUTPUT_BASE}/{task['task_name']}"

    cmd = [
        sys.executable, "trainer/train_zoo.py",
        "--timesteps", str(TIMESTEPS),
        "--seed", str(task["seed"]),
        "--layout", LAYOUT,
        "--hider-speed-mult", str(HSM),
        "--latest-prob", str(task["latest_prob"]),
        "--zoo-interval", "50",
        "--zoo-max-size", "50",
        "--sampling-strategy", "uniform",
        "--batch-size", "4096",
        "--num-envs", "64",
        "--output-dir", output_dir,
    ]
    cmd += REWARD_ARGS[task["reward"]]

    print(f"Task {task['task_name']}")
    print(f"  reward={task['reward']}  A={task['A']}  latest_prob={task['latest_prob']}  seed={task['seed']}")
    print(f"  cmd: {' '.join(cmd)}")

    if dry_run:
        print("  [DRY RUN]")
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(output_dir) / "train.log"

    with open(log_path, "w") as log:
        proc = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=str(Path(__file__).parent.parent),
        )

    status = "OK" if proc.returncode == 0 else f"FAIL({proc.returncode})"
    print(f"  Result: {status}")
    sys.exit(proc.returncode)


def main():
    parser = argparse.ArgumentParser(description="Design C grid task launcher")
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--count", action="store_true")
    args = parser.parse_args()

    tasks = build_task_table()

    if args.count:
        print(len(tasks))
        return

    if args.list:
        print(f"Total tasks: {len(tasks)}")
        print(f"{'ID':>3s}  {'reward':<18s} {'A':>4s}  {'seed':>4s}")
        print("-" * 36)
        for i, t in enumerate(tasks):
            print(f"{i:3d}  {t['reward']:<18s} {t['A']:4.1f}  {t['seed']:4d}")
        return

    if args.task_id is not None:
        if args.task_id < 0 or args.task_id >= len(tasks):
            print(f"ERROR: task-id {args.task_id} out of range [0, {len(tasks)-1}]",
                  file=sys.stderr)
            sys.exit(1)
        run_task(tasks[args.task_id], dry_run=args.dry_run)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
