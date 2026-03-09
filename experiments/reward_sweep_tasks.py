#!/usr/bin/env python3
"""
Generate task table for reward shaping sweep.

8 reward presets × 2 algorithms (PPO, SAC) × 3 seeds = 48 tasks

Usage:
  python experiments/reward_sweep_tasks.py --list
  python experiments/reward_sweep_tasks.py --task-id $SLURM_ARRAY_TASK_ID
  python experiments/reward_sweep_tasks.py --count
"""
import argparse
import itertools
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from reward_presets import PRESETS, get_preset_cli_args

# ── Sweep axes ──────────────────────────────────────────────────────
PRESET_NAMES = list(PRESETS.keys())
ALGORITHMS = ["ppo", "sac"]
SEEDS = [0, 1, 2]

TIMESTEPS = 5_000_000
OUTPUT_BASE = "experiments/results/reward_sweep"


def build_task_table():
    """Return list of dicts, one per task. Index = SLURM_ARRAY_TASK_ID."""
    tasks = []
    for preset, algo, seed in itertools.product(PRESET_NAMES, ALGORITHMS, SEEDS):
        task_name = f"{preset}/{algo}/seed_{seed}"
        tasks.append(dict(
            task_name=task_name,
            preset=preset,
            algo=algo,
            seed=seed,
        ))
    return tasks


def run_task(task, dry_run=False):
    """Launch training for a single task."""
    output_dir = f"{OUTPUT_BASE}/{task['task_name']}"

    # Select trainer script
    if task["algo"] == "ppo":
        script = "trainer/train_selfplay.py"
    else:
        script = "trainer/train_selfplay_sac.py"

    # Build command
    cmd = [
        sys.executable, script,
        "--timesteps", str(TIMESTEPS),
        "--seed", str(task["seed"]),
        "--output-dir", output_dir,
    ]
    # Add preset-specific reward args
    cmd += get_preset_cli_args(task["preset"])

    # SAC-specific defaults
    if task["algo"] == "sac":
        cmd += ["--warmup-steps", "10000"]
        cmd += ["--log-interval", "5000"]
        cmd += ["--save-interval", "100000"]

    # PPO-specific defaults
    if task["algo"] == "ppo":
        cmd += ["--batch-size", "4096"]
        cmd += ["--num-envs", "64"]

    print(f"Task {task['task_name']}")
    print(f"  preset={task['preset']}  algo={task['algo']}  seed={task['seed']}")
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
    parser = argparse.ArgumentParser(description="Reward sweep task launcher")
    parser.add_argument("--task-id", type=int, default=None,
                        help="SLURM_ARRAY_TASK_ID: run this single task")
    parser.add_argument("--list", action="store_true",
                        help="Print full task table and exit")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--count", action="store_true",
                        help="Print total task count and exit")
    args = parser.parse_args()

    tasks = build_task_table()

    if args.count:
        print(len(tasks))
        return

    if args.list:
        print(f"Total tasks: {len(tasks)}")
        print(f"{'ID':>4s}  {'preset':<22s} {'algo':<5s} {'seed':>4s}")
        print("-" * 42)
        for i, t in enumerate(tasks):
            print(f"{i:4d}  {t['preset']:<22s} {t['algo']:<5s} {t['seed']:4d}")
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
