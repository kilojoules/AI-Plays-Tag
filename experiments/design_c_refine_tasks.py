#!/usr/bin/env python3
"""
Design C REFINE round: extend the main grid with two new seeds (3, 4) to
go from 3 seeds/cell to 5 seeds/cell.

Triggered by pre-reg v2 §A6 REFINE branch on the first analysis (β_RA = 0.81
with cluster-robust CI bracketing 0). Adding 2 new seeds × 2 rewards × 2 A
= 8 runs.

(Note: pre-reg v2 §A6 records "+6 runs" as a back-of-envelope arithmetic
slip; the actual minimum coherent extension is 8 runs to keep all 4 cells
balanced. Pre-reg v1's original budget was correct (12 runs for 3 new
seeds), and we are choosing the cheaper 2-seed extension first.)

Usage:
  python experiments/design_c_refine_tasks.py --list
  python experiments/design_c_refine_tasks.py --task-id $SLURM_ARRAY_TASK_ID
"""
import argparse
import itertools
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from design_c_grid_tasks import REWARD_ARGS, REWARDS, A_VALUES, TIMESTEPS, HSM, LAYOUT

REFINE_SEEDS = [3, 4]
OUTPUT_BASE = "experiments/results/design_c/grid"


def build_task_table():
    tasks = []
    for reward, A, seed in itertools.product(REWARDS, A_VALUES, REFINE_SEEDS):
        a_str = f"A{int(A * 100):02d}"
        task_name = f"{reward}/{a_str}/seed_{seed}"
        tasks.append(dict(
            task_name=task_name,
            reward=reward,
            A=A,
            latest_prob=round(1.0 - A, 2),
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

    print(f"Refine task {task['task_name']}")
    print(f"  reward={task['reward']}  A={task['A']}  seed={task['seed']}")

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--count", action="store_true")
    args = parser.parse_args()

    tasks = build_task_table()

    if args.count:
        print(len(tasks)); return
    if args.list:
        for i, t in enumerate(tasks):
            print(f"  {i}  {t['task_name']}")
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
