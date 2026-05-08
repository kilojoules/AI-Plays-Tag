#!/usr/bin/env python3
"""
Design C anchor task table — 4 fixed reference policies at the corners
of the grid, trained at seed=42 (a seed not used in the main grid).

See pre-reg v2 §A4. These run independently of the main grid and form
part of the cross-evaluation pool to break self-referential structure.

Usage:
  python experiments/design_c_anchors_tasks.py --list
  python experiments/design_c_anchors_tasks.py --task-id $SLURM_ARRAY_TASK_ID
"""
import argparse
import itertools
import subprocess
import sys
from pathlib import Path

# Reuse the inline reward args from the grid task file.
sys.path.insert(0, str(Path(__file__).parent))
from design_c_grid_tasks import REWARD_ARGS, REWARDS, A_VALUES, TIMESTEPS, HSM, LAYOUT

ANCHOR_SEED = 42
OUTPUT_BASE = "experiments/results/design_c/anchors"


def build_task_table():
    tasks = []
    for reward, A in itertools.product(REWARDS, A_VALUES):
        a_str = f"A{int(A * 100):02d}"
        task_name = f"{reward}/{a_str}/seed_{ANCHOR_SEED}"
        tasks.append(dict(
            task_name=task_name,
            reward=reward,
            A=A,
            latest_prob=round(1.0 - A, 2),
            seed=ANCHOR_SEED,
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

    print(f"Anchor task {task['task_name']}")
    print(f"  reward={task['reward']}  A={task['A']}  latest_prob={task['latest_prob']}  seed={task['seed']}")

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
    parser = argparse.ArgumentParser(description="Design C anchors launcher")
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
        print(f"Total anchor tasks: {len(tasks)}")
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
