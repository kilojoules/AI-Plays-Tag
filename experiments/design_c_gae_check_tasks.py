#!/usr/bin/env python3
"""
Design C GAE-fix bounding runs (results.md §18.6).

Every Design C run through the v3 power extension trained with the GAE
done-mask off-by-one (terminal transitions bootstrapped γ·V(next
episode's reset state); see trainer/train_zoo.py). The bug was shared
across arms, but its magnitude scales with |V(reset)| — largest under
dense reward — so it is a candidate artifactual contributor to the
R7-arm variance findings.

Design
------
10 runs with the CORRECTED GAE (no --legacy-gae):
  R4_sparse/A=0  seeds 0-4  and  R7_kitchen_sink/A=0  seeds 0-4

Output labels get a _gaefix suffix so anchor-panel rows never mix with
the legacy cells. Comparison of the anchor-mean WR distribution and the
healthy/over-spec/basin mix against the same-seed legacy runs bounds the
bug's contribution: if the R7 mix shifts materially, the taxonomy needs
a GAE-sensitivity footnote and the σ claims a partial re-run.

Usage:
  python experiments/design_c_gae_check_tasks.py --list
  python experiments/design_c_gae_check_tasks.py --task-id $LSB_TASK_ID
  python experiments/design_c_gae_check_tasks.py --count
"""
import argparse
import itertools
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from design_c_grid_tasks import REWARD_ARGS, TIMESTEPS, HSM, LAYOUT

REWARDS = ["R4_sparse", "R7_kitchen_sink"]
A_VALUE = 0.0
SEEDS = [0, 1, 2, 3, 4]
OUTPUT_BASE = "experiments/results/design_c/gae_check"


def build_task_table():
    tasks = []
    for reward, seed in itertools.product(REWARDS, SEEDS):
        a_str = f"A{int(A_VALUE * 100):02d}"
        tasks.append(dict(
            task_name=f"{reward}_gaefix/{a_str}/seed_{seed}",
            reward=reward,
            latest_prob=round(1.0 - A_VALUE, 2),
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

    print(f"Task {task['task_name']}  (corrected GAE)")
    print(f"  cmd: {' '.join(cmd)}")

    if dry_run:
        print("  [DRY RUN]")
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(output_dir) / "train.log"

    with open(log_path, "w") as log:
        proc = subprocess.run(
            cmd, stdout=log, stderr=subprocess.STDOUT,
            cwd=str(Path(__file__).parent.parent),
        )

    status = "OK" if proc.returncode == 0 else f"FAIL({proc.returncode})"
    print(f"  Result: {status}")
    sys.exit(proc.returncode)


def main():
    parser = argparse.ArgumentParser(description="Design C GAE-fix bounding tasks")
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
        for i, t in enumerate(tasks):
            print(f"{i:3d}  {t['task_name']}")
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
