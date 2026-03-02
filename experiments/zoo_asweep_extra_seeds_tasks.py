#!/usr/bin/env python3
"""
Extra seeds (3-9) for the zoo A-sweep.

Factorial design (same grid, new seeds only):
  20 game configs (4 STP × 5 HSM)
  × 7 A values
  × 2 sampling modes (uniform, thompson_loss)
  × 7 NEW seeds (3-9)
  = 1960 tasks

Usage:
  python experiments/zoo_asweep_extra_seeds_tasks.py --list
  python experiments/zoo_asweep_extra_seeds_tasks.py --task-id $SLURM_ARRAY_TASK_ID
  python experiments/zoo_asweep_extra_seeds_tasks.py --task-id 0 --dry-run
"""
import argparse
import itertools
import subprocess
import sys
from pathlib import Path


# ── Sweep axes (same grid as original, only seeds differ) ──────────
STPS = [0.005, 0.01, 0.02, 0.05]
HSMS = [1.0, 1.05, 1.1, 1.15, 1.2]
A_VALUES = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 0.9]
SAMPLING_MODES = ["uniform", "thompson_loss"]
SEEDS = [3, 4, 5, 6, 7, 8, 9]

TIMESTEPS = 10_000_000
LAYOUT = "four_corners"
OUTPUT_BASE = "experiments/results/zoo_asweep"


def build_task_table():
    """Return list of dicts, one per task. Index = SLURM_ARRAY_TASK_ID."""
    tasks = []
    for stp, hsm, A, sampling, seed in itertools.product(
        STPS, HSMS, A_VALUES, SAMPLING_MODES, SEEDS
    ):
        stp_str = f"{stp:.4f}".replace("0.", "").rstrip("0") or "0"
        hsm_str = f"{round(hsm * 100)}"
        config_name = f"STP{stp_str}_HSM{hsm_str}"
        a_str = f"A{int(A * 100):02d}"
        task_name = f"{config_name}/{a_str}_{sampling}/seed_{seed}"
        tasks.append(dict(
            task_name=task_name,
            config_name=config_name,
            stp=stp,
            hsm=hsm,
            A=A,
            latest_prob=round(1.0 - A, 2),
            sampling=sampling,
            seed=seed,
        ))
    return tasks


def run_task(task, dry_run=False):
    """Launch train_zoo.py for a single task."""
    output_dir = f"{OUTPUT_BASE}/{task['task_name']}"

    cmd = [
        sys.executable,
        "trainer/train_zoo.py",
        "--timesteps", str(TIMESTEPS),
        "--latest-prob", str(task["latest_prob"]),
        "--seeker-time-penalty", str(-task["stp"]),
        "--hider-speed-mult", str(task["hsm"]),
        "--sampling-strategy", task["sampling"],
        "--layout", LAYOUT,
        "--seed", str(task["seed"]),
        "--output-dir", output_dir,
    ]

    print(f"Task {task['task_name']}")
    print(f"  A={task['A']:.2f}  STP={task['stp']}  HSM={task['hsm']}  "
          f"sampling={task['sampling']}  seed={task['seed']}")
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
    parser = argparse.ArgumentParser(description="Zoo A-sweep extra seeds (3-9)")
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
        print(f"{'ID':>4s}  {'config':<16s} {'A':>4s}  {'sampling':<14s} {'seed':>4s}")
        print("-" * 52)
        for i, t in enumerate(tasks):
            print(f"{i:4d}  {t['config_name']:<16s} {t['A']:4.2f}  "
                  f"{t['sampling']:<14s} {t['seed']:4d}")
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
