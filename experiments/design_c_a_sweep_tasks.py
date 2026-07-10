#!/usr/bin/env python3
"""
A-curve rescue characterization — R7_kitchen_sink across small A values.

The 2x2 matched eval (`ablation_eval.csv`) showed A=0.5 fully rescues R7
from the coverage-induced basin (4/9 collapsed -> 0/9). The natural reviewer
question is "why A=0.5 — what about much smaller A?" A small amount of
opponent diversity may suffice if the rescue is about preventing
over-commitment to the live partner rather than providing gradient signal.

This sweep characterizes the rescue saturation curve at R7_kitchen_sink
only (the cell where A actually matters; R7_no_coverage is already healthy
at A=0). If A=0.05 already rescues, the paper's headline upgrades from
"either intervention fixes it" to "5% zoo exposure is sufficient."

NOT a Design C prereg amendment — this is a diagnostic curve folded into
the existing paper as a saturation figure.

Design
------
15 runs: reward = R7_kitchen_sink, A in {0.05, 0.10, 0.25}, seeds 0-4.
Task ordering puts A=0.05 first (task ids 0-4) so the throttled array
(0-14%3) returns the most informative cell soonest.

Seeds 0-4 are paired with the existing R7_kitchen_sink grid at A=0 and
A=0.5 (same seed -> same init), so the rescue curve is a single-seed
trajectory across A.

Results: experiments/results/design_c/a_sweep/

Usage:
  python experiments/design_c_a_sweep_tasks.py --list
  python experiments/design_c_a_sweep_tasks.py --task-id $SLURM_ARRAY_TASK_ID
"""
import argparse
import itertools
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from design_c_grid_tasks import REWARD_ARGS, TIMESTEPS, HSM, LAYOUT

# Order matters: A=0.05 first so it returns first under the array throttle.
A_VALUES = [0.05, 0.10, 0.25]
SEEDS = list(range(5))
OUTPUT_BASE = "experiments/results/design_c/a_sweep"
REWARD = "R7_kitchen_sink"


def build_task_table():
    tasks = []
    for A, seed in itertools.product(A_VALUES, SEEDS):
        a_str = f"A{int(round(A * 100)):02d}"
        tasks.append(dict(
            task_name=f"{REWARD}/{a_str}/seed_{seed}",
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
        "--zoo-interval", "50", "--zoo-max-size", "50",
        "--sampling-strategy", "uniform",
        "--batch-size", "4096", "--num-envs", "64",
        "--output-dir", output_dir,
    ] + REWARD_ARGS[REWARD]

    print(f"A-sweep task {task['task_name']}")
    print(f"  A={task['A']}  latest_prob={task['latest_prob']}  seed={task['seed']}")
    print(f"  cmd: {' '.join(cmd)}")

    if dry_run:
        print("  [DRY RUN]")
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(output_dir) / "train.log"
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                              cwd=str(Path(__file__).parent.parent))
    print(f"  Result: {'OK' if proc.returncode == 0 else 'FAIL(' + str(proc.returncode) + ')'}")
    sys.exit(proc.returncode)


def main():
    p = argparse.ArgumentParser(description="A-sweep task launcher")
    p.add_argument("--task-id", type=int, default=None)
    p.add_argument("--list", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--count", action="store_true")
    args = p.parse_args()

    tasks = build_task_table()

    if args.count:
        print(len(tasks))
        return

    if args.list:
        print(f"Total tasks: {len(tasks)}")
        for i, t in enumerate(tasks):
            print(f"  {i:2d}  {t['task_name']}")
        return

    if args.task_id is not None:
        if not 0 <= args.task_id < len(tasks):
            print(f"ERROR: task-id {args.task_id} out of range [0, {len(tasks)-1}]",
                  file=sys.stderr)
            sys.exit(1)
        run_task(tasks[args.task_id], dry_run=args.dry_run)
        return

    p.print_help()


if __name__ == "__main__":
    main()
