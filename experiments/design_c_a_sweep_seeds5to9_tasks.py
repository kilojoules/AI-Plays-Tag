#!/usr/bin/env python3
"""
A-curve rescue characterization — extension to seeds 5..9.

First wave (`design_c_a_sweep_tasks.py`, job 18908025) ran seeds 0..4 at
A in {0.05, 0.10, 0.25} for R7_kitchen_sink and revealed a non-monotone
U-shape: small A (0.05, 0.10) gave WORSE matched-eval performance against
the R4 anchor than A=0, with the trough at A=0.10 (4/5 collapsed).

The n=5 per cell leaves wide CIs on the collapse rates. This extension
adds seeds 5..9 at the same three A values (15 more runs) to firm up the
U-shape for publication-grade error bars.

NOT a Design C prereg amendment.

Design
------
15 runs: reward = R7_kitchen_sink, A in {0.05, 0.10, 0.25}, seeds 5..9.
Task ordering matches the first wave: A=0.05 fills 0-4, A=0.10 fills 5-9,
A=0.25 fills 10-14. Seeds 5..9 pair with the existing R7_kitchen_sink
grid (which had seeds 0..8 at A=0 and A=0.5).

Results land in the same tree: experiments/results/design_c/a_sweep/

Usage:
  python experiments/design_c_a_sweep_seeds5to9_tasks.py --list
  python experiments/design_c_a_sweep_seeds5to9_tasks.py --task-id $SLURM_ARRAY_TASK_ID
"""
import argparse
import itertools
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from design_c_grid_tasks import REWARD_ARGS, TIMESTEPS, HSM, LAYOUT
from design_c_a_sweep_tasks import A_VALUES, OUTPUT_BASE, REWARD

SEEDS = list(range(5, 10))


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

    print(f"A-sweep (seeds 5..9) task {task['task_name']}")
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
    p = argparse.ArgumentParser()
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
