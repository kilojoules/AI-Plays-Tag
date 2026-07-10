#!/usr/bin/env python3
"""
HSM flank — is the R7/A=0 bad basin specific to HSM=1.15?

The idea-critic's strongest unstated-claim concern: every Design C run uses
hider_speed_mult=1.15 in four_corners. The bimodal collapse we attribute to
`area_coverage_bonus` might instead be a property of the HSM=1.15 boundary
regime, where small reward perturbations cross a stability frontier. A
referee can argue "you've discovered HSM=1.15 is brittle, not that coverage
is bad."

This flank trains R7_kitchen_sink at A=0 with HSM 1.05 (easier for the
seeker) and HSM 1.20 (harder, but known-trainable from the earlier
STP x HSM study). If the bimodal collapse appears at all three HSMs, the
basin is a property of the reward; if it exists only at 1.15, the
coverage story needs an HSM-contingency caveat.

Evaluation note: the Design C anchors were trained at HSM=1.15, so the
matched-eval is not the right read-out here. Use the within-cell
round-robin (each seeker vs the 8 same-cell hiders at the cell's native
HSM) — see design_c_hsm_roundrobin.py.

Design: 16 runs — R7_kitchen_sink, A=0, HSM in {1.05, 1.20}, seeds 0-7.
8 seeds per cell (the A-sweep taught us n=5 is too noisy for
bimodality-rate claims). NOT a Design C prereg amendment.

Results: experiments/results/design_c/hsm_flank/

Usage:
  python experiments/design_c_hsm_flank_tasks.py --list
  python experiments/design_c_hsm_flank_tasks.py --task-id $SLURM_ARRAY_TASK_ID
"""
import argparse
import itertools
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from design_c_grid_tasks import REWARD_ARGS, TIMESTEPS, LAYOUT

HSM_VALUES = [1.05, 1.20]
SEEDS = list(range(8))
A_VALUE = 0.0
OUTPUT_BASE = "experiments/results/design_c/hsm_flank"
REWARD = "R7_kitchen_sink"


def build_task_table():
    tasks = []
    for hsm, seed in itertools.product(HSM_VALUES, SEEDS):
        hsm_str = f"HSM{int(round(hsm * 100)):03d}"
        tasks.append(dict(
            task_name=f"{REWARD}/{hsm_str}/seed_{seed}",
            hsm=hsm,
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
        "--hider-speed-mult", str(task["hsm"]),
        "--latest-prob", str(task["latest_prob"]),
        "--zoo-interval", "50", "--zoo-max-size", "50",
        "--sampling-strategy", "uniform",
        "--batch-size", "4096", "--num-envs", "64",
        "--output-dir", output_dir,
    ] + REWARD_ARGS[REWARD]

    print(f"HSM-flank task {task['task_name']}")
    print(f"  hsm={task['hsm']}  latest_prob={task['latest_prob']}  seed={task['seed']}")
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
