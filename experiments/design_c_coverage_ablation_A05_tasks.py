#!/usr/bin/env python3
"""
Coverage-bonus ablation at A=0.5 — the identifying experiment for the
de-confounded substitution claim. NOT part of pre-registered Design C.

Motivation
----------
The coverage ablation at A=0.0 (job 18766646, see
`design_c_coverage_ablation_tasks.py`) cut the bad-basin rate from 4/9 to
1/10 by removing `--area-coverage-bonus`. The idea-critic flagged that the
original β_RA (reward × A interaction) measured in the pre-registered grid
is confounded: most of the interaction signal may be A=0.5 *rescuing* R7
from the coverage-induced basin, rather than a genuine substitution effect.

The identifying experiment: re-run the R7_no_coverage cell at A=0.5 and
compare β_RA between (R7_kitchen_sink, R7_no_coverage) grids.
- If β_RA shrinks toward zero under R7_no_coverage, the original
  interaction was dominated by coverage-rescue.
- If β_RA survives at similar magnitude, the substitution effect is real
  and not coverage-mediated.

Design
------
5 runs: reward = R7_no_coverage, A = 0.5, seeds 0-4. Paired seed-for-seed
with the existing experiments/results/design_c/grid/R7_kitchen_sink/A50/
seed_{0..4} baseline (same seed → same init), so the comparison isolates
the coverage term at A=0.5.

Results: experiments/results/design_c/coverage_ablation/R7_no_coverage/A50/

Usage:
  python experiments/design_c_coverage_ablation_A05_tasks.py --list
  python experiments/design_c_coverage_ablation_A05_tasks.py --task-id $SLURM_ARRAY_TASK_ID
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from design_c_grid_tasks import REWARD_ARGS, TIMESTEPS, HSM, LAYOUT
from design_c_coverage_ablation_tasks import _drop_flag, OUTPUT_BASE

A_VALUE = 0.5
SEEDS = list(range(5))

R7_NO_COVERAGE_ARGS = _drop_flag(
    list(REWARD_ARGS["R7_kitchen_sink"]), "--area-coverage-bonus"
)


def build_task_table():
    tasks = []
    for seed in SEEDS:
        a_str = f"A{int(A_VALUE * 100):02d}"
        tasks.append(dict(
            task_name=f"R7_no_coverage/{a_str}/seed_{seed}",
            A=A_VALUE,
            latest_prob=round(1.0 - A_VALUE, 2),  # 0.5 at A=0.5
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
    ] + R7_NO_COVERAGE_ARGS

    print(f"Coverage-ablation A=0.5 task {task['task_name']}")
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
    p = argparse.ArgumentParser(description="Coverage-ablation A=0.5 task launcher")
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
        print(f"Reward args: {' '.join(R7_NO_COVERAGE_ARGS)}")
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
