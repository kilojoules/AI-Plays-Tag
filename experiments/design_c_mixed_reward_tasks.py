#!/usr/bin/env python3
"""
Design C mixed-reward deconfound cells: R7sk_R4hd (results.md §18.5).

The problem being addressed
---------------------------
"Reward shaping" as manipulated in Design C changes the HIDER's reward
too (R7 carries five hider-side terms plus the both-roles coverage
bonus; R4's hider sees only terminal rewards), and seeker and hider
co-train in the same run. Every R4-vs-R7 seeker contrast is therefore
also an opponent-quality / zoo-curriculum-quality contrast: both
headline effects (β_RA > 0 and the R7 σ inflation) admit an
opponent-side explanation with no seeker-side shaping story at all.

Design
------
20 runs: reward = R7sk_R4hd, A ∈ {0.0, 0.5}, seeds 0–9.

R7sk_R4hd = R7's seeker-side terms with R4's terminal-only hider:
  seeker: --seeker-time-penalty -0.015, --seeker-escalating-urgency,
          --distance-reward-scale 0.2
  hider:  all shaping zeroed (R4-identical)
  --area-coverage-bonus excluded: it is a both-roles term that cannot be
  given to one side only, and §12 shows no-coverage cells are healthier
  anyway. The all-R7 comparator at fixed seeker shaping is therefore
  R7_no_coverage (which keeps the hider terms), giving the triangle:

    R4_sparse (n=20/cell)       sparse seeker + sparse hider
    R7sk_R4hd (this file)       shaped seeker + sparse hider
    R7_no_coverage (n=10/5)     shaped seeker + shaped hider

  R4↔R7sk_R4hd isolates seeker-side shaping at fixed opponent reward;
  R7sk_R4hd↔R7_no_coverage isolates hider-side shaping at fixed seeker
  shaping.

Trains with --legacy-gae: every existing Design C run used the
pre-2026-07 GAE mask (see §18.6), so comparability requires it here.
Seeds 0–9 are paired seed-for-seed with the grid and ablation cells.

Usage:
  python experiments/design_c_mixed_reward_tasks.py --list
  python experiments/design_c_mixed_reward_tasks.py --task-id $LSB_TASK_ID
  python experiments/design_c_mixed_reward_tasks.py --count
"""
import argparse
import itertools
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from design_c_grid_tasks import TIMESTEPS, HSM, LAYOUT

REWARD_LABEL = "R7sk_R4hd"
A_VALUES = [0.0, 0.5]
SEEDS = list(range(10))
OUTPUT_BASE = "experiments/results/design_c/mixed_reward"

# R7 seeker-side terms verbatim from design_c_grid_tasks.REWARD_ARGS
# ["R7_kitchen_sink"]; every hider-side term explicitly zeroed to R4's
# values; coverage (both-roles) zeroed — see module docstring.
R7SK_R4HD_ARGS = [
    "--seeker-time-penalty", "-0.015",
    "--seeker-escalating-urgency",
    "--distance-reward-scale", "0.2",
    "--hider-dist-reward", "0.0",
    "--hider-abs-dist-reward", "0.0",
    "--hider-wall-prox-penalty", "0.0",
    "--hider-min-speed-reward", "0.0",
    "--area-coverage-bonus", "0.0",
    "--runner-survival-bonus", "0.0",
]


def build_task_table():
    tasks = []
    for A, seed in itertools.product(A_VALUES, SEEDS):
        a_str = f"A{int(A * 100):02d}"
        tasks.append(dict(
            task_name=f"{REWARD_LABEL}/{a_str}/seed_{seed}",
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
        "--legacy-gae",
        "--output-dir", output_dir,
    ]
    cmd += R7SK_R4HD_ARGS

    print(f"Task {task['task_name']}")
    print(f"  A={task['A']}  latest_prob={task['latest_prob']}  seed={task['seed']}")
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
    parser = argparse.ArgumentParser(description="Design C mixed-reward deconfound tasks")
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
