#!/usr/bin/env python3
"""
Urgency-ONLY ablation — completes the coverage x urgency 2x2 factorial at A=0.

The idea-critic flagged that `R7_no_cov_urg` (which drops BOTH
--area-coverage-bonus and --seeker-escalating-urgency) cannot localize the
"urgency is protective" effect to urgency: the 3/10-vs-1/10 contrast against
R7_no_coverage is confounded by coverage's absence and is not significant
(Fisher p~0.58) anyway.

This cell drops ONLY --seeker-escalating-urgency and KEEPS coverage,
completing the factorial:

                       urgency ON           urgency OFF
  coverage ON          R7_kitchen_sink      R7_no_urgency   <- THIS CELL
  coverage OFF         R7_no_coverage       R7_no_cov_urg

With all four cells at A=0 (~9-10 seeds each) a logistic model with
coverage_removed x urgency_removed terms gives the properly identified
urgency effect.

Design: 10 runs — R7_no_urgency, A=0, seeds 0-9, paired seed-for-seed with
the other three cells. NOT a Design C prereg amendment.

Results: experiments/results/design_c/urgency_only_ablation/

Usage:
  python experiments/design_c_urgency_only_tasks.py --list
  python experiments/design_c_urgency_only_tasks.py --task-id $SLURM_ARRAY_TASK_ID
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from design_c_grid_tasks import REWARD_ARGS, TIMESTEPS, HSM, LAYOUT
from design_c_urgency_ablation_tasks import _drop_flag

A_VALUE = 0.0
SEEDS = list(range(10))
OUTPUT_BASE = "experiments/results/design_c/urgency_only_ablation"

# R7_kitchen_sink with ONLY --seeker-escalating-urgency removed (bare bool
# flag); coverage and everything else stay.
R7_NO_URGENCY_ARGS = _drop_flag(
    list(REWARD_ARGS["R7_kitchen_sink"]),
    "--seeker-escalating-urgency", has_value=False,
)


def build_task_table():
    tasks = []
    for seed in SEEDS:
        a_str = f"A{int(A_VALUE * 100):02d}"
        tasks.append(dict(
            task_name=f"R7_no_urgency/{a_str}/seed_{seed}",
            A=A_VALUE,
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
        "--zoo-interval", "50", "--zoo-max-size", "50",
        "--sampling-strategy", "uniform",
        "--batch-size", "4096", "--num-envs", "64",
        "--output-dir", output_dir,
    ] + R7_NO_URGENCY_ARGS

    print(f"Urgency-only ablation task {task['task_name']}")
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
        print(f"Reward args: {' '.join(R7_NO_URGENCY_ARGS)}")
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
