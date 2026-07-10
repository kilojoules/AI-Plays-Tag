#!/usr/bin/env python3
"""
Urgency ablation — diagnostic follow-up to the coverage ablation. NOT
part of pre-registered Design C.

Motivation
----------
The coverage ablation (R7_no_coverage, job 18766646) cut the R7/A=0.0 bad
basin from 4/9 collapsed seeds to 1/10 — `area_coverage_bonus` is the
dominant route into the "roamer" local optimum, but seed 5 still collapsed
(wr 0.20 vs the fixed reference hider) with coverage already removed.

To test whether `seeker_escalating_urgency` is the residual cause, urgency
must be removed *from the no-coverage baseline* — removing it from full
R7_kitchen_sink would leave coverage's own collapses in and confound the
result. So this arm drops BOTH terms.

R7_no_cov_urg = R7_kitchen_sink without --area-coverage-bonus and without
--seeker-escalating-urgency. Remaining seeker-side dense shaping: only
--seeker-time-penalty and --distance-reward-scale (pure pursuit signal).

Read-out: if R7_no_cov_urg has 0 collapsed seeds, escalating urgency is the
residual cause. If seed-5-type collapse persists, the residual is intrinsic
exploration variance, not a reward term.

Design
------
10 runs: reward = R7_no_cov_urg, A = 0.0, seeds 0-9 — same design as the
coverage ablation, so the three groups (R7_kitchen_sink, R7_no_coverage,
R7_no_cov_urg) are paired seed-for-seed at A=0.0.

Results: experiments/results/design_c/urgency_ablation/ — separate tree
from the pre-registered grid.

Usage:
  python experiments/design_c_urgency_ablation_tasks.py --list
  python experiments/design_c_urgency_ablation_tasks.py --task-id $SLURM_ARRAY_TASK_ID
  python experiments/design_c_urgency_ablation_tasks.py --count
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from design_c_grid_tasks import REWARD_ARGS, TIMESTEPS, HSM, LAYOUT

A_VALUE = 0.0
SEEDS = list(range(10))
OUTPUT_BASE = "experiments/results/design_c/urgency_ablation"


def _drop_flag(args, flag, has_value=True):
    """Return `args` with `flag` removed (plus its value if has_value).

    Asserts the flag was present so a rename/removal in design_c_grid_tasks.py
    fails loudly here instead of silently producing an unchanged reward.
    """
    assert flag in args, f"{flag} missing from base R7 args — reward args drifted"
    i = args.index(flag)
    return args[:i] + args[i + (2 if has_value else 1):]


# R7_kitchen_sink without --area-coverage-bonus (flag+value) and without
# --seeker-escalating-urgency (bare store_true flag); everything else identical.
_base = list(REWARD_ARGS["R7_kitchen_sink"])
R7_NO_COV_URG_ARGS = _drop_flag(
    _drop_flag(_base, "--area-coverage-bonus", has_value=True),
    "--seeker-escalating-urgency", has_value=False,
)


def build_task_table():
    tasks = []
    for seed in SEEDS:
        a_str = f"A{int(A_VALUE * 100):02d}"
        tasks.append(dict(
            task_name=f"R7_no_cov_urg/{a_str}/seed_{seed}",
            A=A_VALUE,
            latest_prob=round(1.0 - A_VALUE, 2),  # 1.0 at A=0
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
    ] + R7_NO_COV_URG_ARGS

    print(f"Urgency-ablation task {task['task_name']}")
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
    p = argparse.ArgumentParser(description="Urgency-ablation task launcher")
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
        print(f"Reward args: {' '.join(R7_NO_COV_URG_ARGS)}")
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
