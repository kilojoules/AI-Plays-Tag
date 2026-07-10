#!/usr/bin/env python3
"""
Coverage-bonus ablation — diagnostic, NOT part of pre-registered Design C.

Motivation
----------
The trajectory/behavior analysis (job 18636973) found R7_kitchen_sink seekers
are bimodal *across seeds*: ~44% of A=0.0 seeds (4/9) collapse into a "roamer"
local optimum — long meandering paths, large mean distance to the hider held
all episode — instead of learning direct pursuit. All 9 R7/A=0.5 seeds learned
pursuit, so the bad basin lives entirely at A=0.0.

R7_kitchen_sink has exactly one seeker-side dense term that can be maximised
WITHOUT closing distance: `area_coverage_bonus` (0.05). It is dense and
available from step 0 (any moving policy covers area), so it can bootstrap a
roaming policy before the pursuit signal (`distance_reward_scale`) dominates.

Hypothesis: removing `area_coverage_bonus` eliminates the A=0.0 bad basin.

Design
------
10 runs: reward = R7_no_coverage, A = 0.0, seeds 0–9.
- A fixed at 0.0 — the only condition where the bad basin appears.
- Seeds 0–8 are paired seed-for-seed with the existing
  experiments/results/design_c/grid/R7_kitchen_sink/A00/seed_* runs (same seed
  => same network init), so the comparison isolates the coverage term. Seed 9
  is an extra run for power.

Results go to experiments/results/design_c/coverage_ablation/ — a separate
tree from the pre-registered grid, so this diagnostic cannot contaminate it.

Usage:
  python experiments/design_c_coverage_ablation_tasks.py --list
  python experiments/design_c_coverage_ablation_tasks.py --task-id $SLURM_ARRAY_TASK_ID
  python experiments/design_c_coverage_ablation_tasks.py --count
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from design_c_grid_tasks import REWARD_ARGS, TIMESTEPS, HSM, LAYOUT

A_VALUE = 0.0
SEEDS = list(range(10))
OUTPUT_BASE = "experiments/results/design_c/coverage_ablation"


def _drop_flag(args, flag):
    """Return `args` with `flag` and its single value removed.

    Asserts the flag was present so a rename/removal in design_c_grid_tasks.py
    fails loudly here instead of silently producing an unchanged reward.
    """
    assert flag in args, f"{flag} missing from base R7 args — reward args drifted"
    i = args.index(flag)
    return args[:i] + args[i + 2:]


# R7_kitchen_sink with --area-coverage-bonus removed; everything else identical.
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
    ] + R7_NO_COVERAGE_ARGS

    print(f"Coverage-ablation task {task['task_name']}")
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
    p = argparse.ArgumentParser(description="Coverage-bonus ablation task launcher")
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
