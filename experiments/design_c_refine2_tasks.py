#!/usr/bin/env python3
"""
Design C REFINE round 2 — R7 only, seeds 5–8.

Per the idea-critic's verdict on the MCMC result:
- R4 σ_seeker is small; the cells are tight; no need for more R4 seeds.
- R7 σ_seeker = 1.41 in log-odds is the dominant uncertainty; need more R7
  seeds to tighten σ AND to enable per-seed-property regressions for the
  exploratory σ-decomposition story.

8 runs: R7 × {A=0, A=0.5} × seeds {5, 6, 7, 8}.

Usage:
  python experiments/design_c_refine2_tasks.py --list
  python experiments/design_c_refine2_tasks.py --task-id $SLURM_ARRAY_TASK_ID
"""
import argparse
import itertools
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from design_c_grid_tasks import REWARD_ARGS, A_VALUES, TIMESTEPS, HSM, LAYOUT

REFINE2_REWARDS = ["R7_kitchen_sink"]
REFINE2_SEEDS = [5, 6, 7, 8]
OUTPUT_BASE = "experiments/results/design_c/grid"


def build_task_table():
    tasks = []
    for reward, A, seed in itertools.product(REFINE2_REWARDS, A_VALUES, REFINE2_SEEDS):
        a_str = f"A{int(A * 100):02d}"
        task_name = f"{reward}/{a_str}/seed_{seed}"
        tasks.append(dict(task_name=task_name, reward=reward, A=A,
                          latest_prob=round(1.0 - A, 2), seed=seed))
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
    ] + REWARD_ARGS[task["reward"]]
    print(f"Refine2 task {task['task_name']}")
    if dry_run:
        print("  [DRY RUN]"); return
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
    if args.count: print(len(tasks)); return
    if args.list:
        for i, t in enumerate(tasks): print(f"  {i}  {t['task_name']}")
        return
    if args.task_id is not None:
        if not 0 <= args.task_id < len(tasks):
            print(f"ERROR: task-id out of range", file=sys.stderr); sys.exit(1)
        run_task(tasks[args.task_id], dry_run=args.dry_run)
        return
    p.print_help()


if __name__ == "__main__":
    main()
