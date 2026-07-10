#!/usr/bin/env python3
"""
Design C power extension — new seeds for the four prereg cells.

Per prereg v3 (design_c_prereg_v3_power_extension.md): brings every
prereg cell to n=20 runs so the twice-REFINE'd beta_RA gets a decisive
test. 52 tasks:
  R4_sparse       x A in {0.0, 0.5} x seeds 5-19   (2 x 15)
  R7_kitchen_sink x A in {0.0, 0.5} x seeds 9-19   (2 x 11)

Outputs merge into the existing grid tree
(experiments/results/design_c/grid/<reward>/<Axx>/seed_<s>), so
discover_run() and the anchor panel pick the new runs up unchanged.

Usage:
  python experiments/design_c_power_extension_tasks.py --list
  python experiments/design_c_power_extension_tasks.py --task-id $((LSB_JOBINDEX - 1))
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from design_c_grid_tasks import (
    REWARD_ARGS, TIMESTEPS, HSM, LAYOUT, OUTPUT_BASE, run_task,
)

NEW_SEEDS = {
    "R4_sparse": list(range(5, 20)),
    "R7_kitchen_sink": list(range(9, 20)),
}
A_VALUES = [0.0, 0.5]


def build_task_table():
    tasks = []
    for reward, seeds in NEW_SEEDS.items():
        for A in A_VALUES:
            for seed in seeds:
                a_str = f"A{int(A * 100):02d}"
                tasks.append(dict(
                    task_name=f"{reward}/{a_str}/seed_{seed}",
                    reward=reward,
                    A=A,
                    latest_prob=round(1.0 - A, 2),
                    seed=seed,
                ))
    return tasks


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
