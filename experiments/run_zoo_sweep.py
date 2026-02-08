#!/usr/bin/env python3
"""
Run zoo training sweep experiments.

Sweeps:
- A (latest_prob): 0.05, 0.1, 0.2
- Zoo mode: hider-only, both

Total: 6 experiments
Runs 2 in parallel (1 slot for existing training + 1 new)
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
import json


@dataclass
class ExperimentConfig:
    name: str
    latest_prob: float
    use_seeker_zoo: bool
    timesteps: int = 10_000_000  # Match stable training


def get_experiments() -> List[ExperimentConfig]:
    """Define all experiment configurations."""
    experiments = []

    for A in [0.05, 0.1, 0.2]:
        for use_seeker_zoo in [False, True]:
            zoo_mode = "both" if use_seeker_zoo else "hider_only"
            name = f"A{int(A*100):02d}_{zoo_mode}"
            experiments.append(ExperimentConfig(
                name=name,
                latest_prob=A,
                use_seeker_zoo=use_seeker_zoo,
            ))

    return experiments


def run_experiment(exp: ExperimentConfig, output_base: str) -> subprocess.Popen:
    """Launch an experiment as a subprocess."""
    output_dir = f"{output_base}/{exp.name}"

    cmd = [
        sys.executable,
        "trainer/train_zoo.py",
        "--timesteps", str(exp.timesteps),
        "--latest-prob", str(exp.latest_prob),
        "--output-dir", output_dir,
        "--layout", "four_corners",
    ]

    if exp.use_seeker_zoo:
        cmd.append("--use-seeker-zoo")

    print(f"\nLaunching: {exp.name}")
    print(f"  A = {exp.latest_prob} ({exp.latest_prob*100:.0f}% latest, {(1-exp.latest_prob)*100:.0f}% zoo)")
    print(f"  Seeker zoo: {exp.use_seeker_zoo}")
    print(f"  Output: {output_dir}")
    print(f"  Command: {' '.join(cmd)}")

    # Create log file
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(output_dir) / "train.log"

    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=str(Path(__file__).parent.parent),
        )

    return proc


def main():
    parser = argparse.ArgumentParser(description="Run zoo training sweep")
    parser.add_argument("--max-parallel", type=int, default=2,
                        help="Maximum parallel experiments (default: 2)")
    parser.add_argument("--output-dir", type=str,
                        default="experiments/results/zoo_sweep",
                        help="Base output directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would run without executing")
    parser.add_argument("--start-index", type=int, default=0,
                        help="Start from experiment index (for resuming)")

    args = parser.parse_args()

    experiments = get_experiments()

    print("="*60)
    print("ZOO TRAINING SWEEP")
    print("="*60)
    print(f"\nExperiments to run: {len(experiments)}")
    print(f"Max parallel: {args.max_parallel}")
    print(f"Output: {args.output_dir}")
    print()

    for i, exp in enumerate(experiments):
        marker = " <-- START" if i == args.start_index else ""
        print(f"  [{i}] {exp.name}: A={exp.latest_prob}, seeker_zoo={exp.use_seeker_zoo}{marker}")

    if args.dry_run:
        print("\n[DRY RUN] Would execute above experiments")
        return

    print("\n" + "-"*60)
    print("Starting experiments...")
    print("-"*60)

    running: List[tuple] = []  # (exp, proc)
    completed = []
    queue = list(enumerate(experiments))[args.start_index:]

    while queue or running:
        # Launch new experiments if slots available
        while len(running) < args.max_parallel and queue:
            idx, exp = queue.pop(0)
            proc = run_experiment(exp, args.output_dir)
            running.append((exp, proc))
            time.sleep(2)  # Brief delay between launches

        # Check for completed
        still_running = []
        for exp, proc in running:
            ret = proc.poll()
            if ret is not None:
                status = "SUCCESS" if ret == 0 else f"FAILED (code {ret})"
                print(f"\n[DONE] {exp.name}: {status}")
                completed.append((exp, ret))
            else:
                still_running.append((exp, proc))
        running = still_running

        if running:
            # Print status
            running_names = [exp.name for exp, _ in running]
            print(f"\r  Running: {running_names}, Queue: {len(queue)} remaining", end="", flush=True)
            time.sleep(30)  # Check every 30 seconds

    print("\n\n" + "="*60)
    print("SWEEP COMPLETE")
    print("="*60)

    successes = sum(1 for _, ret in completed if ret == 0)
    failures = len(completed) - successes

    print(f"\nResults: {successes} succeeded, {failures} failed")
    for exp, ret in completed:
        status = "OK" if ret == 0 else f"FAIL({ret})"
        print(f"  {exp.name}: {status}")

    # Save summary
    summary = {
        "experiments": [
            {
                "name": exp.name,
                "latest_prob": exp.latest_prob,
                "use_seeker_zoo": exp.use_seeker_zoo,
                "return_code": ret,
            }
            for exp, ret in completed
        ],
        "successes": successes,
        "failures": failures,
    }

    summary_path = Path(args.output_dir) / "sweep_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
