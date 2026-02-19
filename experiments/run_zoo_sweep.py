#!/usr/bin/env python3
"""
Run zoo training sweep experiments.

Sweeps:
- A (latest_prob): 0.0, 0.05, 0.1, 0.2
- Zoo mode: hider-only, both

Total: 8 experiments
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
    layout: str = "four_corners"
    enable_sprint: bool = False
    hider_speed_mult: float = 1.0
    sprint_speed_mult: float = 1.5
    algorithm: str = "ppo"  # "ppo" or "sac"


def get_experiments(algorithm: str = "ppo") -> List[ExperimentConfig]:
    """Define all experiment configurations."""
    experiments = []

    for A in [0.0, 0.05, 0.1, 0.2]:
        for use_seeker_zoo in [False, True]:
            zoo_mode = "both" if use_seeker_zoo else "hider_only"
            name = f"A{int(A*100):02d}_{zoo_mode}"
            experiments.append(ExperimentConfig(
                name=name,
                latest_prob=A,
                use_seeker_zoo=use_seeker_zoo,
                algorithm=algorithm,
            ))

    return experiments


def find_resume_dir(output_dir: str) -> Optional[str]:
    """Find the latest timestamped run directory with checkpoints to resume from."""
    output_path = Path(output_dir)
    if not output_path.exists():
        return None

    # Find timestamped subdirectories (format: YYYYMMDD_HHMMSS)
    candidates = sorted(output_path.glob("2*"), reverse=True)
    for d in candidates:
        ckpt_dir = d / "checkpoints"
        if ckpt_dir.exists() and any(ckpt_dir.glob("*.pt")):
            return str(d)
    return None


def run_experiment(exp: ExperimentConfig, output_base: str, resume: bool = False) -> subprocess.Popen:
    """Launch an experiment as a subprocess."""
    output_dir = f"{output_base}/{exp.name}"

    # Select training script based on algorithm
    if exp.algorithm == "sac":
        train_script = "trainer/train_zoo_sac.py"
    else:
        train_script = "trainer/train_zoo.py"

    cmd = [
        sys.executable,
        train_script,
        "--timesteps", str(exp.timesteps),
        "--latest-prob", str(exp.latest_prob),
        "--output-dir", output_dir,
        "--layout", exp.layout,
        "--hider-speed-mult", str(exp.hider_speed_mult),
        "--sprint-speed-mult", str(exp.sprint_speed_mult),
    ]

    if exp.use_seeker_zoo:
        cmd.append("--use-seeker-zoo")

    if exp.enable_sprint:
        cmd.append("--enable-sprint")

    # Check for resumable run
    resume_dir = find_resume_dir(output_dir) if resume else None
    if resume_dir:
        cmd.extend(["--resume", resume_dir])

    mode = "RESUMING" if resume_dir else "Starting fresh"
    print(f"\nLaunching: {exp.name} ({mode})")
    print(f"  A = {exp.latest_prob} ({exp.latest_prob*100:.0f}% latest, {(1-exp.latest_prob)*100:.0f}% zoo)")
    print(f"  Seeker zoo: {exp.use_seeker_zoo}")
    print(f"  Output: {output_dir}")
    if resume_dir:
        print(f"  Resume from: {resume_dir}")
    print(f"  Command: {' '.join(cmd)}")

    # Create log file
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(output_dir) / "train.log"

    with open(log_path, "a" if resume_dir else "w") as log_file:
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
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoints in existing run dirs")
    parser.add_argument("--algorithm", type=str, default="ppo",
                        choices=["ppo", "sac"],
                        help="Training algorithm (default: ppo)")

    args = parser.parse_args()

    experiments = get_experiments(algorithm=args.algorithm)

    print("="*60)
    print("ZOO TRAINING SWEEP")
    print("="*60)
    print(f"\nAlgorithm: {args.algorithm.upper()}")
    print(f"Experiments to run: {len(experiments)}")
    print(f"Max parallel: {args.max_parallel}")
    print(f"Output: {args.output_dir}")
    print()

    for i, exp in enumerate(experiments):
        marker = " <-- START" if i == args.start_index else ""
        print(f"  [{i}] {exp.name}: A={exp.latest_prob}, seeker_zoo={exp.use_seeker_zoo}, algo={exp.algorithm}{marker}")

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
            proc = run_experiment(exp, args.output_dir, resume=args.resume)
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
