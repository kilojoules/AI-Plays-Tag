#!/usr/bin/env python3
"""
Run self-play sweep: seeker_time_penalty × hider_speed_mult.

Sweep grid (4×5 = 20 experiments):
- seeker_time_penalty: [-0.005, -0.01, -0.02, -0.05]
- hider_speed_mult:    [1.0, 1.05, 1.1, 1.15, 1.2]
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
    seeker_time_penalty: float
    hider_speed_mult: float
    timesteps: int = 10_000_000
    num_envs: int = 64
    batch_size: int = 2048
    lr: float = 3e-4
    layout: str = "four_corners"


def get_experiments() -> List[ExperimentConfig]:
    """Define all experiment configurations."""
    experiments = []

    for stp in [0.005, 0.01, 0.02, 0.05]:
        for hsm in [1.0, 1.05, 1.1, 1.15, 1.2]:
            # STP0001_HSM100, STP001_HSM110, etc.
            stp_str = f"{stp:.4f}".replace("0.", "").rstrip("0") or "0"
            hsm_str = f"{round(hsm * 100)}"
            name = f"STP{stp_str}_HSM{hsm_str}"
            experiments.append(ExperimentConfig(
                name=name,
                seeker_time_penalty=-stp,
                hider_speed_mult=hsm,
            ))

    return experiments


def find_resume_dir(output_dir: str) -> Optional[str]:
    """Find the latest timestamped run directory with checkpoints to resume from."""
    output_path = Path(output_dir)
    if not output_path.exists():
        return None

    candidates = sorted(output_path.glob("2*"), reverse=True)
    for d in candidates:
        ckpt_dir = d / "checkpoints"
        if ckpt_dir.exists() and any(ckpt_dir.glob("*.pt")):
            return str(d)
    return None


def run_experiment(exp: ExperimentConfig, output_base: str,
                   resume: bool = False) -> subprocess.Popen:
    """Launch an experiment as a subprocess."""
    output_dir = f"{output_base}/{exp.name}"

    cmd = [
        sys.executable,
        "trainer/train_selfplay.py",
        "--timesteps", str(exp.timesteps),
        "--num-envs", str(exp.num_envs),
        "--batch-size", str(exp.batch_size),
        "--lr", str(exp.lr),
        "--layout", exp.layout,
        "--seeker-time-penalty", str(exp.seeker_time_penalty),
        "--hider-speed-mult", str(exp.hider_speed_mult),
        "--output-dir", output_dir,
    ]

    resume_dir = find_resume_dir(output_dir) if resume else None
    if resume_dir:
        cmd.extend(["--resume", resume_dir])

    mode = "RESUMING" if resume_dir else "Starting fresh"
    print(f"\nLaunching: {exp.name} ({mode})")
    print(f"  seeker_time_penalty = {exp.seeker_time_penalty}")
    print(f"  hider_speed_mult    = {exp.hider_speed_mult}")
    print(f"  Output: {output_dir}")
    if resume_dir:
        print(f"  Resume from: {resume_dir}")
    print(f"  Command: {' '.join(cmd)}")

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
    parser = argparse.ArgumentParser(
        description="Run self-play sweep: seeker_time_penalty × hider_speed_mult")
    parser.add_argument("--max-parallel", type=int, default=16,
                        help="Maximum parallel experiments (default: 16)")
    parser.add_argument("--output-dir", type=str,
                        default="experiments/results/selfplay_sweep",
                        help="Base output directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would run without executing")
    parser.add_argument("--start-index", type=int, default=0,
                        help="Start from experiment index (for resuming)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoints in existing run dirs")

    args = parser.parse_args()

    experiments = get_experiments()

    print("=" * 60)
    print("SELF-PLAY SWEEP: seeker_time_penalty × hider_speed_mult")
    print("=" * 60)
    print(f"\nExperiments to run: {len(experiments)}")
    print(f"Max parallel: {args.max_parallel}")
    print(f"Output: {args.output_dir}")
    print()

    for i, exp in enumerate(experiments):
        marker = " <-- START" if i == args.start_index else ""
        print(f"  [{i:2d}] {exp.name}: "
              f"stp={exp.seeker_time_penalty}, hsm={exp.hider_speed_mult}{marker}")

    if args.dry_run:
        print("\n[DRY RUN] Would execute above experiments")
        return

    print("\n" + "-" * 60)
    print("Starting experiments...")
    print("-" * 60)

    running: List[tuple] = []  # (exp, proc)
    completed = []
    queue = list(enumerate(experiments))[args.start_index:]

    while queue or running:
        # Launch new experiments if slots available
        while len(running) < args.max_parallel and queue:
            idx, exp = queue.pop(0)
            proc = run_experiment(exp, args.output_dir, resume=args.resume)
            running.append((exp, proc))
            time.sleep(1)

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
            running_names = [exp.name for exp, _ in running]
            print(f"\r  Running: {len(running_names)} experiments, "
                  f"Queue: {len(queue)} remaining", end="", flush=True)
            time.sleep(30)

    print("\n\n" + "=" * 60)
    print("SWEEP COMPLETE")
    print("=" * 60)

    successes = sum(1 for _, ret in completed if ret == 0)
    failures = len(completed) - successes

    print(f"\nResults: {successes} succeeded, {failures} failed")
    for exp, ret in completed:
        status = "OK" if ret == 0 else f"FAIL({ret})"
        print(f"  {exp.name}: {status}")

    summary = {
        "experiments": [
            {
                "name": exp.name,
                "seeker_time_penalty": exp.seeker_time_penalty,
                "hider_speed_mult": exp.hider_speed_mult,
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
