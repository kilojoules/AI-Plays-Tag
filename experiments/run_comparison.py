#!/usr/bin/env python3
"""
Run comparison experiments between vanilla self-play and SCRO.

This script orchestrates multiple training runs with different seeds
and generates comparison plots and statistics.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_vanilla_selfplay(
    timesteps: int,
    seed: int,
    output_dir: str,
) -> Dict[str, Any]:
    """Run vanilla self-play training."""
    cmd = [
        sys.executable,
        "trainer/train_fast.py",
        "--timesteps", str(timesteps),
        "--num-envs", "64",
        "--output-dir", output_dir,
    ]

    # Set seed via environment
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(seed)

    print(f"\n{'='*60}")
    print(f"Running Vanilla Self-Play (seed={seed})")
    print(f"{'='*60}")

    start = time.time()
    result = subprocess.run(cmd, env=env, capture_output=False)
    elapsed = time.time() - start

    # Find the latest run directory
    run_dirs = sorted(Path(output_dir).glob("*"))
    if run_dirs:
        latest = run_dirs[-1]
        metadata_path = latest / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            return {
                "algorithm": "vanilla_selfplay",
                "seed": seed,
                "run_dir": str(latest),
                "elapsed": elapsed,
                "metadata": metadata,
            }

    return {
        "algorithm": "vanilla_selfplay",
        "seed": seed,
        "run_dir": output_dir,
        "elapsed": elapsed,
        "error": "No metadata found",
    }


def run_scro(
    generations: int,
    training_steps: int,
    grid_size: int,
    seed: int,
    output_dir: str,
) -> Dict[str, Any]:
    """Run SCRO training."""
    cmd = [
        sys.executable,
        "experiments/train_scro.py",
        "--generations", str(generations),
        "--grid-size", str(grid_size),
        "--training-steps", str(training_steps),
        "--seed", str(seed),
        "--output-dir", output_dir,
    ]

    print(f"\n{'='*60}")
    print(f"Running SCRO (seed={seed}, grid={grid_size}x{grid_size})")
    print(f"{'='*60}")

    start = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - start

    # Find the latest run directory
    run_dirs = sorted(Path(output_dir).glob("*"))
    if run_dirs:
        latest = run_dirs[-1]
        metadata_path = latest / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            return {
                "algorithm": "scro",
                "seed": seed,
                "grid_size": grid_size,
                "run_dir": str(latest),
                "elapsed": elapsed,
                "metadata": metadata,
            }

    return {
        "algorithm": "scro",
        "seed": seed,
        "run_dir": output_dir,
        "elapsed": elapsed,
        "error": "No metadata found",
    }


def compute_equivalent_timesteps(
    scro_generations: int,
    scro_training_steps: int,
    scro_grid_size: int,
) -> int:
    """
    Compute equivalent timesteps for fair comparison.

    SCRO trains multiple agents per generation, so total compute is:
    generations * training_steps * num_agents * 2 (both layers)
    """
    num_agents = scro_grid_size * scro_grid_size
    return scro_generations * scro_training_steps * num_agents * 2


def main():
    parser = argparse.ArgumentParser(description="Run self-play vs SCRO comparison")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456],
                        help="Random seeds for multiple runs")
    parser.add_argument("--scro-generations", type=int, default=30,
                        help="SCRO generations")
    parser.add_argument("--scro-grid-size", type=int, default=3,
                        help="SCRO grid size")
    parser.add_argument("--scro-training-steps", type=int, default=2048,
                        help="SCRO training steps per agent per generation")
    parser.add_argument("--output-dir", type=str, default="experiments/results",
                        help="Base output directory")
    parser.add_argument("--skip-vanilla", action="store_true",
                        help="Skip vanilla self-play runs")
    parser.add_argument("--skip-scro", action="store_true",
                        help="Skip SCRO runs")

    args = parser.parse_args()

    # Compute equivalent timesteps for fair comparison
    vanilla_timesteps = compute_equivalent_timesteps(
        args.scro_generations,
        args.scro_training_steps,
        args.scro_grid_size,
    )

    print(f"Comparison Experiment Setup")
    print(f"="*60)
    print(f"Seeds: {args.seeds}")
    print(f"SCRO: {args.scro_generations} generations, {args.scro_grid_size}x{args.scro_grid_size} grid")
    print(f"SCRO training steps/agent/gen: {args.scro_training_steps}")
    print(f"Equivalent vanilla timesteps: {vanilla_timesteps:,}")
    print(f"Output: {args.output_dir}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(args.output_dir, f"comparison_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)

    results = []

    # Run vanilla self-play for each seed
    if not args.skip_vanilla:
        for seed in args.seeds:
            vanilla_dir = os.path.join(experiment_dir, "vanilla_selfplay", f"seed_{seed}")
            os.makedirs(vanilla_dir, exist_ok=True)

            result = run_vanilla_selfplay(
                timesteps=vanilla_timesteps,
                seed=seed,
                output_dir=vanilla_dir,
            )
            results.append(result)

    # Run SCRO for each seed
    if not args.skip_scro:
        for seed in args.seeds:
            scro_dir = os.path.join(experiment_dir, "scro", f"seed_{seed}")
            os.makedirs(scro_dir, exist_ok=True)

            result = run_scro(
                generations=args.scro_generations,
                training_steps=args.scro_training_steps,
                grid_size=args.scro_grid_size,
                seed=seed,
                output_dir=scro_dir,
            )
            results.append(result)

    # Save experiment summary
    summary = {
        "timestamp": timestamp,
        "seeds": args.seeds,
        "vanilla_timesteps": vanilla_timesteps,
        "scro_config": {
            "generations": args.scro_generations,
            "grid_size": args.scro_grid_size,
            "training_steps": args.scro_training_steps,
        },
        "results": results,
    }

    summary_path = os.path.join(experiment_dir, "experiment_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("Experiment Complete!")
    print(f"{'='*60}")
    print(f"Results saved to: {experiment_dir}")
    print(f"Summary: {summary_path}")

    # Print quick comparison
    print(f"\nQuick Results Summary:")
    for r in results:
        if "error" not in r:
            algo = r["algorithm"]
            seed = r["seed"]
            meta = r.get("metadata", {})
            win_rate = meta.get("seeker_win_rate", meta.get("protagonist_win_rate", "N/A"))
            if isinstance(win_rate, float):
                win_rate = f"{win_rate:.1%}"
            print(f"  {algo} (seed={seed}): win_rate={win_rate}, time={r['elapsed']:.1f}s")


if __name__ == "__main__":
    main()
