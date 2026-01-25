#!/usr/bin/env python3
"""
Visualize comparison results between vanilla self-play and SCRO.

Generates:
1. Learning curves comparison
2. Win rate progression
3. Final performance box plots
4. Statistical comparison
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib required. Install with: pip install matplotlib", file=sys.stderr)
    sys.exit(1)


def load_vanilla_metrics(run_dir: str) -> Dict[str, List[float]]:
    """Load metrics from vanilla self-play run."""
    run_path = Path(run_dir)
    metrics_path = run_path / "metrics.csv"
    metadata_path = run_path / "metadata.json"

    if not metrics_path.exists():
        # Check subdirectories
        subdirs = list(run_path.glob("*/metrics.csv"))
        if subdirs:
            metrics_path = subdirs[0]
            metadata_path = metrics_path.parent / "metadata.json"
        else:
            return {}

    data = {
        "timesteps": [],
        "episodes": [],
        "seeker_win_rate": [],
        "seeker_reward_mean": [],
    }

    with open(metrics_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["timesteps"].append(int(row.get("timesteps", 0)))
            data["episodes"].append(int(row.get("episodes", 0)))
            data["seeker_win_rate"].append(float(row.get("seeker_win_rate", 0)))
            data["seeker_reward_mean"].append(float(row.get("seeker_reward_mean", 0)))

    # If no data rows but metadata exists, use metadata for final stats
    if not data["timesteps"] and metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        total_timesteps = metadata.get("config", {}).get("total_timesteps", 0)
        win_rate = metadata.get("seeker_win_rate", 0)
        data["timesteps"] = [total_timesteps]
        data["seeker_win_rate"] = [win_rate]
        data["episodes"] = [metadata.get("total_episodes", 0)]
        data["seeker_reward_mean"] = [0]  # Not available from metadata

    return data


def load_scro_metrics(run_dir: str) -> Dict[str, List[float]]:
    """Load metrics from SCRO run."""
    metrics_path = Path(run_dir) / "metrics.csv"
    if not metrics_path.exists():
        subdirs = list(Path(run_dir).glob("*/metrics.csv"))
        if subdirs:
            metrics_path = subdirs[0]
        else:
            return {}

    data = {
        "generation": [],
        "timesteps": [],
        "protagonist_win_rate": [],
        "mean_prot_fitness": [],
        "protagonist_pop": [],
    }

    with open(metrics_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["generation"].append(int(row.get("generation", 0)))
            data["timesteps"].append(int(row.get("timesteps", 0)))
            data["protagonist_win_rate"].append(float(row.get("protagonist_win_rate", 0)))
            data["mean_prot_fitness"].append(float(row.get("mean_prot_fitness", 0)))
            data["protagonist_pop"].append(int(row.get("protagonist_pop", 0)))

    return data


def plot_learning_curves(
    vanilla_runs: List[Dict[str, List[float]]],
    scro_runs: List[Dict[str, List[float]]],
    output_path: str,
):
    """Plot learning curves for both algorithms."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Win rate comparison
    ax = axes[0]

    # Plot vanilla runs
    for i, data in enumerate(vanilla_runs):
        if data.get("timesteps") and data.get("seeker_win_rate"):
            alpha = 0.3 if len(vanilla_runs) > 1 else 1.0
            ax.plot(data["timesteps"], data["seeker_win_rate"],
                   'b-', alpha=alpha, label='Vanilla' if i == 0 else None)

    # Plot SCRO runs
    for i, data in enumerate(scro_runs):
        if data.get("timesteps") and data.get("protagonist_win_rate"):
            alpha = 0.3 if len(scro_runs) > 1 else 1.0
            ax.plot(data["timesteps"], data["protagonist_win_rate"],
                   'r-', alpha=alpha, label='SCRO' if i == 0 else None)

    # Plot means if multiple runs
    if len(vanilla_runs) > 1:
        min_len = min(len(d.get("timesteps", [])) for d in vanilla_runs if d.get("timesteps"))
        if min_len > 0:
            mean_wr = np.zeros(min_len)
            for data in vanilla_runs:
                wr = data.get("seeker_win_rate", [])[:min_len]
                mean_wr[:len(wr)] += wr
            mean_wr = mean_wr / len(vanilla_runs)
            timesteps = vanilla_runs[0].get("timesteps", list(range(min_len)))[:min_len]
            ax.plot(timesteps, mean_wr, 'b-', linewidth=2, label='Vanilla (mean)')

    if len(scro_runs) > 1:
        min_len = min(len(d.get("timesteps", [])) for d in scro_runs if d.get("timesteps"))
        if min_len > 0:
            mean_wr = np.zeros(min_len)
            for data in scro_runs:
                wr = data.get("protagonist_win_rate", [])[:min_len]
                mean_wr[:len(wr)] += wr
            mean_wr = mean_wr / len(scro_runs)
            timesteps = scro_runs[0].get("timesteps", list(range(min_len)))[:min_len]
            ax.plot(timesteps, mean_wr, 'r-', linewidth=2, label='SCRO (mean)')

    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Win Rate (Seeker/Protagonist)')
    ax.set_title('Win Rate Learning Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # Fitness/Reward comparison
    ax = axes[1]

    for i, data in enumerate(vanilla_runs):
        if data.get("timesteps") and data.get("seeker_reward_mean"):
            alpha = 0.3 if len(vanilla_runs) > 1 else 1.0
            ax.plot(data["timesteps"], data["seeker_reward_mean"],
                   'b-', alpha=alpha, label='Vanilla' if i == 0 else None)

    for i, data in enumerate(scro_runs):
        if data.get("timesteps") and data.get("mean_prot_fitness"):
            alpha = 0.3 if len(scro_runs) > 1 else 1.0
            ax.plot(data["timesteps"], data["mean_prot_fitness"],
                   'r-', alpha=alpha, label='SCRO' if i == 0 else None)

    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Mean Reward/Fitness')
    ax.set_title('Reward/Fitness Learning Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved learning curves: {output_path}")


def plot_final_comparison(
    vanilla_runs: List[Dict[str, List[float]]],
    scro_runs: List[Dict[str, List[float]]],
    output_path: str,
):
    """Plot box plot comparison of final performance."""
    fig, ax = plt.subplots(figsize=(8, 6))

    vanilla_final = []
    scro_final = []

    for data in vanilla_runs:
        if data.get("seeker_win_rate"):
            vanilla_final.append(data["seeker_win_rate"][-1])

    for data in scro_runs:
        if data.get("protagonist_win_rate"):
            scro_final.append(data["protagonist_win_rate"][-1])

    if vanilla_final or scro_final:
        data_to_plot = []
        labels = []

        if vanilla_final:
            data_to_plot.append(vanilla_final)
            labels.append(f'Vanilla\n(n={len(vanilla_final)})')

        if scro_final:
            data_to_plot.append(scro_final)
            labels.append(f'SCRO\n(n={len(scro_final)})')

        bp = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True)

        colors = ['#3498db', '#e74c3c']
        for patch, color in zip(bp['boxes'], colors[:len(data_to_plot)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Add individual points
        for i, data in enumerate(data_to_plot):
            x = np.random.normal(i + 1, 0.04, size=len(data))
            ax.scatter(x, data, alpha=0.6, s=50, c=colors[i], edgecolors='black')

    ax.set_ylabel('Final Win Rate')
    ax.set_title('Final Performance Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved final comparison: {output_path}")


def generate_statistics(
    vanilla_runs: List[Dict[str, List[float]]],
    scro_runs: List[Dict[str, List[float]]],
    output_path: str,
):
    """Generate statistical comparison report."""
    vanilla_final = [d["seeker_win_rate"][-1] for d in vanilla_runs if d.get("seeker_win_rate")]
    scro_final = [d["protagonist_win_rate"][-1] for d in scro_runs if d.get("protagonist_win_rate")]

    report = []
    report.append("=" * 60)
    report.append("Statistical Comparison Report")
    report.append("=" * 60)
    report.append("")

    report.append("Vanilla Self-Play:")
    if vanilla_final:
        report.append(f"  N runs: {len(vanilla_final)}")
        report.append(f"  Mean final win rate: {np.mean(vanilla_final):.3f}")
        report.append(f"  Std: {np.std(vanilla_final):.3f}")
        report.append(f"  Min: {np.min(vanilla_final):.3f}")
        report.append(f"  Max: {np.max(vanilla_final):.3f}")
    else:
        report.append("  No data available")

    report.append("")
    report.append("SCRO:")
    if scro_final:
        report.append(f"  N runs: {len(scro_final)}")
        report.append(f"  Mean final win rate: {np.mean(scro_final):.3f}")
        report.append(f"  Std: {np.std(scro_final):.3f}")
        report.append(f"  Min: {np.min(scro_final):.3f}")
        report.append(f"  Max: {np.max(scro_final):.3f}")
    else:
        report.append("  No data available")

    report.append("")
    report.append("-" * 60)

    if vanilla_final and scro_final:
        diff = np.mean(scro_final) - np.mean(vanilla_final)
        report.append(f"Difference (SCRO - Vanilla): {diff:+.3f}")

        # Simple t-test if scipy available
        try:
            from scipy import stats
            t_stat, p_value = stats.ttest_ind(vanilla_final, scro_final)
            report.append(f"T-test: t={t_stat:.3f}, p={p_value:.4f}")
            if p_value < 0.05:
                winner = "SCRO" if diff > 0 else "Vanilla"
                report.append(f"Result: {winner} significantly better (p < 0.05)")
            else:
                report.append("Result: No significant difference (p >= 0.05)")
        except ImportError:
            report.append("(scipy not available for statistical test)")

    report.append("")
    report_text = "\n".join(report)

    with open(output_path, "w") as f:
        f.write(report_text)

    print(report_text)
    print(f"\nReport saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize comparison results")
    parser.add_argument("experiment_dir", type=str,
                        help="Path to experiment directory (containing vanilla_selfplay and scro subdirs)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for plots (default: experiment_dir)")

    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    output_dir = Path(args.output_dir) if args.output_dir else experiment_dir

    # Load vanilla runs
    vanilla_runs = []
    vanilla_dir = experiment_dir / "vanilla_selfplay"
    if vanilla_dir.exists():
        for seed_dir in vanilla_dir.iterdir():
            if seed_dir.is_dir():
                data = load_vanilla_metrics(str(seed_dir))
                if data:
                    vanilla_runs.append(data)
    print(f"Loaded {len(vanilla_runs)} vanilla self-play runs")

    # Load SCRO runs
    scro_runs = []
    scro_dir = experiment_dir / "scro"
    if scro_dir.exists():
        for seed_dir in scro_dir.iterdir():
            if seed_dir.is_dir():
                data = load_scro_metrics(str(seed_dir))
                if data:
                    scro_runs.append(data)
    print(f"Loaded {len(scro_runs)} SCRO runs")

    if not vanilla_runs and not scro_runs:
        print("No data found!")
        return

    # Generate visualizations
    os.makedirs(output_dir, exist_ok=True)

    plot_learning_curves(
        vanilla_runs, scro_runs,
        str(output_dir / "learning_curves.png")
    )

    plot_final_comparison(
        vanilla_runs, scro_runs,
        str(output_dir / "final_comparison.png")
    )

    generate_statistics(
        vanilla_runs, scro_runs,
        str(output_dir / "statistics.txt")
    )


if __name__ == "__main__":
    main()
