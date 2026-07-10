#!/usr/bin/env python3
"""
Analyze A-sweep results and produce comparison plots.

Reads metrics.csv from all sweep experiments and produces:
1. A-curve: seeker win rate vs A (main result), with PPO vs SAC if both present
2. Training dynamics: learning curves overlaid by A value
3. Zoo mode comparison: hider_only vs both
4. Robustness from cross-evaluation (if available)

Usage:
    python experiments/analyze_a_sweep.py experiments/results/sweep/
    python experiments/analyze_a_sweep.py experiments/results/sweep/ppo  # PPO only
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
except ImportError:
    print("matplotlib required. Install with: pip install matplotlib", file=sys.stderr)
    sys.exit(1)


def discover_experiments(sweep_dir: str) -> List[Dict]:
    """Find all experiments and their metrics.

    Returns list of dicts with keys:
        algorithm, A, zoo_mode, seed, metrics_path, metadata_path
    """
    sweep_path = Path(sweep_dir)
    experiments = []

    # Check if sweep_dir contains algo subdirs (ppo/, sac/) or direct experiments
    algo_dirs = []
    for d in sorted(sweep_path.iterdir()):
        if d.is_dir() and d.name in ('ppo', 'sac'):
            algo_dirs.append(d)

    if not algo_dirs:
        # No algo subdirs — treat sweep_dir as a single algorithm
        algo_dirs = [sweep_path]

    for algo_dir in algo_dirs:
        algorithm = algo_dir.name if algo_dir.name in ('ppo', 'sac') else 'ppo'

        for exp_dir in sorted(algo_dir.iterdir()):
            if not exp_dir.is_dir():
                continue

            name = exp_dir.name
            A, zoo_mode = _parse_name(name)

            # Check for seed subdirs
            seed_dirs = sorted(exp_dir.glob("seed_*"))
            if seed_dirs:
                for sd in seed_dirs:
                    seed = int(sd.name.split('_')[1])
                    _add_experiment(sd, algorithm, A, zoo_mode, seed, experiments)
            else:
                _add_experiment(exp_dir, algorithm, A, zoo_mode, None, experiments)

    return experiments


def _parse_name(name: str) -> Tuple[Optional[float], Optional[str]]:
    """Parse experiment directory name to A value and zoo mode."""
    if name.startswith('selfplay'):
        return 0.0, 'selfplay'

    parts = name.replace('__', '_').split('_', 1)
    A = None
    zoo_mode = None

    if parts[0].startswith('A'):
        try:
            A = int(parts[0][1:]) / 100.0
        except ValueError:
            pass
    if len(parts) > 1:
        zoo_mode = parts[1]

    return A, zoo_mode


def _add_experiment(run_dir: Path, algorithm: str, A: Optional[float],
                    zoo_mode: Optional[str], seed: Optional[int],
                    experiments: List[Dict]):
    """Find metrics.csv in run_dir (possibly in timestamp subdir) and add to list."""
    # Direct metrics.csv
    metrics = run_dir / "metrics.csv"
    metadata = run_dir / "metadata.json"

    if not metrics.exists():
        # Check timestamped subdirs
        for ts_dir in sorted(run_dir.glob("2*"), reverse=True):
            m = ts_dir / "metrics.csv"
            if m.exists():
                metrics = m
                md = ts_dir / "metadata.json"
                if md.exists():
                    metadata = md
                break

    if not metrics.exists():
        return

    experiments.append({
        'algorithm': algorithm,
        'A': A,
        'zoo_mode': zoo_mode,
        'seed': seed,
        'metrics_path': str(metrics),
        'metadata_path': str(metadata) if metadata.exists() else None,
    })


def load_metrics(path: str) -> List[Dict[str, float]]:
    """Load metrics.csv into list of dicts."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append({k: float(v) for k, v in row.items()})
            except (ValueError, TypeError):
                continue
    return rows


def load_metadata(path: str) -> Dict:
    """Load metadata.json."""
    with open(path) as f:
        return json.load(f)


def get_final_metrics(exp: Dict) -> Dict[str, float]:
    """Get the final row from metrics.csv."""
    rows = load_metrics(exp['metrics_path'])
    if not rows:
        return {}
    return rows[-1]


def millions(x, _):
    return f"{x / 1e6:.1f}M"


def plot_a_curve(experiments: List[Dict], output_dir: str):
    """Plot seeker win rate vs A — the main result figure.

    Separate curves per algorithm, error bars from seeds.
    """
    # Group by (algorithm, A, zoo_mode)
    from collections import defaultdict
    groups = defaultdict(list)

    for exp in experiments:
        if exp['A'] is None:
            continue
        key = (exp['algorithm'], exp['A'], exp['zoo_mode'])
        final = get_final_metrics(exp)
        if 'seeker_win_rate' in final:
            groups[key].append(final['seeker_win_rate'])

    if not groups:
        print("No data for A-curve plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    algorithms = sorted(set(k[0] for k in groups))
    zoo_modes = sorted(set(k[2] for k in groups if k[2] != 'selfplay'))

    colors = {'ppo': '#1f77b4', 'sac': '#ff7f0e'}
    linestyles = {'hider_only': '-', 'both': '--'}
    markers = {'hider_only': 'o', 'both': 's'}

    for algo in algorithms:
        for zm in zoo_modes:
            a_vals = []
            means = []
            stds = []

            for (a, z, mod), wrs in sorted(groups.items()):
                if a != algo or z != zm:
                    continue
                a_vals.append(mod)

            # Re-collect properly
            a_vals = sorted(set(k[1] for k in groups if k[0] == algo and k[2] == zm))
            for a_val in a_vals:
                wrs = groups[(algo, a_val, zm)]
                means.append(np.mean(wrs))
                stds.append(np.std(wrs) if len(wrs) > 1 else 0)

            if not a_vals:
                continue

            label = f"{algo.upper()} ({zm})"
            color = colors.get(algo, 'gray')
            ls = linestyles.get(zm, '-')
            marker = markers.get(zm, 'o')

            ax.errorbar(a_vals, means, yerr=stds,
                       label=label, color=color, linestyle=ls,
                       marker=marker, capsize=4, markersize=6)

    # Add selfplay baselines
    for algo in algorithms:
        sp_wrs = []
        for exp in experiments:
            if exp['algorithm'] == algo and exp['zoo_mode'] == 'selfplay':
                final = get_final_metrics(exp)
                if 'seeker_win_rate' in final:
                    sp_wrs.append(final['seeker_win_rate'])
        if sp_wrs:
            ax.axhline(np.mean(sp_wrs), color=colors.get(algo, 'gray'),
                      linestyle=':', alpha=0.5,
                      label=f"{algo.upper()} self-play (A=0)")

    ax.set_xlabel("A (zoo sampling probability)", fontsize=12)
    ax.set_ylabel("Seeker Win Rate", fontsize=12)
    ax.set_title("Effect of Opponent Diversity (A) on Agent Performance", fontsize=14)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.3)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "a_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_training_dynamics(experiments: List[Dict], output_dir: str):
    """Plot learning curves overlaid by A value, one subplot per algorithm."""
    from collections import defaultdict

    # Group experiments
    by_algo = defaultdict(list)
    for exp in experiments:
        if exp['A'] is not None and exp['zoo_mode'] == 'hider_only':
            by_algo[exp['algorithm']].append(exp)

    if not by_algo:
        print("No data for training dynamics plot.")
        return

    algorithms = sorted(by_algo.keys())
    fig, axes = plt.subplots(1, len(algorithms), figsize=(7 * len(algorithms), 5),
                              squeeze=False)

    cmap = plt.cm.viridis

    for col, algo in enumerate(algorithms):
        ax = axes[0, col]
        exps = by_algo[algo]

        # Get unique A values for colormap
        a_vals = sorted(set(e['A'] for e in exps))
        a_to_color = {a: cmap(i / max(len(a_vals) - 1, 1))
                      for i, a in enumerate(a_vals)}

        for exp in exps:
            rows = load_metrics(exp['metrics_path'])
            if not rows:
                continue
            ts = [r['timesteps'] for r in rows]
            wr = [r['seeker_win_rate'] for r in rows]
            color = a_to_color[exp['A']]
            seed_label = f" s{exp['seed']}" if exp['seed'] is not None else ""
            ax.plot(ts, wr, color=color, alpha=0.5, linewidth=1)

        # Add colorbar-like legend
        for a_val in a_vals:
            ax.plot([], [], color=a_to_color[a_val], linewidth=2,
                   label=f"A={a_val:.2f}")

        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Seeker Win Rate")
        ax.set_title(f"{algo.upper()} Training Dynamics (hider_only)")
        ax.axhline(0.5, color='gray', linestyle=':', alpha=0.3)
        ax.set_ylim(0, 1)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(millions))
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "training_dynamics.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_zoo_mode_comparison(experiments: List[Dict], output_dir: str):
    """Compare hider_only vs both zoo modes."""
    from collections import defaultdict

    # Group by (algorithm, A, zoo_mode) -> final win rates
    groups = defaultdict(list)
    for exp in experiments:
        if exp['A'] is None or exp['zoo_mode'] == 'selfplay':
            continue
        key = (exp['algorithm'], exp['A'], exp['zoo_mode'])
        final = get_final_metrics(exp)
        if 'seeker_win_rate' in final:
            groups[key].append(final['seeker_win_rate'])

    algorithms = sorted(set(k[0] for k in groups))
    if not algorithms:
        print("No data for zoo mode comparison.")
        return

    fig, axes = plt.subplots(1, len(algorithms), figsize=(7 * len(algorithms), 5),
                              squeeze=False)

    for col, algo in enumerate(algorithms):
        ax = axes[0, col]

        for zm, color, marker in [('hider_only', '#1f77b4', 'o'),
                                   ('both', '#ff7f0e', 's')]:
            a_vals = sorted(set(k[1] for k in groups if k[0] == algo and k[2] == zm))
            means = []
            stds = []
            for a_val in a_vals:
                wrs = groups[(algo, a_val, zm)]
                means.append(np.mean(wrs))
                stds.append(np.std(wrs) if len(wrs) > 1 else 0)

            if a_vals:
                ax.errorbar(a_vals, means, yerr=stds,
                           label=zm, color=color, marker=marker,
                           capsize=4, markersize=6)

        ax.set_xlabel("A (zoo sampling probability)")
        ax.set_ylabel("Seeker Win Rate")
        ax.set_title(f"{algo.upper()}: hider_only vs both")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color='gray', linestyle=':', alpha=0.3)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "zoo_mode_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_ppo_vs_sac(experiments: List[Dict], output_dir: str):
    """Direct PPO vs SAC comparison (the money plot).

    Only produced if both algorithms have data.
    """
    from collections import defaultdict

    groups = defaultdict(list)
    for exp in experiments:
        if exp['A'] is None or exp['zoo_mode'] != 'hider_only':
            continue
        key = (exp['algorithm'], exp['A'])
        final = get_final_metrics(exp)
        if 'seeker_win_rate' in final:
            groups[key].append(final['seeker_win_rate'])

    algorithms = sorted(set(k[0] for k in groups))
    if len(algorithms) < 2:
        print("Need both PPO and SAC data for comparison plot. Skipping.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'ppo': '#1f77b4', 'sac': '#ff7f0e'}
    markers = {'ppo': 'o', 'sac': 's'}

    for algo in algorithms:
        a_vals = sorted(set(k[1] for k in groups if k[0] == algo))
        means = [np.mean(groups[(algo, a)]) for a in a_vals]
        stds = [np.std(groups[(algo, a)]) if len(groups[(algo, a)]) > 1 else 0
                for a in a_vals]

        ax.errorbar(a_vals, means, yerr=stds,
                   label=algo.upper(), color=colors.get(algo, 'gray'),
                   marker=markers.get(algo, 'o'), capsize=4,
                   markersize=8, linewidth=2)

    ax.set_xlabel("A (zoo sampling probability)", fontsize=12)
    ax.set_ylabel("Seeker Win Rate", fontsize=12)
    ax.set_title("PPO vs SAC: A-Curve Comparison (hider_only zoo)", fontsize=14)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.3)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "ppo_vs_sac.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def print_summary_table(experiments: List[Dict]):
    """Print a summary table of all experiments."""
    from collections import defaultdict

    groups = defaultdict(list)
    for exp in experiments:
        key = (exp['algorithm'], exp['A'], exp['zoo_mode'])
        final = get_final_metrics(exp)
        if 'seeker_win_rate' in final:
            groups[key].append(final['seeker_win_rate'])

    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Algorithm':<8} {'A':<6} {'Zoo Mode':<12} {'Seeds':<6} "
          f"{'Win Rate':<12} {'Std':<8}")
    print("-" * 70)

    for key in sorted(groups.keys()):
        algo, A, zm = key
        wrs = groups[key]
        a_str = f"{A:.2f}" if A is not None else "N/A"
        zm_str = zm or "N/A"
        print(f"{algo:<8} {a_str:<6} {zm_str:<12} {len(wrs):<6} "
              f"{np.mean(wrs):<12.1%} {np.std(wrs):<8.3f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze A-sweep results")
    parser.add_argument("sweep_dir", type=str,
                        help="Path to sweep results directory")
    args = parser.parse_args()

    sweep_dir = args.sweep_dir

    print(f"Scanning {sweep_dir} for experiments...")
    experiments = discover_experiments(sweep_dir)

    if not experiments:
        print(f"No experiments found in {sweep_dir}")
        sys.exit(1)

    print(f"Found {len(experiments)} experiment runs.")
    print_summary_table(experiments)

    # Generate plots
    print("\nGenerating plots...")
    plot_a_curve(experiments, sweep_dir)
    plot_training_dynamics(experiments, sweep_dir)
    plot_zoo_mode_comparison(experiments, sweep_dir)
    plot_ppo_vs_sac(experiments, sweep_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
