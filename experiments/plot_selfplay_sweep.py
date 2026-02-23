#!/usr/bin/env python3
"""
Plot results from the self-play sweep: seeker_time_penalty × hider_speed_mult.

Generates:
1. Heatmaps of final seeker_win_rate, episode_length, rewards across the 2D grid
2. Line plots showing training curves grouped by each parameter
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

SWEEP_DIR = Path("experiments/results/selfplay_sweep")

STP_VALUES = [0.005, 0.01, 0.02, 0.05]  # absolute values (penalties are negative)
HSM_VALUES = [1.0, 1.05, 1.1, 1.15, 1.2]

# Build experiment names matching run_selfplay_sweep.py naming
EXPERIMENTS = {}
for stp in STP_VALUES:
    for hsm in HSM_VALUES:
        stp_str = f"{stp:.4f}".replace("0.", "").rstrip("0") or "0"
        hsm_str = f"{round(hsm * 100)}"
        name = f"STP{stp_str}_HSM{hsm_str}"
        EXPERIMENTS[name] = {"stp": stp, "hsm": hsm}

# Colors for line plots
STP_COLORS = {0.005: "#1f77b4", 0.01: "#ff7f0e", 0.02: "#2ca02c", 0.05: "#d62728"}
HSM_STYLES = {1.0: "-", 1.05: "--", 1.1: "-.", 1.15: ":", 1.2: (0, (3, 1, 1, 1))}


def load_metrics(name):
    """Load metrics.csv for a named experiment."""
    exp_dir = SWEEP_DIR / name
    candidates = sorted(exp_dir.glob("2*"), reverse=True)
    for d in candidates:
        csv_path = d / "metrics.csv"
        if csv_path.exists():
            rows = []
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append({k: float(v) for k, v in row.items()})
            return rows
    return []


def get_final_metrics(rows, tail_frac=0.1):
    """Average metrics over the last tail_frac of training."""
    if not rows:
        return None
    n = max(1, int(len(rows) * tail_frac))
    tail = rows[-n:]
    return {
        "seeker_win_rate": np.mean([r["seeker_win_rate"] for r in tail]),
        "episode_length_mean": np.mean([r["episode_length_mean"] for r in tail]),
        "seeker_reward_mean": np.mean([r["seeker_reward_mean"] for r in tail]),
        "hider_reward_mean": np.mean([r["hider_reward_mean"] for r in tail]),
    }


def millions(x, _):
    return f"{x / 1e6:.1f}M"


def plot_heatmaps(all_data):
    """Plot 2×2 heatmaps of final metrics across the parameter grid."""
    metrics = ["seeker_win_rate", "episode_length_mean",
               "seeker_reward_mean", "hider_reward_mean"]
    titles = ["Seeker Win Rate", "Episode Length",
              "Seeker Reward", "Hider Reward"]
    cmaps = ["RdYlGn", "viridis", "RdYlGn", "RdYlGn_r"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Self-Play Sweep: seeker_time_penalty × hider_speed_mult\n"
                 "(final 10% of training)", fontsize=13, fontweight="bold")

    for ax, metric, title, cmap in zip(axes.flat, metrics, titles, cmaps):
        grid = np.full((len(STP_VALUES), len(HSM_VALUES)), np.nan)

        for name, params in EXPERIMENTS.items():
            if name not in all_data or all_data[name] is None:
                continue
            si = STP_VALUES.index(params["stp"])
            hi = HSM_VALUES.index(params["hsm"])
            grid[si, hi] = all_data[name][metric]

        im = ax.imshow(grid, aspect="auto", origin="lower", cmap=cmap)
        fig.colorbar(im, ax=ax, shrink=0.8)

        ax.set_xticks(range(len(HSM_VALUES)))
        ax.set_xticklabels([f"{v:.1f}" for v in HSM_VALUES])
        ax.set_yticks(range(len(STP_VALUES)))
        ax.set_yticklabels([f"{v}" for v in STP_VALUES])
        ax.set_xlabel("hider_speed_mult")
        ax.set_ylabel("seeker_time_penalty (abs)")
        ax.set_title(title)

        # Annotate cells
        for si in range(len(STP_VALUES)):
            for hi in range(len(HSM_VALUES)):
                val = grid[si, hi]
                if not np.isnan(val):
                    fmt = f"{val:.2f}" if abs(val) < 100 else f"{val:.0f}"
                    ax.text(hi, si, fmt, ha="center", va="center",
                            fontsize=8, fontweight="bold",
                            color="white" if abs(val) > np.nanmax(abs(grid)) * 0.7 else "black")

    plt.tight_layout()
    out_path = SWEEP_DIR / "heatmaps.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved heatmaps to {out_path}")
    plt.close()


def plot_training_curves(all_rows):
    """Plot training curves: one subplot per metric, colored by stp, styled by hsm."""
    metrics = ["seeker_win_rate", "episode_length_mean",
               "seeker_reward_mean", "hider_reward_mean"]
    titles = ["Seeker Win Rate", "Episode Length",
              "Seeker Reward", "Hider Reward"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Self-Play Sweep: Training Curves", fontsize=13, fontweight="bold")

    for ax, metric, title in zip(axes.flat, metrics, titles):
        for name, rows in all_rows.items():
            if not rows:
                continue
            params = EXPERIMENTS[name]
            ts = [r["timesteps"] for r in rows]
            vals = [r[metric] for r in rows]
            ax.plot(ts, vals,
                    color=STP_COLORS[params["stp"]],
                    ls=HSM_STYLES[params["hsm"]],
                    alpha=0.7, linewidth=1.2,
                    label=f"stp={params['stp']}, hsm={params['hsm']:.1f}")

        ax.set_xlabel("Timesteps")
        ax.set_title(title)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(millions))
        ax.grid(True, alpha=0.3)

        if metric == "seeker_win_rate":
            ax.set_ylim(0, 1)
            ax.axhline(0.5, color="gray", ls=":", alpha=0.5)
        elif "reward" in metric:
            ax.axhline(0, color="gray", ls=":", alpha=0.5)

    # Single legend outside
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=7,
               bbox_to_anchor=(1.15, 0.5))

    plt.tight_layout()
    out_path = SWEEP_DIR / "training_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved training curves to {out_path}")
    plt.close()


def plot_line_slices(all_data):
    """Plot 1D slices: metric vs one param, lines for the other param."""
    metrics = ["seeker_win_rate", "episode_length_mean"]
    titles = ["Seeker Win Rate", "Episode Length"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Self-Play Sweep: Parameter Slices (final 10%)",
                 fontsize=13, fontweight="bold")

    # Left column: vary stp, lines per hsm
    for row, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[row, 0]
        for hsm in HSM_VALUES:
            xs, ys = [], []
            for stp in STP_VALUES:
                stp_str = f"{stp:.4f}".replace("0.", "").rstrip("0") or "0"
                hsm_str = f"{round(hsm * 100)}"
                name = f"STP{stp_str}_HSM{hsm_str}"
                if name in all_data and all_data[name] is not None:
                    xs.append(stp)
                    ys.append(all_data[name][metric])
            if xs:
                ax.plot(xs, ys, "o-", label=f"hsm={hsm:.1f}")
        ax.set_xscale("log")
        ax.set_xlabel("seeker_time_penalty (abs)")
        ax.set_ylabel(title)
        ax.set_title(f"{title} vs seeker_time_penalty")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Right column: vary hsm, lines per stp
    for row, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[row, 1]
        for stp in STP_VALUES:
            xs, ys = [], []
            for hsm in HSM_VALUES:
                stp_str = f"{stp:.4f}".replace("0.", "").rstrip("0") or "0"
                hsm_str = f"{round(hsm * 100)}"
                name = f"STP{stp_str}_HSM{hsm_str}"
                if name in all_data and all_data[name] is not None:
                    xs.append(hsm)
                    ys.append(all_data[name][metric])
            if xs:
                ax.plot(xs, ys, "o-", label=f"stp={stp}")
        ax.set_xlabel("hider_speed_mult")
        ax.set_ylabel(title)
        ax.set_title(f"{title} vs hider_speed_mult")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = SWEEP_DIR / "parameter_slices.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved parameter slices to {out_path}")
    plt.close()


def main():
    print("Loading metrics...")
    all_rows = {}
    all_final = {}

    for name in EXPERIMENTS:
        rows = load_metrics(name)
        if rows:
            all_rows[name] = rows
            all_final[name] = get_final_metrics(rows)
            print(f"  {name}: {len(rows)} rows, "
                  f"{rows[-1]['timesteps']:.0f} steps")
        else:
            print(f"  {name}: no data")

    if not all_rows:
        print("No data found!")
        return

    print(f"\nLoaded {len(all_rows)}/{len(EXPERIMENTS)} experiments")

    plot_heatmaps(all_final)
    plot_training_curves(all_rows)
    plot_line_slices(all_final)

    print("\nDone!")


if __name__ == "__main__":
    main()
