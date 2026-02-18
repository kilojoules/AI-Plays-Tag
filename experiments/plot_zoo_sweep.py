#!/usr/bin/env python3
"""Plot training progress for zoo sweep experiments."""
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

SWEEP_DIR = Path("experiments/results/zoo_sweep")
EXPERIMENTS = ["A05_hider_only", "A05_both", "A10_hider_only", "A10_both", "A20_hider_only", "A20_both"]
LABELS = {
    "A05_hider_only": "A=0.05 hider zoo",
    "A05_both": "A=0.05 both zoos",
    "A10_hider_only": "A=0.10 hider zoo",
    "A10_both": "A=0.10 both zoos",
    "A20_hider_only": "A=0.20 hider zoo",
    "A20_both": "A=0.20 both zoos",
}
# Paired colors: hider_only solid, both dashed
STYLES = {
    "A05_hider_only": dict(color="#1f77b4", ls="-"),
    "A05_both":       dict(color="#1f77b4", ls="--"),
    "A10_hider_only": dict(color="#ff7f0e", ls="-"),
    "A10_both":       dict(color="#ff7f0e", ls="--"),
    "A20_hider_only": dict(color="#2ca02c", ls="-"),
    "A20_both":       dict(color="#2ca02c", ls="--"),
}


def load_metrics(name):
    exp_dir = SWEEP_DIR / name
    # Find the timestamped subdir
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


def millions(x, _):
    return f"{x / 1e6:.1f}M"


def main():
    data = {}
    for name in EXPERIMENTS:
        rows = load_metrics(name)
        if rows:
            data[name] = rows
            print(f"{name}: {len(rows)} rows, {rows[-1]['timesteps']:.0f} steps")

    if not data:
        print("No data found!")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Zoo Training Sweep Progress", fontsize=14, fontweight="bold")

    # Plot 1: Seeker win rate
    ax = axes[0, 0]
    for name, rows in data.items():
        ts = [r["timesteps"] for r in rows]
        vals = [r["seeker_win_rate"] for r in rows]
        ax.plot(ts, vals, label=LABELS[name], **STYLES[name], alpha=0.8)
    ax.set_ylabel("Seeker Win Rate")
    ax.set_xlabel("Timesteps")
    ax.axhline(0.5, color="gray", ls=":", alpha=0.5)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(millions))
    ax.legend(fontsize=7)
    ax.set_title("Seeker Win Rate")

    # Plot 2: Episode length
    ax = axes[0, 1]
    for name, rows in data.items():
        ts = [r["timesteps"] for r in rows]
        vals = [r["episode_length_mean"] for r in rows]
        ax.plot(ts, vals, label=LABELS[name], **STYLES[name], alpha=0.8)
    ax.set_ylabel("Mean Episode Length")
    ax.set_xlabel("Timesteps")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(millions))
    ax.set_title("Episode Length")

    # Plot 3: Seeker reward
    ax = axes[1, 0]
    for name, rows in data.items():
        ts = [r["timesteps"] for r in rows]
        vals = [r["seeker_reward_mean"] for r in rows]
        ax.plot(ts, vals, label=LABELS[name], **STYLES[name], alpha=0.8)
    ax.set_ylabel("Mean Reward")
    ax.set_xlabel("Timesteps")
    ax.axhline(0, color="gray", ls=":", alpha=0.5)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(millions))
    ax.set_title("Seeker Reward")

    # Plot 4: Hider reward
    ax = axes[1, 1]
    for name, rows in data.items():
        ts = [r["timesteps"] for r in rows]
        vals = [r["hider_reward_mean"] for r in rows]
        ax.plot(ts, vals, label=LABELS[name], **STYLES[name], alpha=0.8)
    ax.set_ylabel("Mean Reward")
    ax.set_xlabel("Timesteps")
    ax.axhline(0, color="gray", ls=":", alpha=0.5)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(millions))
    ax.set_title("Hider Reward")

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = SWEEP_DIR / "training_progress.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
