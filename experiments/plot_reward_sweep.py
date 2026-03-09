#!/usr/bin/env python3
"""Generate all plots for the reward shaping study docs page."""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

BASE = Path("experiments/results/reward_sweep")
OUT = Path("docs/reward_shaping")
OUT.mkdir(parents=True, exist_ok=True)

# Short labels for plots
SHORT = {
    "R0_baseline/ppo": "R0 Base\nPPO",
    "R0_baseline/sac": "R0 Base\nSAC",
    "R1_seeker_pursuit/ppo": "R1 Pursuit\nPPO",
    "R1_seeker_pursuit/sac": "R1 Pursuit\nSAC",
    "R2_hider_active/ppo": "R2 Active\nPPO",
    "R2_hider_active/sac": "R2 Active\nSAC",
    "R3_both_shaped/ppo": "R3 Both\nPPO",
    "R3_both_shaped/sac": "R3 Both\nSAC",
    "R4_sparse/ppo": "R4 Sparse\nPPO",
    "R4_sparse/sac": "R4 Sparse\nSAC",
    "R5_escalating/ppo": "R5 Escal.\nPPO",
    "R5_escalating/sac": "R5 Escal.\nSAC",
    "R6_coverage/ppo": "R6 Cover.\nPPO",
    "R6_coverage/sac": "R6 Cover.\nSAC",
    "R7_kitchen_sink/ppo": "R7 Kitchen\nPPO",
    "R7_kitchen_sink/sac": "R7 Kitchen\nSAC",
}

PRESET_SHORT = {
    "R0_baseline": "R0 Baseline",
    "R1_seeker_pursuit": "R1 Pursuit",
    "R2_hider_active": "R2 Active",
    "R3_both_shaped": "R3 Both",
    "R4_sparse": "R4 Sparse",
    "R5_escalating": "R5 Escalating",
    "R6_coverage": "R6 Coverage",
    "R7_kitchen_sink": "R7 Kitchen Sink",
}


def plot_gauntlet_heatmap():
    """16x16 win rate heatmap."""
    with open(BASE / "gauntlet" / "gauntlet_results.json") as f:
        data = json.load(f)

    configs = data["configs"]
    win_matrix = np.array(data["win_matrix"])
    n = len(configs)
    labels = [SHORT.get(c, c) for c in configs]

    fig, ax = plt.subplots(figsize=(14, 12))
    cmap = plt.cm.RdYlGn
    im = ax.imshow(win_matrix * 100, cmap=cmap, vmin=0, vmax=100, aspect="equal")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Hider Config", fontsize=11, labelpad=10)
    ax.set_ylabel("Seeker Config", fontsize=11, labelpad=10)
    ax.set_title("Cross-Config Gauntlet: Seeker Win Rate (%)", fontsize=13, pad=12)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = win_matrix[i, j] * 100
            color = "white" if val < 30 or val > 70 else "black"
            ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                    fontsize=6, color=color, fontweight="bold")

    # Add PPO/SAC separator lines
    for k in range(1, n):
        if k % 2 == 0:
            ax.axhline(k - 0.5, color="white", linewidth=0.5, alpha=0.3)
            ax.axvline(k - 0.5, color="white", linewidth=0.5, alpha=0.3)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="Seeker Win Rate (%)")
    plt.tight_layout()
    plt.savefig(OUT / "gauntlet_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved gauntlet_heatmap.png")


def plot_strength_bars():
    """Seeker and hider strength bar charts, grouped by preset with PPO/SAC side by side."""
    with open(BASE / "gauntlet" / "gauntlet_results.json") as f:
        data = json.load(f)

    configs = data["configs"]
    win_matrix = np.array(data["win_matrix"])

    # Compute strengths
    seeker_wr = {c: win_matrix[i, :].mean() for i, c in enumerate(configs)}
    hider_surv = {c: 1.0 - win_matrix[:, j].mean() for j, c in enumerate(configs)}

    presets = list(PRESET_SHORT.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Seeker strength
    ax = axes[0]
    x = np.arange(len(presets))
    width = 0.35
    ppo_vals = [seeker_wr.get(f"{p}/ppo", 0) * 100 for p in presets]
    sac_vals = [seeker_wr.get(f"{p}/sac", 0) * 100 for p in presets]
    bars1 = ax.bar(x - width/2, ppo_vals, width, label="PPO", color="#4493f8", alpha=0.85)
    bars2 = ax.bar(x + width/2, sac_vals, width, label="SAC", color="#f85149", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([PRESET_SHORT[p] for p in presets], fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("Mean Win Rate as Seeker (%)")
    ax.set_title("Seeker Strength", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 75)
    ax.grid(axis="y", alpha=0.3)

    # Hider strength
    ax = axes[1]
    ppo_vals = [hider_surv.get(f"{p}/ppo", 0) * 100 for p in presets]
    sac_vals = [hider_surv.get(f"{p}/sac", 0) * 100 for p in presets]
    bars1 = ax.bar(x - width/2, ppo_vals, width, label="PPO", color="#4493f8", alpha=0.85)
    bars2 = ax.bar(x + width/2, sac_vals, width, label="SAC", color="#f85149", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([PRESET_SHORT[p] for p in presets], fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("Mean Survival Rate as Hider (%)")
    ax.set_title("Hider Strength", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT / "strength_bars.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved strength_bars.png")


def plot_learning_curves():
    """Win rate and episode length learning curves, one subplot per preset."""
    presets = list(PRESET_SHORT.keys())

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey="row")

    for idx, preset in enumerate(presets):
        ax = axes[idx // 4, idx % 4]

        for algo, color, ls in [("ppo", "#4493f8", "-"), ("sac", "#f85149", "-")]:
            all_wr = []
            all_steps = []
            for seed in [0, 1, 2]:
                task_dir = BASE / preset / algo / f"seed_{seed}"
                # Find latest run
                runs = sorted(task_dir.glob("2026*"))
                if not runs:
                    continue
                csv_path = runs[-1] / "metrics.csv"
                if not csv_path.exists():
                    continue
                try:
                    df = pd.read_csv(csv_path)
                except Exception:
                    continue

                # PPO has "update" and "timesteps" columns; SAC has "timesteps"
                if "timesteps" in df.columns:
                    steps = df["timesteps"].values
                elif "update" in df.columns:
                    # Estimate from update number
                    steps = df.iloc[:, 1].values if df.shape[1] > 1 else np.arange(len(df))
                else:
                    continue

                wr_col = "seeker_win_rate"
                if wr_col not in df.columns:
                    continue

                wr = df[wr_col].values
                all_wr.append((steps, wr))

            if not all_wr:
                continue

            # Interpolate to common x axis
            max_steps = max(s[-1] for s, _ in all_wr)
            x_common = np.linspace(0, max_steps, 200)
            interp_wr = []
            for steps, wr in all_wr:
                interp_wr.append(np.interp(x_common, steps, wr))
            interp_wr = np.array(interp_wr)
            mean_wr = interp_wr.mean(axis=0)
            std_wr = interp_wr.std(axis=0)

            ax.plot(x_common / 1e6, mean_wr * 100, color=color, linewidth=1.5,
                    label=algo.upper())
            ax.fill_between(x_common / 1e6, (mean_wr - std_wr) * 100,
                           (mean_wr + std_wr) * 100, color=color, alpha=0.15)

        ax.set_title(PRESET_SHORT[preset], fontsize=10)
        ax.set_ylim(0, 100)
        ax.axhline(50, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
        ax.grid(alpha=0.2)
        if idx % 4 == 0:
            ax.set_ylabel("Seeker Win Rate (%)")
        if idx >= 4:
            ax.set_xlabel("Timesteps (M)")
        if idx == 0:
            ax.legend(fontsize=8)

    plt.suptitle("Learning Curves: Seeker Win Rate over Training (mean ± std, 3 seeds)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(OUT / "learning_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved learning_curves.png")


def plot_algo_comparison():
    """Scatter plot: PPO vs SAC seeker/hider strength per preset."""
    with open(BASE / "gauntlet" / "gauntlet_results.json") as f:
        data = json.load(f)

    configs = data["configs"]
    win_matrix = np.array(data["win_matrix"])

    seeker_wr = {c: win_matrix[i, :].mean() * 100 for i, c in enumerate(configs)}
    hider_surv = {c: (1.0 - win_matrix[:, j].mean()) * 100 for j, c in enumerate(configs)}

    presets = list(PRESET_SHORT.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(presets)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Seeker: PPO vs SAC
    ax = axes[0]
    for i, preset in enumerate(presets):
        ppo_key = f"{preset}/ppo"
        sac_key = f"{preset}/sac"
        if ppo_key in seeker_wr and sac_key in seeker_wr:
            ax.scatter(seeker_wr[ppo_key], seeker_wr[sac_key],
                      color=colors[i], s=100, zorder=5, edgecolor="white", linewidth=0.5)
            ax.annotate(PRESET_SHORT[preset], (seeker_wr[ppo_key], seeker_wr[sac_key]),
                       fontsize=7, ha="left", va="bottom", xytext=(3, 3),
                       textcoords="offset points")
    lim = [0, 70]
    ax.plot(lim, lim, "k--", alpha=0.3, linewidth=0.5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("PPO Seeker Win Rate (%)")
    ax.set_ylabel("SAC Seeker Win Rate (%)")
    ax.set_title("Seeker: PPO vs SAC")
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)

    # Hider: PPO vs SAC
    ax = axes[1]
    for i, preset in enumerate(presets):
        ppo_key = f"{preset}/ppo"
        sac_key = f"{preset}/sac"
        if ppo_key in hider_surv and sac_key in hider_surv:
            ax.scatter(hider_surv[ppo_key], hider_surv[sac_key],
                      color=colors[i], s=100, zorder=5, edgecolor="white", linewidth=0.5)
            ax.annotate(PRESET_SHORT[preset], (hider_surv[ppo_key], hider_surv[sac_key]),
                       fontsize=7, ha="left", va="bottom", xytext=(3, 3),
                       textcoords="offset points")
    lim = [0, 100]
    ax.plot(lim, lim, "k--", alpha=0.3, linewidth=0.5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("PPO Hider Survival (%)")
    ax.set_ylabel("SAC Hider Survival (%)")
    ax.set_title("Hider: PPO vs SAC")
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)

    plt.suptitle("Algorithm Comparison: PPO vs SAC by Reward Preset", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(OUT / "algo_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved algo_comparison.png")


if __name__ == "__main__":
    plot_gauntlet_heatmap()
    plot_strength_bars()
    plot_learning_curves()
    plot_algo_comparison()
    print(f"\nAll plots saved to {OUT}/")
