#!/usr/bin/env python3
"""Generate publication-quality plots for the cross-method gauntlet results."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

RESULTS = Path("experiments/results/cross_method_gauntlet/gauntlet_results.json")
OUT_DIR = Path("docs/reward_shaping")

# Readable method display names
METHOD_DISPLAY = {
    "fr/PPO": "FR\nPPO",
    "fr/SAC": "FR\nSAC",
    "fr_v2/PPO": "FR v2\nPPO",
    "fr_v2/SAC": "FR v2\nSAC",
    "reward/PPO": "Reward\nPPO",
    "reward/SAC": "Reward\nSAC",
    "selfplay": "Self-\nplay",
    "zoo": "Zoo",
    "zoo_shaped": "Zoo\nshaped",
}

# Sort order: SAC methods first (strongest), then PPO, then zoo
METHOD_ORDER = [
    "fr_v2/SAC", "reward/SAC", "fr/SAC",        # SAC tier
    "selfplay", "reward/PPO",                     # mid tier
    "fr/PPO", "zoo_shaped", "zoo", "fr_v2/PPO",  # PPO/zoo tier
]

# Tier boundaries for visual grouping
TIER_BOUNDARIES = [3, 5]

# Custom diverging colormap: red (hider wins) -> white (50%) -> green (seeker wins)
CMAP = LinearSegmentedColormap.from_list(
    "tag_wr", ["#d73027", "#f4a582", "#ffffff", "#a1d99b", "#1a9850"])


def load_results():
    with open(RESULTS) as f:
        return json.load(f)


def compute_method_h2h(results):
    """Aggregate win matrix into method-level head-to-head."""
    labels = results["labels"]
    methods = results["methods"]
    W = np.array(results["win_matrix"])

    unique = METHOD_ORDER
    nm = len(unique)
    h2h = np.zeros((nm, nm))
    counts = np.zeros((nm, nm))

    method_to_idx = {m: i for i, m in enumerate(unique)}

    for i in range(len(labels)):
        for j in range(len(labels)):
            mi = method_to_idx.get(methods[i])
            mj = method_to_idx.get(methods[j])
            if mi is not None and mj is not None:
                h2h[mi, mj] += W[i, j]
                counts[mi, mj] += 1

    mask = counts > 0
    h2h[mask] /= counts[mask]
    return unique, h2h


def compute_grouped_matrix(results):
    """Reorder the full 27x27 matrix grouped by method."""
    labels = results["labels"]
    methods = results["methods"]
    W = np.array(results["win_matrix"])

    ordered_idx = []
    group_sizes = []
    for m in METHOD_ORDER:
        indices = [i for i, x in enumerate(methods) if x == m]
        ordered_idx.extend(indices)
        group_sizes.append(len(indices))

    n = len(ordered_idx)
    grouped_W = np.zeros((n, n))
    grouped_labels = []
    grouped_methods = []

    for new_i, old_i in enumerate(ordered_idx):
        for new_j, old_j in enumerate(ordered_idx):
            grouped_W[new_i, new_j] = W[old_i, old_j]
        grouped_labels.append(labels[old_i])
        grouped_methods.append(methods[old_i])

    return grouped_W, grouped_labels, grouped_methods, group_sizes


def plot_method_heatmap(unique_methods, h2h):
    """9x9 method head-to-head heatmap -- the main summary visual."""
    nm = len(unique_methods)
    fig, ax = plt.subplots(figsize=(9, 8))

    im = ax.imshow(h2h * 100, cmap=CMAP, vmin=0, vmax=100, aspect="equal")

    display = [METHOD_DISPLAY.get(m, m) for m in unique_methods]
    ax.set_xticks(range(nm))
    ax.set_yticks(range(nm))
    ax.set_xticklabels(display, fontsize=9, ha="center")
    ax.set_yticklabels(display, fontsize=9, va="center")

    for i in range(nm):
        for j in range(nm):
            v = h2h[i, j] * 100
            color = "white" if v < 25 or v > 75 else "black"
            ax.text(j, i, f"{v:.0f}%", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=color)

    for b in TIER_BOUNDARIES:
        ax.axhline(b - 0.5, color="#333333", linewidth=2, linestyle="-")
        ax.axvline(b - 0.5, color="#333333", linewidth=2, linestyle="-")

    tier_labels = [
        (1, "SAC tier"),
        (4, "Mid tier"),
        (6.5, "PPO / Zoo tier"),
    ]
    for y, label in tier_labels:
        ax.text(nm + 0.3, y, label, fontsize=8, color="#555555",
                va="center", ha="left", style="italic")

    ax.set_xlabel("Hider method", fontsize=12, labelpad=10)
    ax.set_ylabel("Seeker method", fontsize=12, labelpad=10)
    ax.set_title("Cross-Method Gauntlet: Seeker Win Rate by Training Method",
                 fontsize=13, fontweight="bold", pad=15)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.12)
    cbar.set_label("Seeker Win Rate (%)", fontsize=10)

    plt.tight_layout()
    out = OUT_DIR / "xmethod_h2h_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_grouped_full_heatmap(grouped_W, grouped_labels, grouped_methods, group_sizes):
    """Full 27x27 heatmap reordered by method with group separators."""
    n = len(grouped_labels)
    fig, ax = plt.subplots(figsize=(16, 14))

    im = ax.imshow(grouped_W * 100, cmap=CMAP, vmin=0, vmax=100, aspect="equal")

    short = []
    for l in grouped_labels:
        parts = l.split("/")
        if len(parts) >= 3:
            short.append(f"{parts[-2]}\n{parts[-1]}")
        elif len(parts) == 2:
            short.append(parts[-1])
        else:
            short.append(l)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short, fontsize=5.5, rotation=90, ha="center")
    ax.set_yticklabels(short, fontsize=5.5, va="center")

    method_colors = {}
    cmap_methods = plt.colormaps.get_cmap("tab10")
    for mi, m in enumerate(METHOD_ORDER):
        method_colors[m] = cmap_methods(mi / len(METHOD_ORDER))

    for i, m in enumerate(grouped_methods):
        c = method_colors.get(m, "black")
        ax.get_yticklabels()[i].set_color(c)
        ax.get_xticklabels()[i].set_color(c)

    for i in range(n):
        for j in range(n):
            v = grouped_W[i, j] * 100
            color = "white" if v < 25 or v > 75 else "black"
            ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                    fontsize=4.5, color=color)

    # Group separator lines
    pos = 0
    for gs in group_sizes[:-1]:
        pos += gs
        ax.axhline(pos - 0.5, color="#333333", linewidth=1.5)
        ax.axvline(pos - 0.5, color="#333333", linewidth=1.5)

    # Tier boundary lines (thicker)
    for boundary_after in TIER_BOUNDARIES:
        cumsum = sum(group_sizes[:boundary_after])
        ax.axhline(cumsum - 0.5, color="black", linewidth=3)
        ax.axvline(cumsum - 0.5, color="black", linewidth=3)

    # Method group labels on the left
    pos = 0
    for gi, m in enumerate(METHOD_ORDER):
        mid = pos + group_sizes[gi] / 2 - 0.5
        ax.text(-1.5, mid, METHOD_DISPLAY.get(m, m).replace("\n", " "),
                fontsize=6.5, color=method_colors.get(m, "black"),
                va="center", ha="right", fontweight="bold")
        pos += group_sizes[gi]

    ax.set_title("Cross-Method Gauntlet: All 27 Agents (grouped by method)",
                 fontsize=13, fontweight="bold", pad=15)

    cbar = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("Seeker Win Rate (%)", fontsize=10)

    plt.tight_layout()
    out = OUT_DIR / "xmethod_full_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_strength_bars(results):
    """Bar chart: seeker and hider strength per method, sorted by combined."""
    methods = results["methods"]
    W = np.array(results["win_matrix"])

    seeker_str = W.mean(axis=1)
    hider_str = 1 - W.mean(axis=0)

    stats = []
    for m in METHOD_ORDER:
        idx = [i for i, x in enumerate(methods) if x == m]
        if not idx:
            continue
        s = np.mean([seeker_str[i] for i in idx])
        h = np.mean([hider_str[i] for i in idx])
        stats.append((m, s, h, (s + h) / 2))

    stats.sort(key=lambda x: -x[3])

    fig, ax = plt.subplots(figsize=(10, 5))

    names = [METHOD_DISPLAY.get(s[0], s[0]).replace("\n", " ") for s in stats]
    seekers = [s[1] for s in stats]
    hiders = [s[2] for s in stats]
    combined = [s[3] for s in stats]

    x = np.arange(len(stats))
    w = 0.3

    ax.bar(x - w/2, seekers, w, label="Seeker strength (mean WR)",
           color="#f85149", alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.bar(x + w/2, hiders, w, label="Hider strength (mean survival)",
           color="#58a6ff", alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.plot(x, combined, "ko-", markersize=6, linewidth=1.5, label="Combined",
            zorder=5)

    for i in range(len(stats)):
        ax.text(i - w/2, seekers[i] + 0.02, f"{seekers[i]:.0%}",
                ha="center", fontsize=7, color="#f85149", fontweight="bold")
        ax.text(i + w/2, hiders[i] + 0.02, f"{hiders[i]:.0%}",
                ha="center", fontsize=7, color="#58a6ff", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Strength", fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.set_title("Agent Strength by Training Method", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)

    ax.axvspan(-0.5, 2.5, alpha=0.06, color="green", zorder=0)
    ax.text(1, 1.04, "SAC tier", ha="center", fontsize=8, color="green", style="italic")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = OUT_DIR / "xmethod_strength.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def main():
    results = load_results()

    unique_methods, h2h = compute_method_h2h(results)
    plot_method_heatmap(unique_methods, h2h)

    grouped_W, grouped_labels, grouped_methods, group_sizes = compute_grouped_matrix(results)
    plot_grouped_full_heatmap(grouped_W, grouped_labels, grouped_methods, group_sizes)

    plot_strength_bars(results)

    print("All plots generated.")


if __name__ == "__main__":
    main()
