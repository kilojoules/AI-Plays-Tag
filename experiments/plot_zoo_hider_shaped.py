#!/usr/bin/env python3
"""
Generate all 4 README plots for the zoo_hider_shaped sweep.

Win rate plots use gauntlet evaluation data (20 eval episodes per matchup,
final seeker vs final hider) rather than noisy training metrics.

Outputs:
  experiments/results/zoo_hider_shaped/seeker_wr_vs_A.png
  experiments/results/zoo_hider_shaped/zoo_improvement_summary.png
  experiments/results/zoo_hider_shaped/zoo_improvement_by_difficulty.png
  experiments/results/zoo_hider_shaped/gauntlet/fr_vs_A.png
"""
import json
import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path("experiments/results/zoo_hider_shaped")
GAUNTLET_DIR = BASE_DIR / "gauntlet"

STPS = [0.005, 0.01, 0.02, 0.05]
HSMS = [1.0, 1.05, 1.1, 1.15, 1.2]
A_VALUES = [0.05, 0.1, 0.2, 0.3, 0.5]
SAMPLING_MODES = ["uniform", "thompson_loss"]
N_EVAL = 20  # episodes per gauntlet matchup


def stp_str(stp):
    return f"{stp:.4f}".replace("0.", "").rstrip("0") or "0"


def config_name(stp, hsm):
    return f"STP{stp_str(stp)}_HSM{round(hsm * 100)}"


def load_gauntlet_data():
    """Load final win rates and FR from gauntlet results across all seeds.

    Returns:
        wr_data: dict (config_name, stp, hsm, A, sampling) -> list of win rates (one per seed)
        fr_data: dict (config_name, stp, hsm, A, sampling) -> list of FR values (one per seed)
    """
    wr_data = {}
    fr_data = {}
    for jf in sorted(GAUNTLET_DIR.rglob("gauntlet_result.json")):
        with open(jf) as f:
            r = json.load(f)
        wm = r["win_matrix"]
        wr = wm[-1][-1]  # final seeker vs final hider
        key = (r["config_name"], r["stp"], r["hsm"], r["A"], r["sampling"])
        wr_data.setdefault(key, []).append(wr)
        fr_data.setdefault(key, []).append(r["fr_full"])
    return wr_data, fr_data


# ── Plot 1: Seeker WR vs A (4×5 grid) ──────────────────────────────
def plot_wr_vs_A(wr_data):
    fig, axes = plt.subplots(len(STPS), len(HSMS), figsize=(19, 13),
                              sharex=True, sharey=True)
    a_pct = [int(a * 100) for a in A_VALUES]

    for i, stp in enumerate(STPS):
        for j, hsm in enumerate(HSMS):
            ax = axes[i, j]
            cn = config_name(stp, hsm)

            for sampling, color, label in [
                ("uniform", "#00bcd4", "uniform"),
                ("thompson_loss", "#e91e63", "thompson_loss"),
            ]:
                means, ses = [], []
                for A in A_VALUES:
                    key = (cn, stp, hsm, A, sampling)
                    vals = wr_data.get(key, [])
                    if vals:
                        means.append(np.mean(vals))
                        ses.append(np.std(vals, ddof=1) / np.sqrt(len(vals)))
                    else:
                        means.append(np.nan)
                        ses.append(0)
                ax.errorbar(a_pct, means, yerr=ses, marker="o", markersize=4,
                           capsize=3, label=label, color=color, linewidth=1.5)

            ax.set_ylim(-0.05, 1.05)
            ax.set_xticks(a_pct)
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.set_title(f"HSM={hsm:.2f}", fontsize=11)
            if j == 0:
                ax.set_ylabel(f"STP={stp}\nSeeker WR", fontsize=10)
            if i == len(STPS) - 1:
                ax.set_xlabel("A (%)", fontsize=10)
            if i == 0 and j == len(HSMS) - 1:
                ax.legend(fontsize=8, loc="lower left")

    fig.suptitle("Seeker Win Rate vs. Zoo Mixing Rate (A) — Hider-Shaped",
                 fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = BASE_DIR / "seeker_wr_vs_A.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Plot 2: Zoo improvement summary (heatmap) ──────────────────────
def plot_improvement_summary(wr_data):
    baseline_A = 0.05
    zoo_As = [a for a in A_VALUES if a > baseline_A]

    improvements = np.full((len(STPS), len(HSMS)), np.nan)
    for i, stp in enumerate(STPS):
        for j, hsm in enumerate(HSMS):
            cn = config_name(stp, hsm)
            bl_vals = []
            for s in SAMPLING_MODES:
                key = (cn, stp, hsm, baseline_A, s)
                bl_vals.extend(wr_data.get(key, []))
            if not bl_vals:
                continue
            bl_mean = np.mean(bl_vals)

            best_zoo = bl_mean
            for A in zoo_As:
                for s in SAMPLING_MODES:
                    key = (cn, stp, hsm, A, s)
                    vals = wr_data.get(key, [])
                    if vals:
                        best_zoo = max(best_zoo, np.mean(vals))
            improvements[i, j] = (best_zoo - bl_mean) * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(improvements, cmap="RdYlGn", vmin=-10, vmax=40,
                   aspect="auto", origin="upper")

    for i in range(len(STPS)):
        for j in range(len(HSMS)):
            val = improvements[i, j]
            if not np.isnan(val):
                color = "white" if abs(val) > 25 else "black"
                ax.text(j, i, f"{val:+.1f}", ha="center", va="center",
                       fontsize=11, fontweight="bold", color=color)

    ax.set_xticks(range(len(HSMS)))
    ax.set_xticklabels([f"{h:.2f}" for h in HSMS])
    ax.set_yticks(range(len(STPS)))
    ax.set_yticklabels([str(s) for s in STPS])
    ax.set_xlabel("Hider Speed Multiplier (HSM)", fontsize=12)
    ax.set_ylabel("Seeker Time Penalty (STP)", fontsize=12)
    ax.set_title("Zoo Improvement (pp) vs. A=5% Baseline — Hider-Shaped", fontsize=13)
    fig.colorbar(im, ax=ax, label="Win Rate Improvement (pp)")
    fig.tight_layout()
    out = BASE_DIR / "zoo_improvement_summary.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Plot 3: Improvement by difficulty (bar chart) ──────────────────
def plot_improvement_by_difficulty(wr_data):
    baseline_A = 0.05
    zoo_As = [a for a in A_VALUES if a > baseline_A]

    configs = []
    for stp in STPS:
        for hsm in HSMS:
            cn = config_name(stp, hsm)
            bl_vals = []
            for s in SAMPLING_MODES:
                key = (cn, stp, hsm, baseline_A, s)
                bl_vals.extend(wr_data.get(key, []))
            if not bl_vals:
                continue
            bl_mean = np.mean(bl_vals)

            best_zoo_mean = bl_mean
            best_A = baseline_A
            best_sampling = ""
            for A in zoo_As:
                for s in SAMPLING_MODES:
                    key = (cn, stp, hsm, A, s)
                    vals = wr_data.get(key, [])
                    if vals and np.mean(vals) > best_zoo_mean:
                        best_zoo_mean = np.mean(vals)
                        best_A = A
                        best_sampling = s

            # SE of improvement: pooled from baseline and best zoo seeds
            best_vals = wr_data.get((cn, stp, hsm, best_A, best_sampling), []) if best_A != baseline_A else bl_vals
            se_bl = np.std(bl_vals, ddof=1) / np.sqrt(len(bl_vals)) if len(bl_vals) > 1 else 0
            se_zoo = np.std(best_vals, ddof=1) / np.sqrt(len(best_vals)) if len(best_vals) > 1 else 0
            se_diff = np.sqrt(se_bl**2 + se_zoo**2)

            improvement = (best_zoo_mean - bl_mean) * 100
            configs.append(dict(
                cn=cn, bl=bl_mean, improvement=improvement,
                se=se_diff * 100, best_A=best_A,
            ))

    configs.sort(key=lambda x: x["improvement"], reverse=True)

    fig, ax = plt.subplots(figsize=(16, 6.5))
    x = np.arange(len(configs))
    colors = []
    for c in configs:
        bl = c["bl"]
        if bl < 0.7:
            colors.append("#e53935")  # hard
        elif bl < 0.9:
            colors.append("#ffa726")  # medium
        else:
            colors.append("#66bb6a")  # easy

    ax.bar(x, [c["improvement"] for c in configs],
           yerr=[c["se"] for c in configs],
           capsize=3, color=colors, edgecolor="white", linewidth=0.5)

    for i, c in enumerate(configs):
        a_label = f"A={int(c['best_A']*100)}%"
        y = c["improvement"]
        ax.text(i, y + c["se"] + 0.8, a_label, ha="center", va="bottom",
               fontsize=7, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels([c["cn"] for c in configs], rotation=45, ha="right", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Win Rate Improvement (pp)", fontsize=12)
    ax.set_title("Zoo Improvement per Config (Best Zoo A vs. A=5% Baseline) — Hider-Shaped",
                 fontsize=13)
    ax.grid(axis="y", alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e53935", label="Hard (baseline WR < 70%)"),
        Patch(facecolor="#ffa726", label="Medium (70–90%)"),
        Patch(facecolor="#66bb6a", label="Easy (> 90%)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    fig.tight_layout()
    out = BASE_DIR / "zoo_improvement_by_difficulty.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Plot 4: Forgetting Regret vs A (4×5 grid) ─────────────────────
def plot_fr_vs_A(fr_data):
    if not fr_data:
        print("No FR data, skipping fr_vs_A plot")
        return

    fig, axes = plt.subplots(len(STPS), len(HSMS), figsize=(19, 13),
                              sharex=True, sharey=True)
    a_pct = [int(a * 100) for a in A_VALUES]

    for i, stp in enumerate(STPS):
        for j, hsm in enumerate(HSMS):
            ax = axes[i, j]
            cn = config_name(stp, hsm)

            for sampling, color, label in [
                ("uniform", "#00bcd4", "uniform"),
                ("thompson_loss", "#e91e63", "thompson_loss"),
            ]:
                means, ses = [], []
                for A in A_VALUES:
                    key = (cn, stp, hsm, A, sampling)
                    vals = fr_data.get(key, [])
                    if vals:
                        means.append(np.mean(vals))
                        ses.append(np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0)
                    else:
                        means.append(np.nan)
                        ses.append(0)
                ax.errorbar(a_pct, means, yerr=ses, marker="o", markersize=4,
                           capsize=3, label=label, color=color, linewidth=1.5)

            ax.set_ylim(-0.02, 0.8)
            ax.set_xticks(a_pct)
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.set_title(f"HSM={hsm:.2f}", fontsize=11)
            if j == 0:
                ax.set_ylabel(f"STP={stp}\nForgetting Regret", fontsize=10)
            if i == len(STPS) - 1:
                ax.set_xlabel("A (%)", fontsize=10)
            if i == 0 and j == len(HSMS) - 1:
                ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("Forgetting Regret vs. Zoo Mixing Rate (A) — Hider-Shaped",
                 fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = GAUNTLET_DIR / "fr_vs_A.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Plot 5: FR heatmap examples ────────────────────────────────────
def plot_fr_heatmap_examples():
    """Show win-rate matrix heatmaps for high-FR and low-FR runs."""
    # Find high-FR and low-FR examples
    high_fr = None
    low_fr = None
    for jf in sorted(GAUNTLET_DIR.rglob("gauntlet_result.json")):
        with open(jf) as f:
            r = json.load(f)
        if r["n_checkpoints"] < 12:
            continue
        wm_mean = np.array(r["win_matrix"]).mean()
        # High FR: want visible forgetting with some wins
        if r["fr_full"] > 0.3 and wm_mean > 0.3:
            if high_fr is None or r["fr_full"] > high_fr["fr_full"]:
                high_fr = r
        # Low FR: want a run that mostly wins (high WR, low forgetting)
        if r["fr_full"] < 0.05 and wm_mean > 0.7:
            if low_fr is None or wm_mean > np.array(low_fr["win_matrix"]).mean():
                low_fr = r

    if not high_fr or not low_fr:
        print("Could not find suitable FR examples, skipping heatmap")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, result, title_suffix in [
        (axes[0], high_fr, "High Forgetting"),
        (axes[1], low_fr, "Low Forgetting"),
    ]:
        wm = np.array(result["win_matrix"])
        n = len(wm)
        updates = result["updates"]

        im = ax.imshow(wm, cmap="RdYlGn", vmin=0, vmax=1,
                       aspect="equal", origin="upper")

        # Label every other tick to avoid crowding
        tick_step = max(1, n // 7)
        tick_idx = list(range(0, n, tick_step))
        if n - 1 not in tick_idx:
            tick_idx.append(n - 1)
        ax.set_xticks(tick_idx)
        ax.set_xticklabels([str(updates[i]) for i in tick_idx], fontsize=7, rotation=45)
        ax.set_yticks(tick_idx)
        ax.set_yticklabels([str(updates[i]) for i in tick_idx], fontsize=7)
        ax.set_xlabel("Hider checkpoint (update)", fontsize=10)
        ax.set_ylabel("Seeker checkpoint (update)", fontsize=10)

        cn = result["config_name"]
        a_pct = int(result["A"] * 100)
        sampling = result["sampling"]
        fr = result["fr_full"]
        ax.set_title(f"{title_suffix}: {cn}, A={a_pct}% {sampling}\nFR = {fr:.3f}",
                     fontsize=11)

    fig.colorbar(im, ax=axes, label="Seeker Win Rate", shrink=0.8)
    fig.tight_layout()
    out = BASE_DIR / "fr_heatmap_examples.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Plot 6: A* vs Forgetting Regret (scatter) ─────────────────────
def plot_astar_vs_fr(wr_data, fr_data):
    """Scatter plot of optimal A (A*) vs its forgetting regret for each config."""
    baseline_A = 0.05
    zoo_As = [a for a in A_VALUES if a > baseline_A]

    points = []
    for stp in STPS:
        for hsm in HSMS:
            cn = config_name(stp, hsm)
            # Find A* (best A and sampling mode by win rate)
            best_wr = -1
            best_A = baseline_A
            best_s = SAMPLING_MODES[0]
            for A in A_VALUES:
                for s in SAMPLING_MODES:
                    key = (cn, stp, hsm, A, s)
                    vals = wr_data.get(key, [])
                    if vals and np.mean(vals) > best_wr:
                        best_wr = np.mean(vals)
                        best_A = A
                        best_s = s

            # Get FR for A*
            fr_key = (cn, stp, hsm, best_A, best_s)
            fr_vals = fr_data.get(fr_key, [])
            if not fr_vals:
                continue
            fr_mean = np.mean(fr_vals)
            fr_se = np.std(fr_vals, ddof=1) / np.sqrt(len(fr_vals)) if len(fr_vals) > 1 else 0

            # Baseline WR for coloring
            bl_vals = []
            for s in SAMPLING_MODES:
                bl_vals.extend(wr_data.get((cn, stp, hsm, baseline_A, s), []))
            bl_wr = np.mean(bl_vals) if bl_vals else 0

            points.append(dict(
                cn=cn, A_star=best_A, fr=fr_mean, fr_se=fr_se,
                bl_wr=bl_wr, best_wr=best_wr,
            ))

    fig, ax = plt.subplots(figsize=(8, 6))

    for p in points:
        if p["bl_wr"] < 0.7:
            color = "#e53935"
        elif p["bl_wr"] < 0.9:
            color = "#ffa726"
        else:
            color = "#66bb6a"
        ax.errorbar(p["A_star"] * 100, p["fr"], yerr=p["fr_se"],
                    marker="o", markersize=8, color=color, capsize=4,
                    markeredgecolor="white", markeredgewidth=0.5)
        ax.annotate(p["cn"], (p["A_star"] * 100, p["fr"]),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=6, alpha=0.7)

    ax.set_xlabel("Optimal Zoo Fraction A* (%)", fontsize=12)
    ax.set_ylabel("Forgetting Regret at A*", fontsize=12)
    ax.set_title("Optimal A* vs. Forgetting Regret — Hider-Shaped", fontsize=13)
    ax.grid(True, alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e53935", label="Hard (baseline WR < 70%)"),
        Patch(facecolor="#ffa726", label="Medium (70–90%)"),
        Patch(facecolor="#66bb6a", label="Easy (> 90%)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    fig.tight_layout()
    out = BASE_DIR / "astar_vs_fr.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    print("Loading gauntlet data...")
    wr_data, fr_data = load_gauntlet_data()
    print(f"  {len(wr_data)} (config, A, sampling) combos loaded")
    n_seeds = [len(v) for v in wr_data.values()]
    print(f"  seeds per combo: min={min(n_seeds)}, max={max(n_seeds)}")

    print("\nGenerating plots...")
    plot_wr_vs_A(wr_data)
    plot_improvement_summary(wr_data)
    plot_improvement_by_difficulty(wr_data)
    plot_fr_vs_A(fr_data)
    plot_fr_heatmap_examples()
    plot_astar_vs_fr(wr_data, fr_data)
    print("\nDone!")


if __name__ == "__main__":
    main()
