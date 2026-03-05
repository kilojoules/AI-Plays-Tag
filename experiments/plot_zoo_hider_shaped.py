#!/usr/bin/env python3
"""
Generate all 4 README plots for the zoo_hider_shaped sweep.

Outputs:
  experiments/results/zoo_hider_shaped/seeker_wr_vs_A.png
  experiments/results/zoo_hider_shaped/zoo_improvement_summary.png
  experiments/results/zoo_hider_shaped/zoo_improvement_by_difficulty.png
  experiments/results/zoo_hider_shaped/gauntlet/fr_vs_A.png
"""
import csv
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
SEEDS = [0, 1, 2]


def stp_str(stp):
    return f"{stp:.4f}".replace("0.", "").rstrip("0") or "0"


def config_name(stp, hsm):
    return f"STP{stp_str(stp)}_HSM{round(hsm * 100)}"


def load_final_win_rates():
    """Load final seeker win rate for each (config, A, sampling, seed)."""
    results = {}
    for stp, hsm, A, sampling, seed in itertools.product(
        STPS, HSMS, A_VALUES, SAMPLING_MODES, SEEDS
    ):
        cn = config_name(stp, hsm)
        a_str = f"A{int(A * 100):02d}"
        run_dir = BASE_DIR / cn / f"{a_str}_{sampling}" / f"seed_{seed}"
        if not run_dir.exists():
            continue
        # Find the latest timestamped subdir
        subdirs = sorted(run_dir.glob("2*"), key=lambda p: p.name)
        if not subdirs:
            continue
        metrics_path = subdirs[-1] / "metrics.csv"
        if not metrics_path.exists():
            continue
        with open(metrics_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            continue
        final_wr = float(rows[-1]["seeker_win_rate"])
        key = (cn, stp, hsm, A, sampling)
        results.setdefault(key, []).append(final_wr)
    return results


def load_fr_summary():
    """Load forgetting regret summary CSV."""
    results = {}
    csv_path = GAUNTLET_DIR / "fr_summary.csv"
    if not csv_path.exists():
        print(f"WARNING: {csv_path} not found, skipping FR plot")
        return results
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["config_name"], float(row["stp"]), float(row["hsm"]),
                   float(row["A"]), row["sampling"])
            results[key] = float(row["fr_full"])
    return results


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
                        ses.append(np.std(vals) / np.sqrt(len(vals)))
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
    # For each config: best zoo A (A > 5%) vs baseline (A = 5%)
    baseline_A = 0.05
    zoo_As = [a for a in A_VALUES if a > baseline_A]

    improvements = np.full((len(STPS), len(HSMS)), np.nan)
    for i, stp in enumerate(STPS):
        for j, hsm in enumerate(HSMS):
            cn = config_name(stp, hsm)
            # Baseline: best of both sampling modes at A=5%
            bl_vals = []
            for s in SAMPLING_MODES:
                key = (cn, stp, hsm, baseline_A, s)
                bl_vals.extend(wr_data.get(key, []))
            if not bl_vals:
                continue
            bl_mean = np.mean(bl_vals)

            # Best zoo: best mean across A > 5% and both sampling modes
            best_zoo = bl_mean
            for A in zoo_As:
                for s in SAMPLING_MODES:
                    key = (cn, stp, hsm, A, s)
                    vals = wr_data.get(key, [])
                    if vals:
                        best_zoo = max(best_zoo, np.mean(vals))
            improvements[i, j] = (best_zoo - bl_mean) * 100  # percentage points

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

            # SE of the difference (pooled from both)
            best_vals = []
            for s in SAMPLING_MODES:
                key = (cn, stp, hsm, best_A, s)
                if best_A == baseline_A or s == best_sampling:
                    best_vals.extend(wr_data.get(key, []))
            se_bl = np.std(bl_vals) / np.sqrt(len(bl_vals)) if len(bl_vals) > 1 else 0
            se_zoo = np.std(best_vals) / np.sqrt(len(best_vals)) if len(best_vals) > 1 else 0
            se_diff = np.sqrt(se_bl**2 + se_zoo**2)

            improvement = (best_zoo_mean - bl_mean) * 100
            configs.append(dict(
                cn=cn, bl=bl_mean, improvement=improvement,
                se=se_diff * 100, best_A=best_A,
            ))

    # Sort by improvement magnitude
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

    bars = ax.bar(x, [c["improvement"] for c in configs],
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

    # Legend
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
                vals = []
                for A in A_VALUES:
                    key = (cn, stp, hsm, A, sampling)
                    vals.append(fr_data.get(key, np.nan))
                ax.plot(a_pct, vals, marker="o", markersize=4,
                       label=label, color=color, linewidth=1.5)

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


def main():
    print("Loading win rate data...")
    wr_data = load_final_win_rates()
    print(f"  {len(wr_data)} (config, A, sampling) combos loaded")

    print("Loading FR data...")
    fr_data = load_fr_summary()
    print(f"  {len(fr_data)} FR entries loaded")

    print("\nGenerating plots...")
    plot_wr_vs_A(wr_data)
    plot_improvement_summary(wr_data)
    plot_improvement_by_difficulty(wr_data)
    plot_fr_vs_A(fr_data)
    print("\nDone!")


if __name__ == "__main__":
    main()
