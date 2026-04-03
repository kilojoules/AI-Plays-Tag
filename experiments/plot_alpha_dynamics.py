#!/usr/bin/env python3
"""
Plot alpha (entropy temperature) dynamics for AAMAS paper.

Figure 1: 1A counterfactual — alpha trajectory for control vs fixed conditions
Figure 2: 1B all presets — alpha trajectory colored by preset
Figure 3: Alpha trajectory vs SWR (seeker win rate) for control
"""
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})

BASE = Path("experiments/results/paper_ablations")
OUT = BASE / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def load_metrics(csv_path):
    """Load metrics CSV into dict of numpy arrays."""
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return None
    data = {}
    for key in rows[0]:
        try:
            data[key] = np.array([float(r[key]) for r in rows])
        except (ValueError, KeyError):
            pass
    return data


def find_run_csvs(base_dir, pattern="**/metrics.csv"):
    return sorted(base_dir.rglob("metrics.csv"))


def load_condition(base_dir):
    """Load all seeds for a condition, return list of metrics dicts."""
    csvs = find_run_csvs(base_dir)
    return [load_metrics(p) for p in csvs if load_metrics(p) is not None]


def mean_and_envelope(runs, x_key, y_key):
    """Compute mean and min/max envelope across seeds, interpolated to common x."""
    if not runs:
        return None, None, None, None
    # Use shortest run's x range
    min_len = min(len(r[x_key]) for r in runs)
    x = runs[0][x_key][:min_len]
    ys = np.array([r[y_key][:min_len] for r in runs])
    return x, ys.mean(axis=0), ys.min(axis=0), ys.max(axis=0)


# =============================================================
# Figure 1: Counterfactual alpha trajectories
# =============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

conditions = [
    ("control", "Auto-tuned (control)", "#2196F3"),
    ("alpha0", r"Fixed $\alpha=0$", "#F44336"),
    ("alpha01", r"Fixed $\alpha=0.1$", "#FF9800"),
]

for role_idx, (role, role_label) in enumerate([
    ("seeker_alpha", "Seeker"),
    ("hider_alpha", "Hider"),
]):
    ax = axes[role_idx]
    for cond_name, cond_label, color in conditions:
        runs = load_condition(BASE / "1A_counterfactual" / "R4_sparse" / cond_name)
        if not runs:
            continue
        x, mean, lo, hi = mean_and_envelope(runs, "timesteps", role)
        if x is None:
            continue
        x_m = x / 1e6
        ax.plot(x_m, mean, color=color, label=cond_label, linewidth=1.5)
        ax.fill_between(x_m, lo, hi, color=color, alpha=0.15)

    ax.set_xlabel("Timesteps (M)")
    ax.set_ylabel(r"$\alpha$ (entropy temperature)" if role_idx == 0 else "")
    ax.set_title(f"{role_label} entropy temperature")
    ax.legend(loc="upper right")
    ax.set_xlim(0, 5)
    ax.set_ylim(-0.01, 0.7)
    ax.axhline(y=0, color='gray', linestyle=':', linewidth=0.5)

fig.suptitle("Entropy temperature dynamics: R4_sparse (no reward shaping)", y=1.02)
fig.tight_layout()
fig.savefig(OUT / "alpha_counterfactual.png", bbox_inches="tight")
print(f"Saved {OUT / 'alpha_counterfactual.png'}")


# =============================================================
# Figure 2: Alpha dynamics across all presets (hider only)
# =============================================================
presets = [
    ("R0_baseline", "R0 Baseline", "#9E9E9E"),
    ("R1_seeker_pursuit", "R1 Seeker Pursuit", "#2196F3"),
    ("R2_hider_active", "R2 Hider Active", "#4CAF50"),
    ("R3_both_shaped", "R3 Both Shaped", "#9C27B0"),
    ("R4_sparse", "R4 Sparse", "#F44336"),
    ("R5_escalating", "R5 Escalating", "#FF9800"),
    ("R6_coverage", "R6 Coverage", "#00BCD4"),
    ("R7_kitchen_sink", "R7 Kitchen Sink", "#795548"),
]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

for role_idx, (role, role_label) in enumerate([
    ("seeker_alpha", "Seeker"),
    ("hider_alpha", "Hider"),
]):
    ax = axes[role_idx]
    for preset_name, preset_label, color in presets:
        runs = load_condition(BASE / "1B_alpha_dynamics" / preset_name / "auto")
        if not runs:
            continue
        x, mean, lo, hi = mean_and_envelope(runs, "timesteps", role)
        if x is None:
            continue
        x_m = x / 1e6
        ax.plot(x_m, mean, color=color, label=preset_label, linewidth=1.2)
        ax.fill_between(x_m, lo, hi, color=color, alpha=0.08)

    ax.set_xlabel("Timesteps (M)")
    ax.set_ylabel(r"$\alpha$ (entropy temperature)" if role_idx == 0 else "")
    ax.set_title(f"{role_label} entropy temperature by preset")
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    ax.set_xlim(0, 5)

fig.suptitle("How reward shaping affects learned entropy temperature", y=1.02)
fig.tight_layout()
fig.savefig(OUT / "alpha_by_preset.png", bbox_inches="tight")
print(f"Saved {OUT / 'alpha_by_preset.png'}")


# =============================================================
# Figure 3: Alpha + SWR co-plot for control (the mechanistic story)
# =============================================================
fig, ax1 = plt.subplots(figsize=(8, 4.5))

runs = load_condition(BASE / "1A_counterfactual" / "R4_sparse" / "control")
if runs:
    x, alpha_mean, alpha_lo, alpha_hi = mean_and_envelope(
        runs, "timesteps", "hider_alpha")
    _, swr_mean, swr_lo, swr_hi = mean_and_envelope(
        runs, "timesteps", "seeker_win_rate")

    x_m = x / 1e6

    color1 = "#2196F3"
    ax1.plot(x_m, alpha_mean, color=color1, linewidth=1.5, label=r"Hider $\alpha$")
    ax1.fill_between(x_m, alpha_lo, alpha_hi, color=color1, alpha=0.15)
    ax1.set_xlabel("Timesteps (M)")
    ax1.set_ylabel(r"$\alpha$ (entropy temperature)", color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xlim(0, 5)

    ax2 = ax1.twinx()
    color2 = "#F44336"
    ax2.plot(x_m, swr_mean, color=color2, linewidth=1.5, label="Seeker win rate")
    ax2.fill_between(x_m, swr_lo, swr_hi, color=color2, alpha=0.15)
    ax2.set_ylabel("Seeker win rate", color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(-0.05, 1.05)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

fig.suptitle(r"Entropy temperature decay tracks competitive dynamics (R4\_sparse, auto-tuned)")
fig.tight_layout()
fig.savefig(OUT / "alpha_vs_swr.png", bbox_inches="tight")
print(f"Saved {OUT / 'alpha_vs_swr.png'}")


# =============================================================
# Figure 4: Final alpha values by preset (bar chart)
# =============================================================
fig, ax = plt.subplots(figsize=(10, 4))

final_alphas = []
labels = []
colors_list = []
for preset_name, preset_label, color in presets:
    runs = load_condition(BASE / "1B_alpha_dynamics" / preset_name / "auto")
    if not runs:
        continue
    # Average final alpha (last 10 entries) across seeds, both roles
    vals = []
    for r in runs:
        vals.append(np.mean(r["seeker_alpha"][-10:]))
        vals.append(np.mean(r["hider_alpha"][-10:]))
    final_alphas.append((np.mean(vals), np.std(vals)))
    labels.append(preset_label)
    colors_list.append(color)

x_pos = np.arange(len(labels))
means = [v[0] for v in final_alphas]
stds = [v[1] for v in final_alphas]

bars = ax.bar(x_pos, means, yerr=stds, color=colors_list, capsize=4, alpha=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
ax.set_ylabel(r"Final $\alpha$ (mean of last 10 log entries)")
ax.set_title("Converged entropy temperature by reward preset")
fig.tight_layout()
fig.savefig(OUT / "final_alpha_by_preset.png", bbox_inches="tight")
print(f"Saved {OUT / 'final_alpha_by_preset.png'}")

print("\nDone!")
