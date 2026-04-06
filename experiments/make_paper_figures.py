#!/usr/bin/env python3
"""
Generate all publication-quality figures for the AAAI paper.

Figures:
  1. Cross-method gauntlet heatmap (SAC dominance)
  2. Entropy schedule (universal alpha trajectory + cross-domain)
  3. Counterfactual (auto vs fixed vs none, tag + Ant Sumo)
  4. Mechanism (transplant + freeze + layerwise)
  5. Init-alpha inverted-U (multi-layout + Ant Sumo)
  6. Geometry & corner camping (9 layouts)
"""
from __future__ import annotations

import json
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Paper style
plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
})

OUT = Path("paper/figures")
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    'sac': '#2196F3',
    'ppo': '#F44336',
    'auto': '#2196F3',
    'fixed': '#FF9800',
    'none': '#9E9E9E',
    '0.05': '#F44336',
    '0.2': '#FF9800',
    '0.607': '#2196F3',
    '2.0': '#9C27B0',
}


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


# ============================================================
# Figure 2: Entropy schedule
# ============================================================
def fig2_entropy_schedule():
    """Alpha trajectory across presets + cross-domain comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))

    # Panel A: All 8 presets (from paper_ablations 1B)
    ax = axes[0]
    presets = [
        ("R0_baseline", "#9E9E9E"), ("R1_seeker_pursuit", "#2196F3"),
        ("R2_hider_active", "#4CAF50"), ("R3_both_shaped", "#9C27B0"),
        ("R4_sparse", "#F44336"), ("R5_escalating", "#FF9800"),
        ("R6_coverage", "#00BCD4"), ("R7_kitchen_sink", "#795548"),
    ]
    base = Path("experiments/results/paper_ablations/1B_alpha_dynamics")
    for preset_name, color in presets:
        csvs = list((base / preset_name / "auto").rglob("metrics.csv"))
        if not csvs:
            continue
        for csv_path in csvs[:1]:  # Just seed 0
            data = load_metrics(csv_path)
            if data is None:
                continue
            x = data['timesteps'] / 1e6
            ax.plot(x, data['hider_alpha'], color=color, linewidth=0.8, alpha=0.7,
                    label=preset_name.replace('_', ' '))
    ax.set_xlabel("Timesteps (M)")
    ax.set_ylabel(r"$\alpha$ (entropy temperature)")
    ax.set_title("(a) Tag: identical across 8 reward presets")
    ax.set_xlim(0, 5)
    ax.set_ylim(-0.01, 0.65)
    ax.legend(fontsize=5, ncol=2, loc='upper right')

    # Panel B: Tag vs Ant Sumo
    ax = axes[1]
    # Tag control
    tag_csvs = list(Path("experiments/results/paper_ablations/1A_counterfactual/R4_sparse/control").rglob("metrics.csv"))
    if tag_csvs:
        data = load_metrics(tag_csvs[0])
        if data is not None:
            x = data['timesteps'] / 1e6
            ax.plot(x, data['hider_alpha'], color='#2196F3', linewidth=1.2, label='Tag (2D)')

    # Ant Sumo
    sumo_csvs = list(Path("experiments/results/ant_sumo/baseline_02/seed_0").rglob("metrics.csv"))
    if sumo_csvs:
        data = load_metrics(sumo_csvs[0])
        if data is not None:
            x = data['timesteps'] / 1e6
            # Ant sumo has different column names
            alpha_col = 'hider_alpha' if 'hider_alpha' in data else 'seeker_alpha'
            if alpha_col in data:
                ax.plot(x, data[alpha_col], color='#F44336', linewidth=1.2, label='Ant Sumo (3D MuJoCo)')

    ax.set_xlabel("Timesteps (M)")
    ax.set_title("(b) Same schedule across domains")
    ax.set_xlim(0, 5)
    ax.set_ylim(-0.01, 0.65)
    ax.legend(loc='upper right')

    fig.tight_layout()
    fig.savefig(OUT / "fig2_entropy_schedule.pdf")
    fig.savefig(OUT / "fig2_entropy_schedule.png")
    print("Saved fig2_entropy_schedule")


# ============================================================
# Figure 3: Counterfactual
# ============================================================
def fig3_counterfactual():
    """Auto vs fixed vs none for tag and Ant Sumo."""
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.5))

    # Tag results (from paper_ablations gauntlet)
    tag_data = {
        'Auto-tuned': 57.1,
        r'$\alpha=0$': 46.0,
        r'Fixed $\alpha=0.1$': 9.9,
    }

    ax = axes[0]
    colors = ['#2196F3', '#9E9E9E', '#FF9800']
    bars = ax.bar(range(3), list(tag_data.values()), color=colors)
    ax.set_xticks(range(3))
    ax.set_xticklabels(list(tag_data.keys()), fontsize=8)
    ax.set_ylabel("Combined strength (%)")
    ax.set_title("(a) Tag")
    ax.set_ylim(0, 70)
    for bar, val in zip(bars, tag_data.values()):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1, f'{val:.0f}%',
                ha='center', fontsize=8)

    # Ant Sumo results
    sumo_gauntlet = Path("experiments/results/ant_sumo/gauntlet/gauntlet_results.json")
    if sumo_gauntlet.exists():
        with open(sumo_gauntlet) as f:
            sg = json.load(f)
        conds = sg['conditions']
        W = np.array(sg['win_matrix'])
        sumo_data = {}
        for j, c in enumerate(conds):
            sk = W[j, :].mean()
            surv = 1.0 - W[:, j].mean()
            sumo_data[c] = (sk + surv) / 2 * 100

        ax = axes[1]
        # Pick the 3 comparable conditions
        labels = ['baseline_02', 'no_entropy', 'fixed_01']
        display = ['Auto-tuned\n(init=0.2)', r'$\alpha=0$', r'Fixed $\alpha=0.1$']
        vals = [sumo_data.get(l, 0) for l in labels]
        bars = ax.bar(range(3), vals, color=colors)
        ax.set_xticks(range(3))
        ax.set_xticklabels(display, fontsize=8)
        ax.set_title("(b) Ant Sumo")
        ax.set_ylim(0, 70)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, val + 1, f'{val:.0f}%',
                    ha='center', fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "fig3_counterfactual.pdf")
    fig.savefig(OUT / "fig3_counterfactual.png")
    print("Saved fig3_counterfactual")


# ============================================================
# Figure 4: Mechanism — transplant + freeze + layerwise
# ============================================================
def fig4_mechanism():
    """Actor is the carrier of strength."""
    gauntlet_path = Path("experiments/results/paper_final/mechanism_deep_dive/gauntlet_results.json")
    if not gauntlet_path.exists():
        print("SKIP fig4: no gauntlet data")
        return

    with open(gauntlet_path) as f:
        data = json.load(f)

    conds = data['conditions']
    W = np.array(data['win_matrix'])
    n = len(conds)

    scores = {}
    for j, c in enumerate(conds):
        sk = W[j, :].mean()
        surv = 1.0 - W[:, j].mean()
        scores[c] = (sk + surv) / 2 * 100

    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.5))

    # Panel A: Transplant (control vs transplant)
    ax = axes[0]
    transplant_conds = [
        ('1_transplant_5seed/control_607', 'ctrl\n0.607', '#2196F3'),
        ('1_transplant_5seed/control_005', 'ctrl\n0.05', '#F44336'),
        ('1_transplant_5seed/transplant_005critic', '0.607 actor\n+0.05 critic', '#4CAF50'),
        ('1_transplant_5seed/transplant_607critic', '0.05 actor\n+0.607 critic', '#FF9800'),
    ]
    vals = [scores.get(c, 0) for c, _, _ in transplant_conds]
    colors = [col for _, _, col in transplant_conds]
    labels = [l for _, l, _ in transplant_conds]
    bars = ax.bar(range(len(vals)), vals, color=colors)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel("Combined strength (%)")
    ax.set_title("(a) Transplant")
    ax.set_ylim(0, 70)

    # Panel B: Freeze vs continued
    ax = axes[1]
    freeze_conds = [
        ('5_freeze_actor/freeze_607', 'Freeze\nactor', '#4CAF50'),
        ('3_random_critic/random_critic', 'Random\ncritic', '#FF9800'),
        ('1_transplant_5seed/control_607', 'Continued\ntraining', '#9E9E9E'),
    ]
    vals = [scores.get(c, 0) for c, _, _ in freeze_conds]
    colors = [col for _, _, col in freeze_conds]
    labels = [l for _, l, _ in freeze_conds]
    bars = ax.bar(range(len(vals)), vals, color=colors)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_title("(b) Freeze vs continue")
    ax.set_ylim(0, 70)

    # Panel C: Layerwise
    ax = axes[2]
    layer_conds = [
        ('4_layerwise/layer0_only', 'Layer 0\nonly', '#2196F3'),
        ('4_layerwise/heads_only', 'Heads\nonly', '#FF9800'),
    ]
    vals = [scores.get(c, 0) for c, _, _ in layer_conds]
    colors = [col for _, _, col in layer_conds]
    labels = [l for _, l, _ in layer_conds]
    bars = ax.bar(range(len(vals)), vals, color=colors)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_title("(c) Layer-wise")
    ax.set_ylim(0, 70)

    fig.tight_layout()
    fig.savefig(OUT / "fig4_mechanism.pdf")
    fig.savefig(OUT / "fig4_mechanism.png")
    print("Saved fig4_mechanism")


# ============================================================
# Figure 5: Init-alpha
# ============================================================
def fig5_init_alpha():
    """Inverted-U for tag + Ant Sumo comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))

    # Tag (from paper_final gauntlet)
    tag_gauntlet = Path("experiments/results/paper_final/gauntlet/paper_final_gauntlet.json")
    if tag_gauntlet.exists():
        with open(tag_gauntlet) as f:
            data = json.load(f)
        ia_data = data.get('init_alpha', {})
        if ia_data:
            configs = ia_data['configs']
            W = np.array(ia_data['win_matrix'])
            ax = axes[0]
            alphas = []
            scores = []
            for j, c in enumerate(configs):
                ia = float(c.replace('initalpha_', ''))
                sk = W[j, :].mean()
                surv = 1.0 - W[:, j].mean()
                alphas.append(ia)
                scores.append((sk + surv) / 2 * 100)
            ax.plot(alphas, scores, 'o-', color='#2196F3', markersize=8, linewidth=2)
            ax.set_xlabel("Initial alpha")
            ax.set_ylabel("Combined strength (%)")
            ax.set_title("(a) Tag (four_corners)")
            ax.set_xscale('log')
            for a, s in zip(alphas, scores):
                ax.annotate(f'{s:.0f}%', (a, s), textcoords="offset points",
                           xytext=(0, 8), ha='center', fontsize=7)

    # Ant Sumo
    sumo_gauntlet = Path("experiments/results/ant_sumo/gauntlet/gauntlet_results.json")
    if sumo_gauntlet.exists():
        with open(sumo_gauntlet) as f:
            sg = json.load(f)
        conds = sg['conditions']
        W = np.array(sg['win_matrix'])
        ax = axes[1]
        ia_map = {'baseline_02': 0.2, 'optimal_0607': 0.607,
                  'low_005': 0.05, 'high_20': 2.0}
        alphas, scores = [], []
        for j, c in enumerate(conds):
            if c not in ia_map:
                continue
            sk = W[j, :].mean()
            surv = 1.0 - W[:, j].mean()
            alphas.append(ia_map[c])
            scores.append((sk + surv) / 2 * 100)
        order = np.argsort(alphas)
        alphas = [alphas[i] for i in order]
        scores = [scores[i] for i in order]
        ax.plot(alphas, scores, 's-', color='#F44336', markersize=8, linewidth=2)
        ax.set_xlabel("Initial alpha")
        ax.set_title("(b) Ant Sumo")
        ax.set_xscale('log')
        for a, s in zip(alphas, scores):
            ax.annotate(f'{s:.0f}%', (a, s), textcoords="offset points",
                       xytext=(0, 8), ha='center', fontsize=7)

    fig.tight_layout()
    fig.savefig(OUT / "fig5_init_alpha.pdf")
    fig.savefig(OUT / "fig5_init_alpha.png")
    print("Saved fig5_init_alpha")


# ============================================================
# Figure 6: Geometry & corner camping
# ============================================================
def fig6_geometry():
    """Corner camping by layout."""
    gauntlet = Path("experiments/results/paper_final/gauntlet/paper_final_gauntlet.json")
    if not gauntlet.exists():
        print("SKIP fig6: no gauntlet data")
        return

    with open(gauntlet) as f:
        data = json.load(f)

    geo = data.get('geometry', {})
    if not geo:
        print("SKIP fig6: no geometry data")
        return

    layouts = list(geo.keys())
    corners = [geo[l]['mean_hider_corner_frac'] for l in layouts]
    swr = [geo[l]['seeker_win_rate'] for l in layouts]

    # Sort by corner camping
    order = np.argsort(corners)
    layouts = [layouts[i] for i in order]
    corners = [corners[i] for i in order]
    swr = [swr[i] for i in order]

    fig, ax1 = plt.subplots(figsize=(6.5, 3))

    x = np.arange(len(layouts))
    width = 0.35

    bars1 = ax1.bar(x - width/2, corners, width, color='#F44336', alpha=0.8, label='Corner camping')
    ax1.set_ylabel('Corner camping fraction', color='#F44336')
    ax1.tick_params(axis='y', labelcolor='#F44336')

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, swr, width, color='#2196F3', alpha=0.8, label='Seeker win rate')
    ax2.set_ylabel('Seeker win rate', color='#2196F3')
    ax2.tick_params(axis='y', labelcolor='#2196F3')

    ax1.set_xticks(x)
    ax1.set_xticklabels([l.replace('_', '\n') for l in layouts], fontsize=7, rotation=30, ha='right')
    ax1.set_title('Corner camping and seeker effectiveness by arena layout')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=7)

    fig.tight_layout()
    fig.savefig(OUT / "fig6_geometry.pdf")
    fig.savefig(OUT / "fig6_geometry.png")
    print("Saved fig6_geometry")


def main():
    print("Generating paper figures...")
    print(f"Output: {OUT}\n")

    fig2_entropy_schedule()
    fig3_counterfactual()
    fig4_mechanism()
    fig5_init_alpha()
    fig6_geometry()

    print(f"\nAll figures saved to {OUT}")


if __name__ == "__main__":
    main()
