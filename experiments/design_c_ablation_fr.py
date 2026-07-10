#!/usr/bin/env python3
"""
Forgetting Regret on the R7 reward-ablation checkpoints.

Builds a checkpoint x checkpoint self-play gauntlet per run (12 saved
PPO checkpoints, every 50 updates), then computes FR (observed + permutation
null) per run and aggregates per group. Tests whether removing
--area-coverage-bonus (and additionally --seeker-escalating-urgency) reduces
*training-trajectory* regret, complementing the endpoint-WR matched eval.

Groups:
  R7_kitchen_sink  (baseline, A=0, seeds 0..8)
  R7_no_coverage   (coverage off, A=0, seeds 0..9)
  R7_no_cov_urg    (coverage + urgency off, A=0, seeds 0..9)

Outputs:
  experiments/results/design_c/coverage_ablation/ablation_fr.csv  (per-run)
  experiments/results/design_c/coverage_ablation/ablation_fr.png  (group plot)

Usage:
  python experiments/design_c_ablation_fr.py --episodes 30 --n-perm 500
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from trainer.tag_env import VecTagEnv, TagEnvConfig
from design_c_trajectory_analysis import load_policy, collect_features, HSM, LAYOUT

GRID_DIR = ROOT / "experiments" / "results" / "design_c" / "grid"
COV_DIR = ROOT / "experiments" / "results" / "design_c" / "coverage_ablation"
URG_DIR = ROOT / "experiments" / "results" / "design_c" / "urgency_ablation"
OUT_DIR = COV_DIR  # keep with the rest of the ablation outputs

GROUPS = [
    ("R7_kitchen_sink", GRID_DIR / "R7_kitchen_sink" / "A00", list(range(9))),
    ("R7_no_coverage", COV_DIR / "R7_no_coverage" / "A00", list(range(10))),
    ("R7_no_cov_urg", URG_DIR / "R7_no_cov_urg" / "A00", list(range(10))),
]

CKPT_RE = re.compile(r"seeker_(\d+)\.pt$")


# --- FR functions kept in sync with sophia-results/gauntlet/plot_forgetting_regret.py
def forgetting_regret(W: np.ndarray):
    rmax = np.maximum.accumulate(W, axis=0)
    pr = rmax - W
    return float(pr.mean()), float(pr[-1].mean())


def forgetting_regret_null(W: np.ndarray, n_perm: int, rng):
    n, m = W.shape
    fr_full = np.empty(n_perm)
    fr_final = np.empty(n_perm)
    for b in range(n_perm):
        shuf = np.empty_like(W)
        for j in range(m):
            shuf[:, j] = W[rng.permutation(n), j]
        rmax = np.maximum.accumulate(shuf, axis=0)
        pr = rmax - shuf
        fr_full[b] = pr.mean()
        fr_final[b] = pr[-1].mean()
    return fr_full, fr_final


def find_checkpoints(run_dir: Path):
    """Return sorted list of (update, seeker_path, hider_path) for one run."""
    for ts in sorted(run_dir.glob("2*"), reverse=True):
        ck = ts / "checkpoints"
        if not ck.exists():
            continue
        triples = []
        for sp in sorted(ck.glob("seeker_*.pt")):
            m = CKPT_RE.search(sp.name)
            if not m:
                continue
            update = int(m.group(1))
            hp = ck / f"hider_{update:05d}.pt"
            if hp.exists():
                triples.append((update, sp, hp))
        if triples:
            triples.sort(key=lambda t: t[0])
            return triples
    return []


def build_win_matrix(triples, env_cfg, episodes, obs_dim, act_dim):
    n = len(triples)
    seekers = [load_policy(sp, obs_dim, act_dim) for _, sp, _ in triples]
    hiders = [load_policy(hp, obs_dim, act_dim) for _, _, hp in triples]
    W = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            feats = collect_features(seekers[i], hiders[j], env_cfg, episodes)
            W[i, j] = feats["wr"]
    return W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30,
                    help="Episodes per (seeker, hider) checkpoint pair.")
    ap.add_argument("--n-perm", type=int, default=500,
                    help="Permutations for the FR null distribution.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    env_probe = VecTagEnv(num_envs=1, config=TagEnvConfig(layout=LAYOUT, hider_speed_mult=HSM))
    obs_dim, act_dim = env_probe.obs_dim, env_probe.act_dim
    env_cfg = TagEnvConfig(layout=LAYOUT, hider_speed_mult=HSM)

    rows = []
    for group_name, base, seeds in GROUPS:
        for seed in seeds:
            run_dir = base / f"seed_{seed}"
            triples = find_checkpoints(run_dir)
            if not triples:
                print(f"[skip] {group_name}/seed_{seed}: no checkpoints", file=sys.stderr)
                continue
            print(f"{group_name}/seed_{seed}: {len(triples)} checkpoints "
                  f"({triples[0][0]}..{triples[-1][0]})", flush=True)
            W = build_win_matrix(triples, env_cfg, args.episodes, obs_dim, act_dim)
            obs_full, obs_final = forgetting_regret(W)
            null_full, null_final = forgetting_regret_null(W, args.n_perm, rng)
            row = dict(
                group=group_name, seed=seed,
                n_checkpoints=len(triples),
                fr_full=obs_full, fr_final=obs_final,
                fr_full_null=float(null_full.mean()),
                fr_final_null=float(null_final.mean()),
                fr_full_excess=obs_full - float(null_full.mean()),
                fr_final_excess=obs_final - float(null_final.mean()),
                fr_full_p=float((null_full >= obs_full).mean()),
                fr_final_p=float((null_final >= obs_final).mean()),
            )
            rows.append(row)
            print(f"  fr_full={obs_full:.3f} (null={row['fr_full_null']:.3f}, "
                  f"excess={row['fr_full_excess']:+.3f}, p={row['fr_full_p']:.3f})  "
                  f"fr_final={obs_final:.3f} (null={row['fr_final_null']:.3f}, "
                  f"excess={row['fr_final_excess']:+.3f}, p={row['fr_final_p']:.3f})",
                  flush=True)

    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "ablation_fr.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")

    print("\n" + "=" * 70)
    print("Forgetting Regret by group (training-trajectory regression)")
    print("=" * 70)
    for g, _, _ in GROUPS:
        sub = df[df.group == g]
        if sub.empty:
            continue
        flagged = sub[sub.fr_full_p < 0.05]
        print(f"\n{g} (n={len(sub)})")
        print(f"  fr_full   mean={sub.fr_full.mean():.3f}  std={sub.fr_full.std(ddof=0):.3f}")
        print(f"  excess    mean={sub.fr_full_excess.mean():+.3f}  "
              f"std={sub.fr_full_excess.std(ddof=0):.3f}")
        print(f"  flagged (p<0.05): {len(flagged)}/{len(sub)}"
              + (f"  -> seeds {sorted(flagged.seed.tolist())}" if len(flagged) else ""))

    # --- Plot: per-seed FR_full + excess by group ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        groups_present = [g for g, _, _ in GROUPS if not df[df.group == g].empty]
        colors = {"R7_kitchen_sink": "tab:red", "R7_no_coverage": "tab:blue",
                  "R7_no_cov_urg": "tab:green"}
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, ycol, ylabel in zip(
            axes, ["fr_full", "fr_full_excess"],
            ["FR (observed)", "FR excess over null (obs - null mean)"],
        ):
            for k, g in enumerate(groups_present):
                sub = df[df.group == g]
                x = np.full(len(sub), k) + (rng.random(len(sub)) - 0.5) * 0.15
                ax.scatter(x, sub[ycol], color=colors.get(g, "gray"),
                           s=70, edgecolor="k", alpha=0.85, label=g)
                ax.hlines(sub[ycol].mean(), k - 0.3, k + 0.3,
                          colors=colors.get(g, "gray"), linewidth=3)
            ax.axhline(0, color="grey", lw=0.5)
            ax.set_xticks(range(len(groups_present)))
            ax.set_xticklabels(groups_present, rotation=10)
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel)
            ax.grid(alpha=0.3)
        fig.suptitle("Training-trajectory Forgetting Regret per ablation run "
                     f"({args.episodes} eps/matchup, {args.n_perm} null perms)",
                     fontweight="bold")
        plt.tight_layout()
        png_path = OUT_DIR / "ablation_fr.png"
        plt.savefig(png_path, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"Wrote {png_path}")
    except Exception as e:
        print(f"(plot skipped: {e})", file=sys.stderr)


if __name__ == "__main__":
    main()
