#!/usr/bin/env python3
"""
Seed-5 diagnostic — did the residual R7_no_coverage failure ever learn
pursuit, or did it always roam?

The endpoint-WR matched eval flagged R7_no_coverage/A=0/seed_5 as the only
remaining collapse after the coverage ablation. The FR analysis said its
training trajectory was monotone (no genuine forgetting). To distinguish
"never learned" from "briefly learned then drifted" we build seed 5's full
12 x 12 checkpoint-gauntlet win matrix and plot two skill curves:

  - self-play diagonal: W[i, i] over training time
  - vs-final-hider:     W[i, n-1] over training time (skill against the
                        strongest available hider)

A healthy run rises and plateaus. "Never learned" stays at the floor.
"Learned then drifted" peaks then falls.

Seed 0 (healthy R7_no_coverage, endpoint wr ~0.93) is plotted alongside
as a contrast.

Outputs:
  experiments/results/design_c/coverage_ablation/seed5_diag.png
  experiments/results/design_c/coverage_ablation/seed5_diag.csv

Usage:
  python experiments/design_c_seed5_diagnostic.py --episodes 30
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

COV_DIR = ROOT / "experiments" / "results" / "design_c" / "coverage_ablation"
OUT_DIR = COV_DIR

# (label, seed) — seed-0 = healthy reference, seed-5 = the residual failure.
RUNS = [
    ("R7_no_coverage seed 0 (healthy)", 0),
    ("R7_no_coverage seed 5 (collapsed)", 5),
]

CKPT_RE = re.compile(r"seeker_(\d+)\.pt$")


def find_checkpoints(run_dir: Path):
    for ts in sorted(run_dir.glob("2*"), reverse=True):
        ck = ts / "checkpoints"
        if not ck.exists():
            continue
        triples = []
        for sp in sorted(ck.glob("seeker_*.pt")):
            m = CKPT_RE.search(sp.name)
            if not m:
                continue
            u = int(m.group(1))
            hp = ck / f"hider_{u:05d}.pt"
            if hp.exists():
                triples.append((u, sp, hp))
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
            W[i, j] = collect_features(seekers[i], hiders[j], env_cfg, episodes)["wr"]
    return W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    env_probe = VecTagEnv(num_envs=1,
                          config=TagEnvConfig(layout=LAYOUT, hider_speed_mult=HSM))
    obs_dim, act_dim = env_probe.obs_dim, env_probe.act_dim
    env_cfg = TagEnvConfig(layout=LAYOUT, hider_speed_mult=HSM)

    runs = []
    for label, seed in RUNS:
        run_dir = COV_DIR / "R7_no_coverage" / "A00" / f"seed_{seed}"
        triples = find_checkpoints(run_dir)
        if not triples:
            print(f"[skip] {label}: no checkpoints", file=sys.stderr)
            continue
        print(f"{label}: {len(triples)} checkpoints "
              f"({triples[0][0]}..{triples[-1][0]})", flush=True)
        W = build_win_matrix(triples, env_cfg, args.episodes, obs_dim, act_dim)
        runs.append(dict(label=label, seed=seed,
                         updates=[u for u, _, _ in triples], W=W))
        print(f"  diag mean={float(np.diag(W).mean()):.3f}  "
              f"final-row mean={float(W[-1].mean()):.3f}  "
              f"final-col mean={float(W[:, -1].mean()):.3f}", flush=True)

    rows = []
    for run in runs:
        n = len(run["updates"])
        for i in range(n):
            rows.append(dict(seed=run["seed"], label=run["label"],
                             update=run["updates"][i],
                             diag_wr=float(run["W"][i, i]),
                             vs_final_hider_wr=float(run["W"][i, -1]),
                             final_seeker_vs_this_hider_wr=float(run["W"][-1, i])))
    pd.DataFrame(rows).to_csv(OUT_DIR / "seed5_diag.csv", index=False)
    print(f"\nWrote {OUT_DIR / 'seed5_diag.csv'}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_runs = len(runs)
    fig = plt.figure(figsize=(6.5 * n_runs, 8))
    gs = fig.add_gridspec(2, n_runs, height_ratios=[1.4, 1.0], hspace=0.32, wspace=0.25)

    for k, run in enumerate(runs):
        W = run["W"]
        updates = run["updates"]
        ax = fig.add_subplot(gs[0, k])
        im = ax.imshow(W, cmap="viridis", aspect="auto", origin="lower",
                       vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(updates)))
        ax.set_xticklabels(updates, rotation=45, fontsize=8)
        ax.set_yticks(range(len(updates)))
        ax.set_yticklabels(updates, fontsize=8)
        ax.set_xlabel("hider checkpoint (update)")
        ax.set_ylabel("seeker checkpoint (update)")
        ax.set_title(run["label"], fontweight="bold")
        plt.colorbar(im, ax=ax, label="win rate")
        for i in range(len(updates)):
            for j in range(len(updates)):
                ax.text(j, i, f"{W[i,j]:.2f}", ha="center", va="center",
                        fontsize=7,
                        color="white" if W[i, j] < 0.55 else "black")

        ax2 = fig.add_subplot(gs[1, k])
        ax2.plot(updates, np.diag(W), "o-", lw=2, label="self-play (diagonal)")
        ax2.plot(updates, W[:, -1], "s-", lw=2, label="vs final hider (col -1)")
        ax2.plot(updates, W[-1, :], "^-", lw=2, alpha=0.7,
                 label="final seeker vs each hider (row -1)")
        ax2.set_xlabel("checkpoint update")
        ax2.set_ylabel("win rate")
        ax2.set_ylim(-0.02, 1.05)
        ax2.set_title("Skill curves")
        ax2.grid(alpha=0.3)
        ax2.legend(loc="best", fontsize=9)

    fig.suptitle(
        "R7_no_coverage @ A=0: did seed-5 ever learn pursuit, or always roam?\n"
        f"(12 checkpoints x 12 hiders, {args.episodes} episodes/cell)",
        fontsize=13, fontweight="bold",
    )
    png = OUT_DIR / "seed5_diag.png"
    plt.savefig(png, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Wrote {png}")


if __name__ == "__main__":
    main()
