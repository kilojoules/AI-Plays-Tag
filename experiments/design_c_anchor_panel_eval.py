#!/usr/bin/env python3
"""
Anchor-panel eval — every Design C seeker vs ALL 4 prereg anchors + own hider.

Fixes the idea-critic's top methodological concern: all prior matched evals
used a single reference hider (the R4/A=0 anchor, id=28), while the prereg
(section 4.2) specifies a 4-anchor reference set. Every collapse verdict,
the beta_RA proxy, and the seed-5 over-specialization diagnosis inherited
that single-opponent fragility.

Per seeker this script evaluates 5 opponents x --episodes episodes:
  - the 4 prereg anchors (policy_index.csv, source == "anchor")
  - the seeker's OWN final hider ("own") — generalizing the seed-5
    diagnostic to every run: anchor-mean WR low + own WR high =>
    over-specialization; both low => basin / failure-to-learn.

Also adds the R4_sparse grid cells, which were missing from every prior
ablation table (the critic: "you have no R4_sparse cells — 'shaping causes
the basin' is not testable without them").

Output (per (seeker, opponent) row, episode counts kept for episode-level
GLMM fitting by design_c_glmm_fit.py):
  experiments/results/design_c/anchor_panel/anchor_panel.csv

Usage:
  python experiments/design_c_anchor_panel_eval.py --episodes 30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from trainer.tag_env import VecTagEnv, TagEnvConfig
from design_c_trajectory_analysis import (
    load_policy, collect_features, discover_run, GAUNTLET_DIR, HSM, LAYOUT,
)

GRID_DIR = ROOT / "experiments" / "results" / "design_c" / "grid"
COV_DIR = ROOT / "experiments" / "results" / "design_c" / "coverage_ablation"
URG_DIR = ROOT / "experiments" / "results" / "design_c" / "urgency_ablation"
URGONLY_DIR = ROOT / "experiments" / "results" / "design_c" / "urgency_only_ablation"
ASW_DIR = ROOT / "experiments" / "results" / "design_c" / "a_sweep"
MIXED_DIR = ROOT / "experiments" / "results" / "design_c" / "mixed_reward"
GAECHK_DIR = ROOT / "experiments" / "results" / "design_c" / "gae_check"
OUT_DIR = ROOT / "experiments" / "results" / "design_c" / "anchor_panel"

# (reward label, A, base dir, seeds). Groups whose runs are missing are
# skipped with a notice (e.g. R7_no_urgency until its training lands).
GROUPS = [
    # grid cells extended to n=20 by the prereg v3 power extension
    ("R4_sparse",       0.0,  GRID_DIR,    list(range(20))),
    ("R4_sparse",       0.5,  GRID_DIR,    list(range(20))),
    ("R7_kitchen_sink", 0.0,  GRID_DIR,    list(range(20))),
    ("R7_kitchen_sink", 0.5,  GRID_DIR,    list(range(20))),
    ("R7_kitchen_sink", 0.05, ASW_DIR,     list(range(10))),
    ("R7_kitchen_sink", 0.10, ASW_DIR,     list(range(10))),
    ("R7_kitchen_sink", 0.25, ASW_DIR,     list(range(10))),
    ("R7_no_coverage",  0.0,  COV_DIR,     list(range(10))),
    ("R7_no_coverage",  0.5,  COV_DIR,     list(range(5))),
    ("R7_no_cov_urg",   0.0,  URG_DIR,     list(range(10))),
    ("R7_no_urgency",   0.0,  URGONLY_DIR, list(range(10))),
    # 2026-07 review follow-ups (results.md §18.5, §18.6)
    ("R7sk_R4hd",       0.0,  MIXED_DIR,   list(range(10))),
    ("R7sk_R4hd",       0.5,  MIXED_DIR,   list(range(10))),
    ("R4_sparse_gaefix",       0.0, GAECHK_DIR, list(range(5))),
    ("R7_kitchen_sink_gaefix", 0.0, GAECHK_DIR, list(range(5))),
]


def _localize(ts_dir: str) -> Path:
    """policy_index.csv stores absolute paths from the machine that built it
    (e.g. LUMI scratch). Re-root them at this repo's experiments/results."""
    s = str(ts_dir)
    i = s.find("experiments/results")
    return ROOT / s[i:] if i >= 0 else Path(s)


def load_anchor_hiders(obs_dim, act_dim):
    idx = pd.read_csv(GAUNTLET_DIR / "policy_index.csv")
    anchors = []
    for _, r in idx[idx.source == "anchor"].iterrows():
        label = f"anchor_{'R4' if r.reward == 'R4_sparse' else 'R7'}_A{int(float(r.A) * 100):02d}"
        hider = load_policy(_localize(r.ts_dir) / "policy_hider_final.pt",
                            obs_dim, act_dim)
        anchors.append((label, hider))
    return anchors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30,
                    help="Episodes per (seeker, opponent) pair.")
    ap.add_argument("--only", type=str, default=None,
                    help="Evaluate only groups with this reward label "
                         "(e.g. R7_no_urgency).")
    ap.add_argument("--append", action="store_true",
                    help="Merge into an existing anchor_panel.csv: rows for "
                         "the re-evaluated runs are replaced, others kept.")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip runs that already have rows in "
                         "anchor_panel.csv (incremental mode).")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    env_probe = VecTagEnv(num_envs=1,
                          config=TagEnvConfig(layout=LAYOUT, hider_speed_mult=HSM))
    obs_dim, act_dim = env_probe.obs_dim, env_probe.act_dim
    env_cfg = TagEnvConfig(layout=LAYOUT, hider_speed_mult=HSM)

    anchors = load_anchor_hiders(obs_dim, act_dim)
    print(f"Anchors: {[a[0] for a in anchors]}")
    print(f"Episodes per opponent: {args.episodes}\n")

    groups = [g for g in GROUPS if args.only is None or g[0] == args.only]
    if not groups:
        print(f"ERROR: --only {args.only} matches no group", file=sys.stderr)
        sys.exit(1)

    csv_path = OUT_DIR / "anchor_panel.csv"
    existing_runs = set()
    if args.skip_existing and csv_path.exists():
        prev = pd.read_csv(csv_path)
        existing_runs = {(r, float(A), int(s)) for r, A, s in
                         zip(prev.reward, prev.A, prev.seed)}
        print(f"Incremental: {len(existing_runs)} runs already in panel")

    rows = []
    for reward, A, base, seeds in groups:
        for seed in seeds:
            if (reward, float(A), int(seed)) in existing_runs:
                continue
            ts = discover_run(base, reward, A, seed)
            if ts is None:
                print(f"[skip] {reward}/A={A}/seed_{seed}: no run", file=sys.stderr)
                continue
            seeker = load_policy(ts / "policy_seeker_final.pt", obs_dim, act_dim)
            own_hider = load_policy(ts / "policy_hider_final.pt", obs_dim, act_dim)
            opponents = anchors + [("own", own_hider)]
            wrs = {}
            for opp_label, hider in opponents:
                feats = collect_features(seeker, hider, env_cfg, args.episodes)
                wrs[opp_label] = feats["wr"]
                rows.append(dict(
                    reward=reward, A=A, seed=seed, opponent=opp_label,
                    wr=feats["wr"],
                    wins=int(round(feats["wr"] * args.episodes)),
                    episodes=args.episodes,
                    mean_path_len=feats["mean_path_len"],
                    mean_dist=feats["mean_dist"],
                ))
            anchor_mean = float(np.mean([wrs[a] for a, _ in anchors]))
            detail = "  ".join(
                "{}={:.2f}".format(label.split("_", 1)[1], wrs[label])
                for label, _ in anchors
            )
            print(f"{reward:16s} A={A:<5} seed {seed}  "
                  f"anchor-mean={anchor_mean:.3f}  own={wrs['own']:.3f}  ({detail})",
                  flush=True)

    df = pd.DataFrame(rows)
    if args.append and csv_path.exists():
        evaluated = {(r, float(A), int(s)) for r, A, s in
                     zip(df.reward, df.A, df.seed)}
        old = pd.read_csv(csv_path)
        keep = ~old.apply(
            lambda r: (r["reward"], float(r["A"]), int(r["seed"])) in evaluated,
            axis=1)
        df = pd.concat([old[keep], df], ignore_index=True)
        print(f"\nAppend: kept {int(keep.sum())} existing rows, "
              f"replaced/added {len(rows)}")
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")

    # Cell summary on anchor-mean WR
    anch = df[df.opponent != "own"]
    per_run = anch.groupby(["reward", "A", "seed"]).wr.mean().reset_index()
    print("\nPer-cell anchor-mean WR summary:")
    print(f"{'reward':16s} {'A':>5s}  {'n':>3s}  {'mean':>6s}  {'std':>6s}  "
          f"{'<0.4':>5s} {'<0.5':>5s} {'<0.6':>5s}")
    for (reward, A), g in per_run.groupby(["reward", "A"]):
        print(f"{reward:16s} {A:>5}  {len(g):>3d}  {g.wr.mean():>6.3f}  "
              f"{g.wr.std(ddof=0):>6.3f}  "
              f"{int((g.wr < 0.4).sum()):>5d} {int((g.wr < 0.5).sum()):>5d} "
              f"{int((g.wr < 0.6).sum()):>5d}")


if __name__ == "__main__":
    main()
