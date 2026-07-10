#!/usr/bin/env python3
"""
Coverage-ablation eval — matched comparison against the trajectory analysis.

Scores the R7_no_coverage/A=0.0 seekers (the ablation) and the
R7_kitchen_sink/A=0.0 seekers (the baseline) against the SAME fixed reference
hider used by design_c_trajectory_analysis.py (default: anchor id=28, the
R4_sparse seed-42 hider). Both groups are evaluated in one run with identical
settings, so win rates are directly comparable to the trajectory table and to
each other.

Question: does removing --area-coverage-bonus eliminate the R7/A=0.0
"roamer" bad basin (4/9 baseline seeds collapsed to wr 0.10-0.33)?

Outputs:
    experiments/results/design_c/coverage_ablation/ablation_eval.csv
    + console comparison table

Usage:
    python experiments/design_c_coverage_ablation_eval.py
    python experiments/design_c_coverage_ablation_eval.py --episodes 30 --reference 28
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
ABL_DIR = ROOT / "experiments" / "results" / "design_c" / "coverage_ablation"
URG_DIR = ROOT / "experiments" / "results" / "design_c" / "urgency_ablation"
ASW_DIR = ROOT / "experiments" / "results" / "design_c" / "a_sweep"
OUT_DIR = ABL_DIR
COLLAPSE_WR = 0.5  # below this = failed to learn pursuit (clear gap in trajectory data)

# (label, reward subdir, A, base dir, seeds)
GROUPS = [
    ("R7_kitchen_sink/A=0",    "R7_kitchen_sink", 0.0,  GRID_DIR, list(range(9))),
    ("R7_kitchen_sink/A=0.05", "R7_kitchen_sink", 0.05, ASW_DIR,  list(range(10))),
    ("R7_kitchen_sink/A=0.10", "R7_kitchen_sink", 0.10, ASW_DIR,  list(range(10))),
    ("R7_kitchen_sink/A=0.25", "R7_kitchen_sink", 0.25, ASW_DIR,  list(range(10))),
    ("R7_kitchen_sink/A=0.5",  "R7_kitchen_sink", 0.5,  GRID_DIR, list(range(9))),
    ("R7_no_coverage/A=0",     "R7_no_coverage",  0.0,  ABL_DIR,  list(range(10))),
    ("R7_no_coverage/A=0.5",   "R7_no_coverage",  0.5,  ABL_DIR,  list(range(5))),
    ("R7_no_cov_urg/A=0",      "R7_no_cov_urg",   0.0,  URG_DIR,  list(range(10))),
]


def eval_group(label, reward, A, base, seeds, ref_hider, env_cfg, episodes,
               obs_dim, act_dim):
    rows = []
    for seed in seeds:
        ts = discover_run(base, reward, A, seed)
        if ts is None:
            print(f"  [skip] {label} seed {seed}: no run found", file=sys.stderr)
            continue
        seeker = load_policy(ts / "policy_seeker_final.pt", obs_dim, act_dim)
        feats = collect_features(seeker, ref_hider, env_cfg, episodes)
        feats.update(dict(group=label, reward=reward, A=A, seed=seed))
        rows.append(feats)
        print(f"  {label:22s} seed {seed}  wr={feats['wr']:.3f}  "
              f"path={feats['mean_path_len']:5.1f}  dist={feats['mean_dist']:6.2f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30,
                    help="Episodes per seeker vs the reference hider.")
    ap.add_argument("--reference", type=int, default=28,
                    help="Policy id for the reference hider (default 28 = R4 anchor).")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    idx = pd.read_csv(GAUNTLET_DIR / "policy_index.csv")
    ref_row = idx[idx.id == args.reference].iloc[0]
    print(f"Reference hider: id={args.reference} "
          f"({ref_row.source}/{ref_row.reward}/A{ref_row.A}/s{ref_row.seed})")
    print(f"Episodes per seeker: {args.episodes}\n")

    env_probe = VecTagEnv(num_envs=1, config=TagEnvConfig(layout=LAYOUT, hider_speed_mult=HSM))
    obs_dim, act_dim = env_probe.obs_dim, env_probe.act_dim
    env_cfg = TagEnvConfig(layout=LAYOUT, hider_speed_mult=HSM)

    ref_hider = load_policy(Path(ref_row.ts_dir) / "policy_hider_final.pt", obs_dim, act_dim)

    rows = []
    for label, reward, A, base, seeds in GROUPS:
        rows += eval_group(label, reward, A, base, seeds, ref_hider, env_cfg,
                           args.episodes, obs_dim, act_dim)

    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "ablation_eval.csv"
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 60)
    print(f"Matched eval vs reference hider id={args.reference}")
    print("=" * 60)
    for label, _, _, _, _ in GROUPS:
        g = df[df.group == label]
        if g.empty:
            continue
        collapsed = g[g.wr < COLLAPSE_WR]
        print(f"\n{label}  (n={len(g)})")
        print(f"  win rate: mean={g.wr.mean():.3f}  std={g.wr.std(ddof=0):.3f}  "
              f"min={g.wr.min():.3f}  max={g.wr.max():.3f}")
        print(f"  collapsed (wr<{COLLAPSE_WR}): {len(collapsed)}/{len(g)}"
              + (f"  -> seeds {sorted(collapsed.seed.tolist())}" if len(collapsed) else ""))

    # β_RA identifying readout: 2x2 of (R7_kitchen_sink, R7_no_coverage) x (A=0, A=0.5).
    # If A's WR-lift is much larger at R7_kitchen_sink than at R7_no_coverage,
    # the original interaction was dominated by coverage-rescue. If A's lift is
    # similar at both rewards, the substitution effect is real.
    cells = {}
    for r in ("R7_kitchen_sink", "R7_no_coverage"):
        for A in (0.0, 0.5):
            g = df[(df.reward == r) & (df.A == A)]
            if not g.empty:
                cells[(r, A)] = g.wr.mean()
    if len(cells) == 4:
        m_R7_A0 = cells[("R7_kitchen_sink", 0.0)]
        m_R7_A5 = cells[("R7_kitchen_sink", 0.5)]
        m_NC_A0 = cells[("R7_no_coverage", 0.0)]
        m_NC_A5 = cells[("R7_no_coverage", 0.5)]
        lift_R7 = m_R7_A5 - m_R7_A0
        lift_NC = m_NC_A5 - m_NC_A0
        # Log-odds version (matches prereg model space; clip to avoid log(0))
        def _lo(p):
            p = float(np.clip(p, 0.001, 0.999))
            return float(np.log(p / (1 - p)))
        lift_R7_lo = _lo(m_R7_A5) - _lo(m_R7_A0)
        lift_NC_lo = _lo(m_NC_A5) - _lo(m_NC_A0)
        beta_RA_proxy_lo = lift_R7_lo - lift_NC_lo
        print("\n" + "=" * 60)
        print("β_RA identifying readout (R7_no_coverage substituted for R4)")
        print("=" * 60)
        print(f"                 A=0.0     A=0.5     A-lift (WR)   A-lift (log-odds)")
        print(f"R7_kitchen_sink  {m_R7_A0:.3f}     {m_R7_A5:.3f}     {lift_R7:+.3f}        {lift_R7_lo:+.3f}")
        print(f"R7_no_coverage   {m_NC_A0:.3f}     {m_NC_A5:.3f}     {lift_NC:+.3f}        {lift_NC_lo:+.3f}")
        print(f"\nProxy β'_RA (log-odds DiD) = {beta_RA_proxy_lo:+.3f}")
        # Logic: large positive β'_RA means A=0.5 helps R7 (with coverage) much more
        # than it helps R7_no_coverage. That is the *signature* of A=0.5 rescuing R7
        # specifically from the coverage-induced basin, with little left to offer once
        # coverage is removed — i.e. A and coverage-removal are substitutes for the
        # *rescue* function. Small β'_RA would mean A helps both rewards equally and
        # the original β_RA reflected a coverage-independent substitution effect.
        if abs(beta_RA_proxy_lo) < 0.22:  # < log(1.25), prereg's PURSUE threshold
            print("VERDICT: A's lift is comparable at both rewards. The original β_RA "
                  "reflected a coverage-independent A effect; substitution claim survives "
                  "de-confounding.")
        else:
            print("VERDICT: A's lift is substantially larger under R7_kitchen_sink than "
                  "R7_no_coverage. The original β_RA was dominated by A=0.5 rescuing R7 "
                  "from the coverage-induced basin; A and coverage-removal are substitutes "
                  "for that rescue function. The substitution effect is mechanism-narrow, "
                  "not a general shaping substitute.")

    # A-saturation curve for R7_kitchen_sink (rescue characterization).
    r7 = df[df.reward == "R7_kitchen_sink"].copy()
    if not r7.empty:
        print("\n" + "=" * 60)
        print("Rescue saturation curve — R7_kitchen_sink vs A")
        print("=" * 60)
        print(f"{'A':>6s}  {'n':>3s}  {'mean':>6s}  {'std':>6s}  {'collapsed':>9s}")
        for A in sorted(r7.A.unique()):
            sub = r7[r7.A == A]
            n_coll = int((sub.wr < COLLAPSE_WR).sum())
            print(f"{A:>6.2f}  {len(sub):>3d}  {sub.wr.mean():>6.3f}  "
                  f"{sub.wr.std(ddof=0):>6.3f}  {n_coll:>4d}/{len(sub):<3d}")

    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
