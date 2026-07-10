#!/usr/bin/env python3
"""
HSM within-cell round-robin — is the R7/A=0 bad basin specific to HSM=1.15?

Companion eval for design_c_hsm_flank_tasks.py. The Design C anchors were
all trained at HSM=1.15, so the matched anchor-panel eval cannot score the
HSM 1.05/1.20 flank runs fairly (a seeker trained vs 1.05 hiders meets a
1.15 anchor at the wrong speed regime). Instead, each cell is scored
against itself: every seeker plays every same-cell hider at the cell's
NATIVE hider speed.

Cells (R7_kitchen_sink, A=0 throughout):
  HSM 1.05  experiments/results/design_c/hsm_flank/R7_kitchen_sink/HSM105
  HSM 1.15  experiments/results/design_c/grid/R7_kitchen_sink/A00  (reference)
  HSM 1.20  experiments/results/design_c/hsm_flank/R7_kitchen_sink/HSM120
8 seeds per cell (the 1.15 grid cell has 9; seeds 0-7 are used so all
matrices are 8x8).

Read-out per seeker: cross-WR = mean WR vs the 7 OTHER hiders (the
diagonal is the seeker's own co-adapted hider — that is the
over-specialization probe, reported separately). The basin claim predicts
a bimodal cross-WR distribution at 1.15; if 1.05 and 1.20 are unimodal,
the coverage story needs an HSM-contingency caveat.

Output:
  experiments/results/design_c/hsm_roundrobin/hsm_roundrobin.csv

Usage:
  python experiments/design_c_hsm_roundrobin.py --episodes 30
  python experiments/design_c_hsm_roundrobin.py --hsm 1.05   # one cell only
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
from design_c_trajectory_analysis import load_policy, collect_features, LAYOUT

GRID_DIR = ROOT / "experiments" / "results" / "design_c" / "grid"
FLANK_DIR = ROOT / "experiments" / "results" / "design_c" / "hsm_flank"
OUT_DIR = ROOT / "experiments" / "results" / "design_c" / "hsm_roundrobin"

SEEDS = list(range(8))

# (hsm, seed -> run dir). The flank tree keys cells by HSM, the grid by A.
CELLS = [
    (1.05, lambda s: FLANK_DIR / "R7_kitchen_sink" / "HSM105" / f"seed_{s}"),
    (1.15, lambda s: GRID_DIR / "R7_kitchen_sink" / "A00" / f"seed_{s}"),
    (1.20, lambda s: FLANK_DIR / "R7_kitchen_sink" / "HSM120" / f"seed_{s}"),
]


def discover(run_dir: Path) -> Path | None:
    if not run_dir.exists():
        return None
    for ts_dir in sorted(run_dir.glob("2*"), reverse=True):
        if (ts_dir / "policy_seeker_final.pt").exists() and \
           (ts_dir / "policy_hider_final.pt").exists():
            return ts_dir
    return None


def load_cell(hsm: float, run_dir_fn, obs_dim: int, act_dim: int):
    """Load (seed, seeker, hider) for every completed seed in a cell."""
    runs = []
    for seed in SEEDS:
        ts = discover(run_dir_fn(seed))
        if ts is None:
            print(f"[skip] HSM={hsm} seed_{seed}: no completed run", file=sys.stderr)
            continue
        runs.append((
            seed,
            load_policy(ts / "policy_seeker_final.pt", obs_dim, act_dim),
            load_policy(ts / "policy_hider_final.pt", obs_dim, act_dim),
        ))
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30,
                    help="Episodes per (seeker, hider) pair.")
    ap.add_argument("--hsm", type=float, default=None,
                    help="Evaluate only this HSM cell (e.g. 1.05).")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    env_probe = VecTagEnv(num_envs=1,
                          config=TagEnvConfig(layout=LAYOUT, hider_speed_mult=1.15))
    obs_dim, act_dim = env_probe.obs_dim, env_probe.act_dim

    cells = [c for c in CELLS if args.hsm is None or abs(c[0] - args.hsm) < 1e-9]
    if not cells:
        print(f"ERROR: no cell matches --hsm {args.hsm}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for hsm, run_dir_fn in cells:
        runs = load_cell(hsm, run_dir_fn, obs_dim, act_dim)
        if len(runs) < 2:
            print(f"[skip] HSM={hsm}: only {len(runs)} run(s), need >=2 for round-robin",
                  file=sys.stderr)
            continue
        env_cfg = TagEnvConfig(layout=LAYOUT, hider_speed_mult=hsm)
        print(f"\nHSM={hsm}: {len(runs)} runs, "
              f"{len(runs) ** 2} pairs x {args.episodes} episodes")
        for s_seed, seeker, _ in runs:
            for h_seed, _, hider in runs:
                feats = collect_features(seeker, hider, env_cfg, args.episodes)
                rows.append(dict(
                    hsm=hsm, seeker_seed=s_seed, hider_seed=h_seed,
                    own=(s_seed == h_seed),
                    wr=feats["wr"],
                    wins=int(round(feats["wr"] * args.episodes)),
                    episodes=args.episodes,
                    mean_path_len=feats["mean_path_len"],
                    mean_dist=feats["mean_dist"],
                ))
            row_wrs = [r["wr"] for r in rows
                       if r["hsm"] == hsm and r["seeker_seed"] == s_seed]
            cross = [r["wr"] for r in rows
                     if r["hsm"] == hsm and r["seeker_seed"] == s_seed and not r["own"]]
            own = [r["wr"] for r in rows
                   if r["hsm"] == hsm and r["seeker_seed"] == s_seed and r["own"]]
            print(f"  seeker seed {s_seed}: cross={np.mean(cross):.3f}  "
                  f"own={own[0]:.3f}  all={np.mean(row_wrs):.3f}", flush=True)

    if not rows:
        print("No cells evaluated; nothing to write.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "hsm_roundrobin.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")

    print("\nPer-cell cross-WR summary (diagonal excluded):")
    print(f"{'HSM':>5s}  {'n':>3s}  {'mean':>6s}  {'std':>6s}  {'min':>6s}  "
          f"{'max':>6s}  {'<0.4':>5s} {'<0.5':>5s} {'<0.6':>5s}   sorted per-seeker cross-WR")
    cross = df[~df.own]
    per_seeker = cross.groupby(["hsm", "seeker_seed"]).wr.mean().reset_index()
    for hsm, g in per_seeker.groupby("hsm"):
        vals = np.sort(g.wr.values)
        print(f"{hsm:>5}  {len(g):>3d}  {g.wr.mean():>6.3f}  {g.wr.std(ddof=0):>6.3f}  "
              f"{vals[0]:>6.3f}  {vals[-1]:>6.3f}  "
              f"{int((g.wr < 0.4).sum()):>5d} {int((g.wr < 0.5).sum()):>5d} "
              f"{int((g.wr < 0.6).sum()):>5d}   "
              + " ".join(f"{v:.2f}" for v in vals))

    print("\nOver-specialization probe (own minus cross, per seeker):")
    own = df[df.own].set_index(["hsm", "seeker_seed"]).wr
    for (hsm, seed), cross_wr in per_seeker.set_index(["hsm", "seeker_seed"]).wr.items():
        gap = own.loc[(hsm, seed)] - cross_wr
        flag = "  <-- over-specialized" if gap > 0.3 and cross_wr < 0.5 else ""
        print(f"  HSM={hsm} seed {seed}: own-cross gap = {gap:+.3f}{flag}")


if __name__ == "__main__":
    main()
