#!/usr/bin/env python3
"""
Hider-behavior characterization across A — the unmeasured mechanism.

The idea-critic (concern g): claim 5 ("A's substitution effect is
coverage-rescue / opponent-diversity") is fundamentally a story about how A
reshapes the HIDER distribution, yet no hider behavior was ever measured.

This script rolls every trained hider out against two fixed probe seekers
(the R7/A=0 anchor seeker = strong pursuit probe; the R4/A=0 anchor seeker
= weak probe) and collects HIDER trajectory features: survival rate, path
length, mean distance to seeker, wall/center occupancy.

Per-cell dispersion across seeds (by reward x A) answers: does the hider
distribution actually shift or diversify with A in ways that could carry
the over-specialization mechanism?

Output:
  experiments/results/design_c/anchor_panel/hider_panel.csv

Usage:
  python experiments/design_c_hider_panel_eval.py --episodes 30
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
    load_policy, batch_act, discover_run, GAUNTLET_DIR, HSM, LAYOUT, MAX_STEPS,
)
from design_c_anchor_panel_eval import GROUPS, OUT_DIR


def collect_hider_features(seeker, hider, env_config, n_episodes):
    """Roll out n_episodes; return aggregated HIDER trajectory features."""
    env = VecTagEnv(num_envs=n_episodes, config=env_config)
    obs = env.reset()
    active = np.ones(n_episodes, dtype=bool)
    tagged = np.zeros(n_episodes, dtype=bool)
    steps_taken = np.zeros(n_episodes, dtype=np.int32)

    path_len = np.zeros(n_episodes, dtype=np.float32)
    last_pos = env.positions[:, 1].copy()  # hider is index 1
    dist_sum = np.zeros(n_episodes, dtype=np.float32)
    time_in_center = np.zeros(n_episodes, dtype=np.float32)
    time_at_wall = np.zeros(n_episodes, dtype=np.float32)

    for step in range(MAX_STEPS):
        if not active.any():
            break
        s_act = batch_act(seeker, obs["seeker"])
        h_act = batch_act(hider, obs["hider"])
        obs, _, dones, info = env.step({"seeker": s_act, "hider": h_act})

        newly_done = dones & active
        if newly_done.any():
            tagged[newly_done] = info.get(
                "tagged", np.zeros(n_episodes, dtype=bool))[newly_done]
            steps_taken[newly_done] = step + 1

        cur_pos = env.positions[:, 1]  # hider
        seeker_pos = env.positions[:, 0]
        diff = cur_pos - last_pos
        path_len[active] += np.linalg.norm(diff[active], axis=-1)
        dist_sum[active] += np.linalg.norm((cur_pos - seeker_pos)[active], axis=-1)
        in_center = (np.abs(cur_pos[:, 0]) < 3.0) & (np.abs(cur_pos[:, 1]) < 3.0)
        time_in_center[active & in_center] += 1
        at_wall = (np.abs(cur_pos[:, 0]) > 6.0) | (np.abs(cur_pos[:, 1]) > 6.0)
        time_at_wall[active & at_wall] += 1

        last_pos = cur_pos.copy()
        active &= ~newly_done

    if active.any():
        steps_taken[active] = MAX_STEPS

    n_actual = np.maximum(steps_taken, 1)
    return dict(
        hider_survival=float(1.0 - tagged.mean()),
        hider_mean_steps=float(steps_taken.mean()),
        hider_path_len=float(path_len.mean()),
        hider_speed=float((path_len / n_actual).mean()),
        hider_mean_dist=float((dist_sum / n_actual).mean()),
        hider_frac_center=float((time_in_center / n_actual).mean()),
        hider_frac_wall=float((time_at_wall / n_actual).mean()),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    env_probe = VecTagEnv(num_envs=1,
                          config=TagEnvConfig(layout=LAYOUT, hider_speed_mult=HSM))
    obs_dim, act_dim = env_probe.obs_dim, env_probe.act_dim
    env_cfg = TagEnvConfig(layout=LAYOUT, hider_speed_mult=HSM)

    idx = pd.read_csv(GAUNTLET_DIR / "policy_index.csv")
    anchors = idx[idx.source == "anchor"]

    probes = []
    for reward, label in [("R7_kitchen_sink", "probe_R7_A00"),
                          ("R4_sparse", "probe_R4_A00")]:
        row = anchors[(anchors.reward == reward) & (anchors.A == 0.0)].iloc[0]
        probes.append((label,
                       load_policy(Path(row.ts_dir) / "policy_seeker_final.pt",
                                   obs_dim, act_dim)))
    print(f"Probe seekers: {[p[0] for p in probes]}")
    print(f"Episodes per (probe, hider): {args.episodes}\n")

    rows = []
    for reward, A, base, seeds in GROUPS:
        for seed in seeds:
            ts = discover_run(base, reward, A, seed)
            if ts is None:
                continue
            hider = load_policy(ts / "policy_hider_final.pt", obs_dim, act_dim)
            for probe_label, probe in probes:
                feats = collect_hider_features(probe, hider, env_cfg, args.episodes)
                feats.update(dict(reward=reward, A=A, seed=seed, probe=probe_label))
                rows.append(feats)
            print(f"{reward:16s} A={A:<5} seed {seed}  "
                  f"survival(strong probe)={rows[-2]['hider_survival']:.2f}  "
                  f"dist={rows[-2]['hider_mean_dist']:.1f}", flush=True)

    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "hider_panel.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")

    # Per-cell dispersion (across seeds) of hider features vs the strong probe:
    # the across-seed std IS the "hider distribution diversity" measure.
    feat_cols = ["hider_survival", "hider_path_len", "hider_mean_dist",
                 "hider_frac_wall", "hider_frac_center"]
    strong = df[df.probe == "probe_R7_A00"]
    print("\nHider-style dispersion across seeds (strong probe), by cell:")
    print(f"{'reward':16s} {'A':>5s}  {'n':>3s}  " +
          "  ".join(f"{c.replace('hider_', ''):>14s}(m/sd)" for c in feat_cols))
    for (reward, A), g in strong.groupby(["reward", "A"]):
        stats = "  ".join(
            f"{g[c].mean():>7.2f}/{g[c].std(ddof=0):<6.2f}" for c in feat_cols)
        print(f"{reward:16s} {A:>5}  {len(g):>3d}  {stats}")


if __name__ == "__main__":
    main()
