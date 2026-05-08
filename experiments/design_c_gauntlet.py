#!/usr/bin/env python3
"""
Design C cross-evaluation gauntlet.

Loads all 12 main-grid + 4 anchor PPO finals, runs every-vs-every cross-eval
with 3-stage successive halving on episode count (see pre-reg v2 §6).

Stage 1: 10 episodes per matchup (all 16×16 = 256 matchups)
Stage 2: continue to 30 ep where Wilson 95% CI for p crosses 0.5 ± 0.10
Stage 3: continue to 100 ep where Wilson 95% CI for p crosses 0.5 ± 0.05

Outputs (under experiments/results/design_c/gauntlet/):
  matchups_long.csv  — one row per episode (seeker_id, hider_id, episode_idx, won)
  matchups_summary.csv — one row per matchup (n, wins, wr, ci_lo, ci_hi, stage)
  policy_index.csv   — id, source, reward, A, seed, path

Usage:
  python experiments/design_c_gauntlet.py             # run full gauntlet
  python experiments/design_c_gauntlet.py --dry-run   # list policies and matchup count
  python experiments/design_c_gauntlet.py --episodes 10  # single-stage override
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from trainer.tag_env import VecTagEnv, TagEnvConfig
from trainer.ppo import PPOAgent, PPOConfig


# ── Pool spec (locked in pre-reg v2 §A4) ────────────────────────────
GRID_BASE = ROOT / "experiments" / "results" / "design_c" / "grid"
ANCHORS_BASE = ROOT / "experiments" / "results" / "design_c" / "anchors"
OUT_DIR = ROOT / "experiments" / "results" / "design_c" / "gauntlet"

REWARDS = ["R4_sparse", "R7_kitchen_sink"]
A_VALUES = [0.0, 0.5]
ANCHOR_SEED = 42
# Grid seeds are auto-discovered from the directory tree so REFINE rounds
# (extra seeds added later) get folded in without code edits.

# Successive-halving stages: (cumulative_episodes, half_width_threshold)
# A matchup advances to the next stage iff its Wilson 95% CI half-width
# at the current stage is wider than the threshold (i.e., still ambiguous).
STAGES = [
    (10, 0.10),
    (30, 0.05),
    (100, None),  # final stage; None means "no further escalation"
]

MAX_STEPS = 200
HSM = 1.15
LAYOUT = "four_corners"


# ── Discovery ───────────────────────────────────────────────────────

def _a_str(A): return f"A{int(A * 100):02d}"


def discover_run(base: Path, reward: str, A: float, seed: int) -> Path | None:
    """Find the timestamped subdir containing policy_*_final.pt for one run."""
    run_dir = base / reward / _a_str(A) / f"seed_{seed}"
    if not run_dir.exists():
        return None
    # Pick the most recent timestamp subdir that has both finals.
    for ts_dir in sorted(run_dir.glob("2*"), reverse=True):
        if (ts_dir / "policy_seeker_final.pt").exists() and \
           (ts_dir / "policy_hider_final.pt").exists():
            return ts_dir
    return None


def discover_grid_seeds(base: Path, reward: str, A: float) -> list[int]:
    """Scan {base}/{reward}/A{aa}/seed_* for any seed directories with a final policy."""
    cell = base / reward / _a_str(A)
    if not cell.exists():
        return []
    seeds = []
    for d in sorted(cell.glob("seed_*")):
        try:
            s = int(d.name.split("_", 1)[1])
        except ValueError:
            continue
        if discover_run(base, reward, A, s) is not None:
            seeds.append(s)
    return seeds


def build_pool() -> List[Dict]:
    """Return list of pool entries, each a dict with id, source, reward, A, seed, ts_dir."""
    pool = []
    pid = 0
    # Grid: seeds auto-discovered so REFINE-round additions get included.
    for reward, A in itertools.product(REWARDS, A_VALUES):
        seeds = discover_grid_seeds(GRID_BASE, reward, A)
        if not seeds:
            print(f"  MISSING grid: {reward} {_a_str(A)} (no seeds found)", file=sys.stderr)
            continue
        for seed in seeds:
            ts = discover_run(GRID_BASE, reward, A, seed)
            pool.append(dict(id=pid, source="grid", reward=reward, A=A, seed=seed, ts_dir=ts))
            pid += 1
    # Anchors
    for reward, A in itertools.product(REWARDS, A_VALUES):
        ts = discover_run(ANCHORS_BASE, reward, A, ANCHOR_SEED)
        if ts is None:
            print(f"  MISSING anchor: {reward} {_a_str(A)} seed={ANCHOR_SEED}", file=sys.stderr)
            continue
        pool.append(dict(id=pid, source="anchor", reward=reward, A=A, seed=ANCHOR_SEED, ts_dir=ts))
        pid += 1
    return pool


# ── Policy loading + matchup play ───────────────────────────────────

def load_policy(path: Path, obs_dim: int, act_dim: int) -> PPOAgent:
    cfg = PPOConfig(obs_dim=obs_dim, act_dim=act_dim)
    agent = PPOAgent(cfg)
    ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
    agent.pi.load_state_dict(ckpt["pi"])
    agent.vf.load_state_dict(ckpt["vf"])
    return agent


def batch_act(policy: PPOAgent, obs_batch) -> np.ndarray:
    with torch.no_grad():
        x = torch.as_tensor(obs_batch, dtype=torch.float32)
        logits = policy.pi(x)
        mean, log_std = torch.chunk(logits, 2, dim=-1)
        log_std = torch.clamp(log_std, -2.0, 1.5)
        std = torch.exp(log_std)
        action = torch.tanh(torch.distributions.Normal(mean, std).sample())
        return action.cpu().numpy()


def play_episodes(seeker: PPOAgent, hider: PPOAgent, env_config: TagEnvConfig,
                  n_episodes: int) -> np.ndarray:
    """Run n_episodes vectorized; return shape (n_episodes,) of {0,1} = seeker won."""
    env = VecTagEnv(num_envs=n_episodes, config=env_config)
    obs = env.reset()
    active = np.ones(n_episodes, dtype=bool)
    tagged = np.zeros(n_episodes, dtype=bool)

    for _ in range(MAX_STEPS):
        if not active.any():
            break
        s_act = batch_act(seeker, obs["seeker"])
        h_act = batch_act(hider, obs["hider"])
        obs, _, dones, info = env.step({"seeker": s_act, "hider": h_act})
        newly_done = dones & active
        if newly_done.any():
            tagged[newly_done] = info.get("tagged", np.zeros(n_episodes, dtype=bool))[newly_done]
            active[newly_done] = False

    return tagged.astype(np.int8)


# ── Wilson CI + halving ─────────────────────────────────────────────

def wilson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float, float]:
    """Wilson score interval for a binomial proportion. Returns (p_hat, lo, hi)."""
    if n == 0:
        return 0.5, 0.0, 1.0
    z = 1.959963984540054  # 97.5th pct
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    halfw = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, max(0.0, centre - halfw), min(1.0, centre + halfw)


def is_ambiguous(k: int, n: int, half_threshold: float) -> bool:
    """A matchup is still ambiguous iff its Wilson 95% CI half-width > half_threshold
       OR the CI brackets 0.5 (the analytically interesting boundary)."""
    p, lo, hi = wilson_ci(k, n)
    halfw = (hi - lo) / 2.0
    return halfw > half_threshold or (lo < 0.5 < hi)


# ── Main gauntlet ───────────────────────────────────────────────────

def run_gauntlet(dry_run: bool = False, episodes_override: int | None = None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pool = build_pool()
    n = len(pool)
    print(f"Pool size: {n} policies (need 16 for full design)")
    if n == 0:
        print("Empty pool; aborting.", file=sys.stderr)
        sys.exit(1)

    # Write policy index immediately
    with open(OUT_DIR / "policy_index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "source", "reward", "A", "seed", "ts_dir"])
        for p in pool:
            w.writerow([p["id"], p["source"], p["reward"], p["A"], p["seed"], str(p["ts_dir"])])

    n_matchups = n * n
    print(f"Matchups: {n_matchups}")
    if dry_run:
        for p in pool:
            print(f"  {p['id']:>2d}  {p['source']:<6s}  {p['reward']:<18s}  A={p['A']}  seed={p['seed']}  -> {p['ts_dir'].name}")
        return

    env_config = TagEnvConfig(layout=LAYOUT, hider_speed_mult=HSM)
    env_probe = VecTagEnv(num_envs=1, config=env_config)
    obs_dim, act_dim = env_probe.obs_dim, env_probe.act_dim

    # Load all policies once.
    seekers = [None] * n
    hiders = [None] * n
    for p in pool:
        seekers[p["id"]] = load_policy(p["ts_dir"] / "policy_seeker_final.pt", obs_dim, act_dim)
        hiders[p["id"]]  = load_policy(p["ts_dir"] / "policy_hider_final.pt", obs_dim, act_dim)

    # Per-matchup running outcomes (i = seeker id, j = hider id).
    outcomes: Dict[Tuple[int, int], List[int]] = {(i, j): [] for i in range(n) for j in range(n)}
    final_stage: Dict[Tuple[int, int], int] = {}

    if episodes_override is not None:
        stages = [(episodes_override, None)]
    else:
        stages = STAGES

    active = set(outcomes.keys())
    for stage_idx, (cum_ep, halt_threshold) in enumerate(stages, start=1):
        print(f"\n=== Stage {stage_idx}: cumulative_episodes={cum_ep}, threshold={halt_threshold} ===")
        print(f"  Matchups still active: {len(active)}")
        for k, (i, j) in enumerate(sorted(active), start=1):
            have = len(outcomes[(i, j)])
            need = cum_ep - have
            if need > 0:
                new = play_episodes(seekers[i], hiders[j], env_config, need)
                outcomes[(i, j)].extend(int(x) for x in new)
            if k % 25 == 0 or k == len(active):
                print(f"    [{k}/{len(active)}] (i={i}, j={j}) n={len(outcomes[(i,j)])} k={sum(outcomes[(i,j)])}")
        # Decide which matchups advance to the next stage.
        if halt_threshold is None:
            # Final stage: lock in current stage for all still-active matchups.
            for key in active:
                final_stage[key] = stage_idx
            active = set()
            break
        new_active = set()
        for key in active:
            k_ = sum(outcomes[key])
            n_ = len(outcomes[key])
            if is_ambiguous(k_, n_, halt_threshold):
                new_active.add(key)
            else:
                final_stage[key] = stage_idx
        active = new_active

    # Anything still active after all stages = locked at final stage's count.
    for key in active:
        final_stage[key] = len(stages)

    # ── Write outputs ─────────────────────────────────────────────
    long_path = OUT_DIR / "matchups_long.csv"
    summary_path = OUT_DIR / "matchups_summary.csv"

    with open(long_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seeker_id", "hider_id", "episode_idx", "won"])
        for (i, j), outs in outcomes.items():
            for ep, won in enumerate(outs):
                w.writerow([i, j, ep, won])

    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "seeker_id", "hider_id",
            "seeker_reward", "seeker_A", "seeker_seed", "seeker_source",
            "hider_reward", "hider_A", "hider_seed", "hider_source",
            "n", "wins", "wr", "ci_lo", "ci_hi", "stage_reached", "self_pair",
        ])
        by_id = {p["id"]: p for p in pool}
        for (i, j), outs in outcomes.items():
            n_ = len(outs)
            k_ = sum(outs)
            p_, lo, hi = wilson_ci(k_, n_)
            is_self = (i == j)
            si = by_id[i]; hj = by_id[j]
            w.writerow([
                i, j,
                si["reward"], si["A"], si["seed"], si["source"],
                hj["reward"], hj["A"], hj["seed"], hj["source"],
                n_, k_, p_, lo, hi, final_stage.get((i, j), 1), int(is_self),
            ])

    print(f"\nWrote:")
    print(f"  {long_path}    ({sum(len(v) for v in outcomes.values())} episodes)")
    print(f"  {summary_path} ({len(outcomes)} matchups)")
    print(f"  {OUT_DIR / 'policy_index.csv'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="List pool and matchup count, don't play")
    ap.add_argument("--episodes", type=int, default=None,
                    help="Override stages: run a single fixed-N gauntlet (skip successive halving)")
    args = ap.parse_args()
    run_gauntlet(dry_run=args.dry_run, episodes_override=args.episodes)


if __name__ == "__main__":
    main()
