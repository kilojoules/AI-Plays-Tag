#!/usr/bin/env python3
"""
Cross-evaluate policies from an A-sweep.

Loads final policies from each (algorithm, A, zoo_mode, seed) experiment
and runs round-robin evaluation to measure robustness and cross-play.

Usage:
    python experiments/cross_evaluate_sweep.py experiments/results/sweep/
    python experiments/cross_evaluate_sweep.py experiments/results/sweep/ppo --episodes 200
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from trainer.tag_env import SingleTagEnv, TagEnvConfig
from trainer.ppo import PPOAgent, PPOConfig
from trainer.sac import SACAgent, SACConfig


def detect_policy_type(path: str) -> str:
    """Detect whether a checkpoint is PPO or SAC."""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict) and ckpt.get('type') == 'sac':
        return 'sac'
    return 'ppo'


def load_policy(path: str, obs_dim: int, act_dim: int):
    """Load a policy checkpoint (PPO or SAC). Returns (agent, type_str)."""
    policy_type = detect_policy_type(path)
    if policy_type == 'sac':
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        hidden_dim = ckpt.get('config', {}).get('hidden_dim', 256)
        cfg = SACConfig(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=hidden_dim)
        agent = SACAgent(cfg)
        agent.load_policy(path)
        return agent, 'sac'
    else:
        cfg = PPOConfig(obs_dim=obs_dim, act_dim=act_dim)
        agent = PPOAgent(cfg)
        agent.load_policy(path)
        return agent, 'ppo'


def find_final_policies(sweep_dir: str) -> List[Dict]:
    """Discover all final policies in a sweep directory.

    Handles both flat naming (A05_hider_only/TIMESTAMP/) and
    seeded naming (A05__hider_only/seed_0/TIMESTAMP/).

    Returns list of dicts with keys: name, seeker_path, hider_path, A, zoo_mode, seed, algorithm.
    """
    sweep_path = Path(sweep_dir)
    policies = []

    for algo_dir in sorted(sweep_path.iterdir()):
        if not algo_dir.is_dir():
            continue

        algorithm = algo_dir.name  # 'ppo' or 'sac'
        if algorithm not in ('ppo', 'sac'):
            # Might be a direct experiment dir (no algo subdir)
            algorithm = _guess_algorithm(algo_dir)
            _scan_experiment_dir(algo_dir, algorithm, policies)
            continue

        for exp_dir in sorted(algo_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            _scan_experiment_dir(exp_dir, algorithm, policies)

    return policies


def _guess_algorithm(exp_dir: Path) -> str:
    """Guess algorithm from metadata or checkpoint type."""
    for meta_path in exp_dir.rglob("metadata.json"):
        try:
            meta = json.loads(meta_path.read_text())
            return meta.get('algorithm', 'ppo')
        except (json.JSONDecodeError, KeyError):
            pass
    return 'ppo'


def _scan_experiment_dir(exp_dir: Path, algorithm: str, policies: List[Dict]):
    """Recursively find final policies in an experiment directory."""
    # Parse A value and zoo_mode from directory name
    name = exp_dir.name
    A, zoo_mode = _parse_experiment_name(name)

    # Check for seeded subdirectories
    seed_dirs = sorted(exp_dir.glob("seed_*"))
    if seed_dirs:
        for seed_dir in seed_dirs:
            seed = int(seed_dir.name.split('_')[1])
            _find_policies_in_run(seed_dir, algorithm, A, zoo_mode, seed, policies)
    else:
        _find_policies_in_run(exp_dir, algorithm, A, zoo_mode, None, policies)


def _parse_experiment_name(name: str) -> Tuple[Optional[float], Optional[str]]:
    """Parse 'A05_hider_only' or 'A05__hider_only' -> (0.05, 'hider_only')."""
    # Handle selfplay directories
    if name.startswith('selfplay'):
        return 0.0, 'selfplay'

    parts = name.replace('__', '_').split('_', 1)
    A = None
    zoo_mode = None

    if parts[0].startswith('A'):
        try:
            A = int(parts[0][1:]) / 100.0
        except ValueError:
            pass
    if len(parts) > 1:
        zoo_mode = parts[1]

    return A, zoo_mode


def _find_policies_in_run(run_dir: Path, algorithm: str,
                           A: Optional[float], zoo_mode: Optional[str],
                           seed: Optional[int], policies: List[Dict]):
    """Find final policies in a run directory (possibly with timestamp subdir)."""
    # Look for final policies directly
    seeker = run_dir / "policy_seeker_final.pt"
    hider = run_dir / "policy_hider_final.pt"

    if seeker.exists() and hider.exists():
        label = f"{algorithm}/A={A:.2f}/{zoo_mode}" if A is not None else run_dir.name
        if seed is not None:
            label += f"/s{seed}"
        policies.append({
            'name': label,
            'seeker_path': str(seeker),
            'hider_path': str(hider),
            'A': A,
            'zoo_mode': zoo_mode,
            'seed': seed,
            'algorithm': algorithm,
        })
        return

    # Check timestamped subdirectories
    for ts_dir in sorted(run_dir.glob("2*"), reverse=True):
        seeker = ts_dir / "policy_seeker_final.pt"
        hider = ts_dir / "policy_hider_final.pt"
        if seeker.exists() and hider.exists():
            label = f"{algorithm}/A={A:.2f}/{zoo_mode}" if A is not None else run_dir.name
            if seed is not None:
                label += f"/s{seed}"
            policies.append({
                'name': label,
                'seeker_path': str(seeker),
                'hider_path': str(hider),
                'A': A,
                'zoo_mode': zoo_mode,
                'seed': seed,
                'algorithm': algorithm,
            })
            return  # Use latest only


def evaluate_matchup(seeker_agent, hider_agent,
                     num_episodes: int = 100,
                     env_config: Optional[TagEnvConfig] = None) -> Dict[str, float]:
    """Evaluate a seeker vs hider matchup."""
    env = SingleTagEnv(config=env_config)

    seeker_wins = 0
    total_steps = 0
    seeker_rewards = []
    hider_rewards = []

    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        step = 0
        ep_sr, ep_hr = 0.0, 0.0

        while not done and step < 200:
            s_act, _, _ = seeker_agent.act(obs['seeker'])
            h_act, _, _ = hider_agent.act(obs['hider'])
            obs, rewards, done, info = env.step({'seeker': s_act, 'hider': h_act})
            ep_sr += rewards['seeker']
            ep_hr += rewards['hider']
            step += 1

        total_steps += step
        seeker_rewards.append(ep_sr)
        hider_rewards.append(ep_hr)
        if info.get('tagged', False):
            seeker_wins += 1

    return {
        'seeker_win_rate': seeker_wins / num_episodes,
        'avg_episode_length': total_steps / num_episodes,
        'seeker_reward_mean': float(np.mean(seeker_rewards)),
        'hider_reward_mean': float(np.mean(hider_rewards)),
    }


def main():
    parser = argparse.ArgumentParser(description="Cross-evaluate sweep policies")
    parser.add_argument("sweep_dir", type=str,
                        help="Path to sweep results directory")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Episodes per matchup (default: 100)")
    parser.add_argument("--layout", type=str, default="four_corners",
                        choices=["empty", "four_corners", "central_cross", "playground"],
                        help="Arena layout for evaluation (default: four_corners)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: sweep_dir/cross_evaluation.json)")
    args = parser.parse_args()

    env_config = TagEnvConfig(layout=args.layout)
    env = SingleTagEnv(config=env_config)
    obs_dim, act_dim = env.obs_dim, env.act_dim

    # Discover policies
    print("Scanning for policies...")
    policy_entries = find_final_policies(args.sweep_dir)

    if not policy_entries:
        print(f"No final policies found in {args.sweep_dir}")
        sys.exit(1)

    print(f"Found {len(policy_entries)} policy pairs:")
    for p in policy_entries:
        print(f"  {p['name']}")

    # Load all policies
    print("\nLoading policies...")
    loaded = []
    for entry in policy_entries:
        seeker, s_type = load_policy(entry['seeker_path'], obs_dim, act_dim)
        hider, h_type = load_policy(entry['hider_path'], obs_dim, act_dim)
        loaded.append({**entry, 'seeker_agent': seeker, 'hider_agent': hider})
    print(f"Loaded {len(loaded)} policy pairs.")

    # Run cross-evaluation: each seeker vs each hider
    print(f"\nRunning cross-evaluation ({args.episodes} episodes per matchup)...")
    print(f"Total matchups: {len(loaded) ** 2}")

    results = {}
    for i, s_entry in enumerate(loaded):
        for j, h_entry in enumerate(loaded):
            matchup = f"{s_entry['name']} (S) vs {h_entry['name']} (H)"
            result = evaluate_matchup(
                s_entry['seeker_agent'],
                h_entry['hider_agent'],
                num_episodes=args.episodes,
                env_config=env_config,
            )
            results[matchup] = result

            swr = result['seeker_win_rate']
            print(f"  [{i*len(loaded)+j+1}/{len(loaded)**2}] "
                  f"{s_entry['name']} vs {h_entry['name']}: "
                  f"seeker WR={swr:.1%}")

    # Compute aggregate metrics per policy
    print("\n" + "=" * 60)
    print("ROBUSTNESS SUMMARY")
    print("=" * 60)

    for entry in loaded:
        name = entry['name']
        # Seeker robustness: avg win rate against all hiders
        seeker_wrs = [
            results[f"{name} (S) vs {h['name']} (H)"]['seeker_win_rate']
            for h in loaded
        ]
        # Hider robustness: avg survival rate against all seekers
        hider_srs = [
            1.0 - results[f"{s['name']} (S) vs {name} (H)"]['seeker_win_rate']
            for s in loaded
        ]
        entry['seeker_robustness'] = float(np.mean(seeker_wrs))
        entry['hider_robustness'] = float(np.mean(hider_srs))

        print(f"  {name}:")
        print(f"    Seeker avg WR: {entry['seeker_robustness']:.1%}  "
              f"Hider avg survival: {entry['hider_robustness']:.1%}")

    # Save results
    output_path = args.output or os.path.join(args.sweep_dir, "cross_evaluation.json")
    output = {
        'episodes_per_matchup': args.episodes,
        'layout': args.layout,
        'policies': [
            {k: v for k, v in e.items() if k not in ('seeker_agent', 'hider_agent')}
            for e in loaded
        ],
        'matchups': results,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
