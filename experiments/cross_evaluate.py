#!/usr/bin/env python3
"""
Cross-evaluate agents trained with different approaches.

Runs a round-robin evaluation matrix:
- Vanilla seeker vs Vanilla hider
- Vanilla seeker vs SCRO hider
- SCRO seeker vs Vanilla hider
- SCRO seeker vs SCRO hider
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from trainer.tag_env import SingleTagEnv, TagEnvConfig
from trainer.ppo import PPOAgent, PPOConfig
from experiments.scro_core import Agent as SCROAgent


def load_vanilla_policy(path: str, obs_dim: int, act_dim: int) -> PPOAgent:
    """Load a vanilla self-play trained policy."""
    cfg = PPOConfig(obs_dim=obs_dim, act_dim=act_dim)
    policy = PPOAgent(cfg)
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    # Checkpoint format is {"pi": state_dict, "vf": state_dict}
    policy.pi.load_state_dict(checkpoint["pi"])
    policy.vf.load_state_dict(checkpoint["vf"])
    return policy


def load_scro_policy(path: str, obs_dim: int, act_dim: int, hidden_sizes: List[int] = [128, 128]) -> SCROAgent:
    """Load SCRO policy (uses different hidden sizes)."""
    policy = SCROAgent(obs_dim, act_dim, hidden_sizes)
    state_dict = torch.load(path, map_location='cpu', weights_only=True)
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def get_action(policy, obs: np.ndarray, is_scro: bool) -> np.ndarray:
    """Get action from policy, handling different policy types."""
    with torch.no_grad():
        if is_scro:
            # SCRO agent expects torch tensor, returns action directly
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            action = policy.act(obs_t)
            return torch.tanh(action).squeeze().numpy()  # Apply tanh for bounded actions
        else:
            # Vanilla PPOAgent expects numpy, returns (action, log_prob, value)
            action, _, _ = policy.act(obs)
            return action.squeeze()


def evaluate_matchup(
    seeker_policy,
    hider_policy,
    seeker_is_scro: bool,
    hider_is_scro: bool,
    num_episodes: int = 100,
    max_steps: int = 200,
    seed: int = 42,
    env_config: Optional[TagEnvConfig] = None,
) -> Dict[str, float]:
    """Evaluate a seeker vs hider matchup."""
    env = SingleTagEnv(config=env_config)
    np.random.seed(seed)

    seeker_wins = 0
    hider_wins = 0
    total_steps = 0
    seeker_rewards = []
    hider_rewards = []

    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        step = 0
        ep_seeker_reward = 0
        ep_hider_reward = 0

        while not done and step < max_steps:
            seeker_action = get_action(seeker_policy, obs['seeker'], seeker_is_scro)
            hider_action = get_action(hider_policy, obs['hider'], hider_is_scro)

            actions = {'seeker': seeker_action, 'hider': hider_action}
            obs, rewards, done, info = env.step(actions)

            ep_seeker_reward += rewards['seeker']
            ep_hider_reward += rewards['hider']
            step += 1

        total_steps += step
        seeker_rewards.append(ep_seeker_reward)
        hider_rewards.append(ep_hider_reward)

        if info.get('tagged', False):
            seeker_wins += 1
        else:
            hider_wins += 1

    return {
        'seeker_win_rate': seeker_wins / num_episodes,
        'hider_win_rate': hider_wins / num_episodes,
        'avg_episode_length': total_steps / num_episodes,
        'seeker_reward_mean': np.mean(seeker_rewards),
        'hider_reward_mean': np.mean(hider_rewards),
    }


def find_best_checkpoint(run_dir: str, role: str) -> Optional[str]:
    """Find the best/latest checkpoint for a role."""
    run_path = Path(run_dir)

    # Check for final policy first
    final_path = run_path / f"policy_{role}_final.pt"
    if final_path.exists():
        return str(final_path)

    # Check checkpoints directory
    checkpoint_dir = run_path / "checkpoints"
    if checkpoint_dir.exists():
        checkpoints = sorted(checkpoint_dir.glob(f"{role}_*.pt"))
        if checkpoints:
            return str(checkpoints[-1])  # Latest checkpoint

    return None


def main():
    parser = argparse.ArgumentParser(description="Cross-evaluate trained agents")
    parser.add_argument("experiment_dir", type=str,
                        help="Path to comparison experiment directory")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of evaluation episodes per matchup")
    parser.add_argument("--seed", type=int, default=42,
                        help="Which seed's policies to use")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for results (JSON)")
    parser.add_argument("--layout", type=str, default="empty",
                        choices=["empty", "four_corners", "central_cross"],
                        help="Arena layout (default: empty)")

    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir)

    # Create environment config with layout
    env_config = TagEnvConfig(layout=args.layout)

    # Create sample env to get dimensions
    env = SingleTagEnv(config=env_config)
    obs_dim = env.obs_dim
    act_dim = env.act_dim

    # Find policy paths
    vanilla_dir = list(exp_dir.glob(f"vanilla_selfplay/seed_{args.seed}/*"))[0]
    scro_dir = list(exp_dir.glob(f"scro/seed_{args.seed}/*"))[0]

    print(f"Loading policies...")
    print(f"  Vanilla: {vanilla_dir}")
    print(f"  SCRO: {scro_dir}")

    # Load vanilla policies (use checkpoints since training crashed)
    vanilla_seeker_path = find_best_checkpoint(vanilla_dir, "seeker")
    vanilla_hider_path = find_best_checkpoint(vanilla_dir, "hider")

    if not vanilla_seeker_path or not vanilla_hider_path:
        print(f"Could not find vanilla policies!")
        return

    print(f"  Vanilla seeker: {vanilla_seeker_path}")
    print(f"  Vanilla hider: {vanilla_hider_path}")

    vanilla_seeker = load_vanilla_policy(vanilla_seeker_path, obs_dim, act_dim)
    vanilla_hider = load_vanilla_policy(vanilla_hider_path, obs_dim, act_dim)

    # Load SCRO policies
    scro_seeker_path = scro_dir / "best_protagonist.pt"
    scro_hider_path = scro_dir / "best_antagonist.pt"

    print(f"  SCRO seeker: {scro_seeker_path}")
    print(f"  SCRO hider: {scro_hider_path}")

    scro_seeker = load_scro_policy(str(scro_seeker_path), obs_dim, act_dim)
    scro_hider = load_scro_policy(str(scro_hider_path), obs_dim, act_dim)

    # Define matchups: (seeker_name, hider_name, seeker_policy, hider_policy, seeker_is_scro, hider_is_scro)
    matchups = [
        ("Vanilla Seeker", "Vanilla Hider", vanilla_seeker, vanilla_hider, False, False),
        ("Vanilla Seeker", "SCRO Hider", vanilla_seeker, scro_hider, False, True),
        ("SCRO Seeker", "Vanilla Hider", scro_seeker, vanilla_hider, True, False),
        ("SCRO Seeker", "SCRO Hider", scro_seeker, scro_hider, True, True),
    ]

    print(f"\nRunning cross-evaluation ({args.episodes} episodes each)...\n")
    print("=" * 70)

    results = {}

    for seeker_name, hider_name, seeker_policy, hider_policy, seeker_is_scro, hider_is_scro in matchups:
        matchup_name = f"{seeker_name} vs {hider_name}"
        print(f"{matchup_name}...")

        result = evaluate_matchup(
            seeker_policy, hider_policy,
            seeker_is_scro=seeker_is_scro,
            hider_is_scro=hider_is_scro,
            num_episodes=args.episodes,
            seed=args.seed,
            env_config=env_config,
        )

        results[matchup_name] = result

        print(f"  Seeker Win Rate: {result['seeker_win_rate']:.1%}")
        print(f"  Avg Episode Length: {result['avg_episode_length']:.1f} steps")
        print()

    # Print summary matrix
    print("=" * 70)
    print("\nCROSS-PLAY WIN RATE MATRIX (Seeker Win %)")
    print("-" * 50)
    print(f"{'':20} | {'Vanilla Hider':>14} | {'SCRO Hider':>14}")
    print("-" * 50)

    vanilla_vs_vanilla = results["Vanilla Seeker vs Vanilla Hider"]["seeker_win_rate"]
    vanilla_vs_scro = results["Vanilla Seeker vs SCRO Hider"]["seeker_win_rate"]
    scro_vs_vanilla = results["SCRO Seeker vs Vanilla Hider"]["seeker_win_rate"]
    scro_vs_scro = results["SCRO Seeker vs SCRO Hider"]["seeker_win_rate"]

    print(f"{'Vanilla Seeker':20} | {vanilla_vs_vanilla:>13.1%} | {vanilla_vs_scro:>13.1%}")
    print(f"{'SCRO Seeker':20} | {scro_vs_vanilla:>13.1%} | {scro_vs_scro:>13.1%}")
    print("-" * 50)

    # Compute aggregate stats
    vanilla_seeker_avg = (vanilla_vs_vanilla + vanilla_vs_scro) / 2
    scro_seeker_avg = (scro_vs_vanilla + scro_vs_scro) / 2
    vanilla_hider_avg = 1 - (vanilla_vs_vanilla + scro_vs_vanilla) / 2
    scro_hider_avg = 1 - (vanilla_vs_scro + scro_vs_scro) / 2

    print(f"\nAggregate Performance:")
    print(f"  Vanilla Seeker avg win rate: {vanilla_seeker_avg:.1%}")
    print(f"  SCRO Seeker avg win rate: {scro_seeker_avg:.1%}")
    print(f"  Vanilla Hider avg survival rate: {vanilla_hider_avg:.1%}")
    print(f"  SCRO Hider avg survival rate: {scro_hider_avg:.1%}")

    # Save results
    output_path = args.output or str(exp_dir / "cross_evaluation.json")
    with open(output_path, "w") as f:
        json.dump({
            "seed": args.seed,
            "episodes_per_matchup": args.episodes,
            "matchups": results,
            "summary": {
                "vanilla_seeker_avg_win_rate": vanilla_seeker_avg,
                "scro_seeker_avg_win_rate": scro_seeker_avg,
                "vanilla_hider_avg_survival_rate": vanilla_hider_avg,
                "scro_hider_avg_survival_rate": scro_hider_avg,
            }
        }, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
