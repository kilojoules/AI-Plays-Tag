#!/usr/bin/env python3
"""
Evaluate pre-trained continuation experiments.

Compare:
1. Original vanilla trained agents
2. Pre-trained vanilla continuation
3. Pre-trained SCRO
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from trainer.tag_env import SingleTagEnv, TagEnvConfig
from trainer.ppo import PPOAgent, PPOConfig
from experiments.scro_core import Agent as SCROAgent


def load_vanilla(path: str, obs_dim: int, act_dim: int) -> PPOAgent:
    """Load vanilla PPO policy."""
    cfg = PPOConfig(obs_dim=obs_dim, act_dim=act_dim)
    policy = PPOAgent(cfg)
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    policy.pi.load_state_dict(ckpt["pi"])
    policy.vf.load_state_dict(ckpt["vf"])
    return policy


def load_scro(path: str, obs_dim: int, act_dim: int) -> SCROAgent:
    """Load SCRO policy."""
    policy = SCROAgent(obs_dim, act_dim, [128, 128])
    policy.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    policy.eval()
    return policy


def get_action(policy, obs: np.ndarray, is_scro: bool) -> np.ndarray:
    """Get action from policy."""
    with torch.no_grad():
        if is_scro:
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            action = policy.act(obs_t)
            return torch.tanh(action).squeeze().numpy()
        else:
            action, _, _ = policy.act(obs)
            return action.squeeze()


def evaluate(seeker, hider, seeker_is_scro: bool, hider_is_scro: bool,
             env_config: TagEnvConfig, episodes: int = 100) -> Dict[str, float]:
    """Evaluate matchup."""
    env = SingleTagEnv(config=env_config)
    np.random.seed(42)

    seeker_wins = 0
    total_steps = 0

    for _ in range(episodes):
        obs = env.reset()
        done = False
        step = 0

        while not done and step < 200:
            s_act = get_action(seeker, obs['seeker'], seeker_is_scro)
            h_act = get_action(hider, obs['hider'], hider_is_scro)
            obs, _, done, info = env.step({'seeker': s_act, 'hider': h_act})
            step += 1

        total_steps += step
        if info.get('tagged', False):
            seeker_wins += 1

    return {
        'win_rate': seeker_wins / episodes,
        'avg_length': total_steps / episodes,
    }


def main():
    # Paths
    original_seeker = "experiments/results/obstacles/comparison_20260124_212516/vanilla_selfplay/seed_42/20260124_212519/policy_seeker_final.pt"
    original_hider = "experiments/results/obstacles/comparison_20260124_212516/vanilla_selfplay/seed_42/20260124_212519/policy_hider_final.pt"

    # Find pretrained vanilla
    pretrained_vanilla_dir = Path("experiments/results/pretrained_vanilla")
    latest_vanilla = sorted(pretrained_vanilla_dir.glob("*"))[-1]
    pretrained_v_seeker = latest_vanilla / "policy_seeker_final.pt"
    pretrained_v_hider = latest_vanilla / "policy_hider_final.pt"

    # Find pretrained SCRO
    pretrained_scro_dir = Path("experiments/results/pretrained_scro")
    latest_scro = sorted(pretrained_scro_dir.glob("*"))[-1]
    pretrained_s_seeker = latest_scro / "best_protagonist.pt"
    pretrained_s_hider = latest_scro / "best_antagonist.pt"

    print(f"Original vanilla: {original_seeker}")
    print(f"Pretrained vanilla: {pretrained_v_seeker}")
    print(f"Pretrained SCRO: {pretrained_s_seeker}")

    # Setup
    env_config = TagEnvConfig(layout="four_corners")
    env = SingleTagEnv(config=env_config)
    obs_dim = env.obs_dim
    act_dim = env.act_dim

    # Load policies
    print("\nLoading policies...")
    orig_seeker = load_vanilla(original_seeker, obs_dim, act_dim)
    orig_hider = load_vanilla(original_hider, obs_dim, act_dim)

    pt_v_seeker = load_vanilla(str(pretrained_v_seeker), obs_dim, act_dim)
    pt_v_hider = load_vanilla(str(pretrained_v_hider), obs_dim, act_dim)

    pt_s_seeker = load_scro(str(pretrained_s_seeker), obs_dim, act_dim)
    pt_s_hider = load_scro(str(pretrained_s_hider), obs_dim, act_dim)

    # Define all agents
    seekers = [
        ("Original Vanilla", orig_seeker, False),
        ("Continued Vanilla", pt_v_seeker, False),
        ("Pretrained SCRO", pt_s_seeker, True),
    ]

    hiders = [
        ("Original Vanilla", orig_hider, False),
        ("Continued Vanilla", pt_v_hider, False),
        ("Pretrained SCRO", pt_s_hider, True),
    ]

    # Evaluate all matchups
    episodes = 100
    results = np.zeros((len(seekers), len(hiders)))

    print(f"\nEvaluating {len(seekers) * len(hiders)} matchups ({episodes} episodes each)...")

    for i, (s_name, s_policy, s_scro) in enumerate(seekers):
        for j, (h_name, h_policy, h_scro) in enumerate(hiders):
            res = evaluate(s_policy, h_policy, s_scro, h_scro, env_config, episodes)
            results[i, j] = res['win_rate']
            print(f"  {s_name} vs {h_name}: {res['win_rate']:.1%} seeker WR")

    # Print summary matrix
    print("\n" + "=" * 70)
    print("CROSS-PLAY MATRIX (Seeker Win %)")
    print("=" * 70)

    seeker_labels = [s[0] for s in seekers]
    hider_labels = [h[0] for h in hiders]

    col_header = "Seeker \\ Hider"
    header = f"{col_header:20} |"
    for label in hider_labels:
        header += f" {label:>18} |"
    print(header)
    print("-" * len(header))

    for i, s_label in enumerate(seeker_labels):
        row = f"{s_label:20} |"
        for j in range(len(hiders)):
            row += f" {results[i, j]:>17.1%} |"
        print(row)

    print("-" * len(header))

    # Aggregate stats
    print("\nAggregate Performance:")
    for i, s_name in enumerate(seeker_labels):
        avg = np.mean(results[i, :])
        print(f"  {s_name} seeker avg: {avg:.1%}")

    print()
    for j, h_name in enumerate(hider_labels):
        avg = 1 - np.mean(results[:, j])
        print(f"  {h_name} hider avg survival: {avg:.1%}")

    # Key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # Original vs Original
    orig_vs_orig = results[0, 0]

    # Continued vs Continued
    cont_vs_cont = results[1, 1]

    # Pretrained SCRO vs Pretrained SCRO
    scro_vs_scro = results[2, 2]

    print(f"\nInternal balance (seeker WR against same-method hider):")
    print(f"  Original Vanilla: {orig_vs_orig:.1%}")
    print(f"  Continued Vanilla: {cont_vs_cont:.1%}")
    print(f"  Pretrained SCRO: {scro_vs_scro:.1%}")

    print(f"\nCross-method performance:")
    print(f"  Original seeker vs SCRO hider: {results[0, 2]:.1%}")
    print(f"  SCRO seeker vs Original hider: {results[2, 0]:.1%}")
    print(f"  Continued seeker vs SCRO hider: {results[1, 2]:.1%}")
    print(f"  SCRO seeker vs Continued hider: {results[2, 1]:.1%}")

    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(results * 100, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')

    ax.set_xticks(range(len(hiders)))
    ax.set_xticklabels(hider_labels, rotation=45, ha='right')
    ax.set_yticks(range(len(seekers)))
    ax.set_yticklabels(seeker_labels)
    ax.set_xlabel('Hider')
    ax.set_ylabel('Seeker')
    ax.set_title('Pre-trained Continuation Experiment\nSeeker Win Rate (%)')

    for i in range(len(seekers)):
        for j in range(len(hiders)):
            val = results[i, j] * 100
            color = 'white' if val < 30 or val > 70 else 'black'
            ax.text(j, i, f'{val:.0f}%', ha='center', va='center', color=color, fontsize=12)

    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    output_path = "experiments/results/pretrained_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")


if __name__ == "__main__":
    main()
