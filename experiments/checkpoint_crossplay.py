#!/usr/bin/env python3
"""
Checkpoint cross-play analysis.

Evaluates seeker_i vs hider_j for all checkpoint combinations
to understand co-evolution dynamics during training.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from trainer.tag_env import SingleTagEnv, TagEnvConfig
from trainer.ppo import PPOAgent, PPOConfig


def load_checkpoint(path: str, obs_dim: int, act_dim: int) -> PPOAgent:
    """Load a policy from checkpoint."""
    cfg = PPOConfig(obs_dim=obs_dim, act_dim=act_dim)
    policy = PPOAgent(cfg)
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    policy.pi.load_state_dict(checkpoint["pi"])
    policy.vf.load_state_dict(checkpoint["vf"])
    return policy


def evaluate_matchup(
    seeker_policy: PPOAgent,
    hider_policy: PPOAgent,
    num_episodes: int = 100,
    max_steps: int = 200,
    seed: int = 42,
    env_config: Optional[TagEnvConfig] = None,
) -> Dict[str, float]:
    """Evaluate a seeker vs hider matchup."""
    env = SingleTagEnv(config=env_config)
    np.random.seed(seed)

    seeker_wins = 0
    total_steps = 0

    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        step = 0

        while not done and step < max_steps:
            with torch.no_grad():
                seeker_action, _, _ = seeker_policy.act(obs['seeker'])
                hider_action, _, _ = hider_policy.act(obs['hider'])

            actions = {'seeker': seeker_action.squeeze(), 'hider': hider_action.squeeze()}
            obs, rewards, done, info = env.step(actions)
            step += 1

        total_steps += step
        if info.get('tagged', False):
            seeker_wins += 1

    return {
        'seeker_win_rate': seeker_wins / num_episodes,
        'avg_episode_length': total_steps / num_episodes,
    }


def find_checkpoints(run_dir: Path) -> Dict[str, List[Path]]:
    """Find all seeker and hider checkpoints."""
    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return {'seeker': [], 'hider': []}

    seeker_ckpts = sorted(checkpoint_dir.glob("seeker_*.pt"))
    hider_ckpts = sorted(checkpoint_dir.glob("hider_*.pt"))

    # Also check for final policies
    final_seeker = run_dir / "policy_seeker_final.pt"
    final_hider = run_dir / "policy_hider_final.pt"

    if final_seeker.exists():
        seeker_ckpts.append(final_seeker)
    if final_hider.exists():
        hider_ckpts.append(final_hider)

    return {'seeker': seeker_ckpts, 'hider': hider_ckpts}


def extract_update_num(path: Path) -> int:
    """Extract update number from checkpoint filename."""
    name = path.stem
    if 'final' in name:
        return 999999  # Final goes at the end
    # Format: seeker_00050 or hider_00100
    parts = name.split('_')
    return int(parts[-1])


def main():
    parser = argparse.ArgumentParser(description="Checkpoint cross-play analysis")
    parser.add_argument("run_dir", type=str,
                        help="Path to training run directory with checkpoints")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Episodes per matchup (default: 50)")
    parser.add_argument("--layout", type=str, default="four_corners",
                        choices=["empty", "four_corners", "central_cross"],
                        help="Arena layout")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for results")

    args = parser.parse_args()
    run_dir = Path(args.run_dir)

    # Create environment config
    env_config = TagEnvConfig(layout=args.layout)
    env = SingleTagEnv(config=env_config)
    obs_dim = env.obs_dim
    act_dim = env.act_dim

    # Find checkpoints
    checkpoints = find_checkpoints(run_dir)
    seeker_ckpts = checkpoints['seeker']
    hider_ckpts = checkpoints['hider']

    if not seeker_ckpts or not hider_ckpts:
        print(f"No checkpoints found in {run_dir}")
        return

    print(f"Found {len(seeker_ckpts)} seeker checkpoints, {len(hider_ckpts)} hider checkpoints")

    # Get update numbers for labels
    seeker_updates = [extract_update_num(p) for p in seeker_ckpts]
    hider_updates = [extract_update_num(p) for p in hider_ckpts]

    # Replace 999999 with "final" for display
    seeker_labels = [str(u) if u < 999999 else "final" for u in seeker_updates]
    hider_labels = [str(u) if u < 999999 else "final" for u in hider_updates]

    print(f"Seeker checkpoints: {seeker_labels}")
    print(f"Hider checkpoints: {hider_labels}")

    # Load all policies
    print("\nLoading policies...")
    seeker_policies = []
    for ckpt in seeker_ckpts:
        policy = load_checkpoint(str(ckpt), obs_dim, act_dim)
        seeker_policies.append(policy)
        print(f"  Loaded seeker: {ckpt.name}")

    hider_policies = []
    for ckpt in hider_ckpts:
        policy = load_checkpoint(str(ckpt), obs_dim, act_dim)
        hider_policies.append(policy)
        print(f"  Loaded hider: {ckpt.name}")

    # Evaluate all matchups
    n_seekers = len(seeker_policies)
    n_hiders = len(hider_policies)
    win_matrix = np.zeros((n_seekers, n_hiders))
    length_matrix = np.zeros((n_seekers, n_hiders))

    print(f"\nEvaluating {n_seekers * n_hiders} matchups ({args.episodes} episodes each)...")

    for i, seeker in enumerate(seeker_policies):
        for j, hider in enumerate(hider_policies):
            result = evaluate_matchup(
                seeker, hider,
                num_episodes=args.episodes,
                env_config=env_config,
            )
            win_matrix[i, j] = result['seeker_win_rate']
            length_matrix[i, j] = result['avg_episode_length']
            print(f"  Seeker {seeker_labels[i]} vs Hider {hider_labels[j]}: "
                  f"{result['seeker_win_rate']:.1%} seeker WR, "
                  f"{result['avg_episode_length']:.0f} steps")

    # Print summary matrix
    print("\n" + "=" * 70)
    print("CHECKPOINT CROSS-PLAY MATRIX (Seeker Win %)")
    print("=" * 70)

    # Header row
    header = "Seeker \\ Hider |"
    for label in hider_labels:
        header += f" {label:>8} |"
    print(header)
    print("-" * len(header))

    # Data rows
    for i, s_label in enumerate(seeker_labels):
        row = f"{s_label:>14} |"
        for j in range(n_hiders):
            row += f" {win_matrix[i, j]:>7.1%} |"
        print(row)

    print("-" * len(header))

    # Analyze training dynamics
    print("\n" + "=" * 70)
    print("TRAINING DYNAMICS ANALYSIS")
    print("=" * 70)

    # Diagonal: contemporary matchups (seeker_i vs hider_i)
    min_len = min(n_seekers, n_hiders)
    diagonal = [win_matrix[i, i] for i in range(min_len)]
    print(f"\nContemporary matchups (diagonal - seeker_i vs hider_i):")
    for i in range(min_len):
        print(f"  Update {seeker_labels[i]}: {diagonal[i]:.1%} seeker WR")

    # Row averages: how each seeker performs against all hiders
    print(f"\nSeeker performance across all hiders:")
    for i in range(n_seekers):
        avg = np.mean(win_matrix[i, :])
        print(f"  Seeker {seeker_labels[i]}: {avg:.1%} avg WR")

    # Column averages: how each hider survives against all seekers
    print(f"\nHider survival against all seekers:")
    for j in range(n_hiders):
        avg = 1 - np.mean(win_matrix[:, j])
        print(f"  Hider {hider_labels[j]}: {avg:.1%} avg survival")

    # Check for forgotten strategies
    print(f"\nForgotten strategies check:")
    print("  (Does later seeker struggle against earlier hiders?)")
    if n_seekers >= 2 and n_hiders >= 2:
        latest_seeker = n_seekers - 1
        for j in range(n_hiders - 1):  # All but final hider
            wr = win_matrix[latest_seeker, j]
            contemporary_wr = win_matrix[j, j] if j < n_seekers else None
            if contemporary_wr and wr < contemporary_wr:
                print(f"  ! Seeker {seeker_labels[latest_seeker]} vs Hider {hider_labels[j]}: "
                      f"{wr:.1%} < {contemporary_wr:.1%} (contemporary)")

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Win rate heatmap
    ax1 = axes[0]
    im1 = ax1.imshow(win_matrix * 100, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    ax1.set_xticks(range(n_hiders))
    ax1.set_xticklabels(hider_labels, rotation=45, ha='right')
    ax1.set_yticks(range(n_seekers))
    ax1.set_yticklabels(seeker_labels)
    ax1.set_xlabel('Hider Checkpoint')
    ax1.set_ylabel('Seeker Checkpoint')
    ax1.set_title('Seeker Win Rate (%)')

    # Add text annotations
    for i in range(n_seekers):
        for j in range(n_hiders):
            text = ax1.text(j, i, f'{win_matrix[i, j]*100:.0f}',
                           ha='center', va='center', color='black', fontsize=9)

    plt.colorbar(im1, ax=ax1)

    # Episode length heatmap
    ax2 = axes[1]
    im2 = ax2.imshow(length_matrix, cmap='viridis', aspect='auto')
    ax2.set_xticks(range(n_hiders))
    ax2.set_xticklabels(hider_labels, rotation=45, ha='right')
    ax2.set_yticks(range(n_seekers))
    ax2.set_yticklabels(seeker_labels)
    ax2.set_xlabel('Hider Checkpoint')
    ax2.set_ylabel('Seeker Checkpoint')
    ax2.set_title('Avg Episode Length (steps)')

    for i in range(n_seekers):
        for j in range(n_hiders):
            text = ax2.text(j, i, f'{length_matrix[i, j]:.0f}',
                           ha='center', va='center', color='white', fontsize=9)

    plt.colorbar(im2, ax=ax2)

    plt.suptitle('Checkpoint Cross-Play Analysis: Training Dynamics', fontsize=14)
    plt.tight_layout()

    # Save figure
    fig_path = run_dir / "checkpoint_crossplay.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {fig_path}")
    plt.close()

    # Save results to JSON
    output_path = args.output or str(run_dir / "checkpoint_crossplay.json")
    results = {
        'seeker_checkpoints': seeker_labels,
        'hider_checkpoints': hider_labels,
        'win_matrix': win_matrix.tolist(),
        'length_matrix': length_matrix.tolist(),
        'episodes_per_matchup': args.episodes,
        'layout': args.layout,
    }
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
