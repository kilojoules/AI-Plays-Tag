#!/usr/bin/env python3
"""
Run checkpoint comparison with available data.

Combines checkpoints from multiple SCRO runs for best coverage.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).parent.parent))

from trainer.tag_env import SingleTagEnv, TagEnvConfig
from trainer.ppo import PPOAgent, PPOConfig
from experiments.scro_core import Agent as SCROAgent


def load_vanilla_checkpoint(path: str, obs_dim: int, act_dim: int) -> PPOAgent:
    """Load a vanilla self-play trained policy."""
    cfg = PPOConfig(obs_dim=obs_dim, act_dim=act_dim)
    policy = PPOAgent(cfg)
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    policy.pi.load_state_dict(checkpoint["pi"])
    policy.vf.load_state_dict(checkpoint["vf"])
    return policy


def load_scro_checkpoint(path: str, obs_dim: int, act_dim: int,
                         hidden_sizes: List[int] = [128, 128]) -> SCROAgent:
    """Load SCRO policy."""
    policy = SCROAgent(obs_dim, act_dim, hidden_sizes)
    state_dict = torch.load(path, map_location='cpu', weights_only=True)
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def get_action(policy, obs: np.ndarray, is_scro: bool) -> np.ndarray:
    """Get action from policy, handling different policy types."""
    with torch.no_grad():
        if is_scro:
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            action = policy.act(obs_t)
            return torch.tanh(action).squeeze().numpy()
        else:
            action, _, _ = policy.act(obs)
            return action.squeeze()


def evaluate_matchup(
    seeker_policy,
    hider_policy,
    seeker_is_scro: bool,
    hider_is_scro: bool,
    num_episodes: int = 50,
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
            seeker_action = get_action(seeker_policy, obs['seeker'], seeker_is_scro)
            hider_action = get_action(hider_policy, obs['hider'], hider_is_scro)

            actions = {'seeker': seeker_action, 'hider': hider_action}
            obs, rewards, done, info = env.step(actions)
            step += 1

        total_steps += step
        if info.get('tagged', False):
            seeker_wins += 1

    return {
        'seeker_win_rate': seeker_wins / num_episodes,
        'avg_episode_length': total_steps / num_episodes,
    }


def compute_crossplay_matrix(
    seeker_checkpoints: List[Tuple[str, any, bool]],
    hider_checkpoints: List[Tuple[str, any, bool]],
    env_config: TagEnvConfig,
    episodes: int = 50,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Compute full cross-play matrix."""
    n_seekers = len(seeker_checkpoints)
    n_hiders = len(hider_checkpoints)

    win_matrix = np.zeros((n_seekers, n_hiders))

    seeker_labels = [s[0] for s in seeker_checkpoints]
    hider_labels = [h[0] for h in hider_checkpoints]

    total = n_seekers * n_hiders
    done = 0

    for i, (s_label, s_policy, s_scro) in enumerate(seeker_checkpoints):
        for j, (h_label, h_policy, h_scro) in enumerate(hider_checkpoints):
            result = evaluate_matchup(
                s_policy, h_policy,
                seeker_is_scro=s_scro,
                hider_is_scro=h_scro,
                num_episodes=episodes,
                env_config=env_config,
            )
            win_matrix[i, j] = result['seeker_win_rate']
            done += 1
            print(f"  [{done}/{total}] {s_label} vs {h_label}: {result['seeker_win_rate']:.1%}")

    return win_matrix, seeker_labels, hider_labels


def plot_heatmap(ax, matrix, row_labels, col_labels, title, cmap='RdYlGn',
                 vmin=0, vmax=100):
    """Plot a heatmap with annotations."""
    im = ax.imshow(matrix * 100, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title(title, fontsize=11)

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = matrix[i, j] * 100
            color = 'white' if val < 30 or val > 70 else 'black'
            ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                   color=color, fontsize=8)

    return im


def main():
    # Paths
    vanilla_dir = Path("experiments/results/obstacles/comparison_20260124_212516/vanilla_selfplay/seed_42/20260124_212519")
    scro_original = Path("experiments/results/obstacles/comparison_20260124_212516/scro/seed_42/20260124_225713")
    scro_checkpointed = Path("experiments/results/scro_checkpointed/20260125_144113")

    env_config = TagEnvConfig(layout="four_corners")
    env = SingleTagEnv(config=env_config)
    obs_dim = env.obs_dim
    act_dim = env.act_dim

    print("Loading vanilla checkpoints...")
    vanilla_seekers = []
    vanilla_hiders = []

    ckpt_dir = vanilla_dir / "checkpoints"
    for num in [50, 100, 150, 200, 250]:
        s_path = ckpt_dir / f"seeker_{num:05d}.pt"
        h_path = ckpt_dir / f"hider_{num:05d}.pt"
        if s_path.exists():
            vanilla_seekers.append((str(num), load_vanilla_checkpoint(str(s_path), obs_dim, act_dim), False))
            print(f"  Loaded seeker {num}")
        if h_path.exists():
            vanilla_hiders.append((str(num), load_vanilla_checkpoint(str(h_path), obs_dim, act_dim), False))
            print(f"  Loaded hider {num}")

    # Final
    s_final = vanilla_dir / "policy_seeker_final.pt"
    h_final = vanilla_dir / "policy_hider_final.pt"
    if s_final.exists():
        vanilla_seekers.append(("final", load_vanilla_checkpoint(str(s_final), obs_dim, act_dim), False))
    if h_final.exists():
        vanilla_hiders.append(("final", load_vanilla_checkpoint(str(h_final), obs_dim, act_dim), False))

    print(f"\nLoading SCRO checkpoints...")
    scro_seekers = []
    scro_hiders = []

    # Gen 5 from checkpointed run
    s_g5 = scro_checkpointed / "checkpoints" / "protagonist_gen005.pt"
    h_g5 = scro_checkpointed / "checkpoints" / "antagonist_gen005.pt"
    if s_g5.exists():
        scro_seekers.append(("g5", load_scro_checkpoint(str(s_g5), obs_dim, act_dim), True))
        print(f"  Loaded SCRO seeker g5")
    if h_g5.exists():
        scro_hiders.append(("g5", load_scro_checkpoint(str(h_g5), obs_dim, act_dim), True))
        print(f"  Loaded SCRO hider g5")

    # Gen 10 from original run
    s_g10 = scro_original / "checkpoints" / "protagonist_gen010.pt"
    h_g10 = scro_original / "checkpoints" / "antagonist_gen010.pt"
    if s_g10.exists():
        scro_seekers.append(("g10", load_scro_checkpoint(str(s_g10), obs_dim, act_dim), True))
        print(f"  Loaded SCRO seeker g10")
    if h_g10.exists():
        scro_hiders.append(("g10", load_scro_checkpoint(str(h_g10), obs_dim, act_dim), True))
        print(f"  Loaded SCRO hider g10")

    # Final from original run
    s_final = scro_original / "best_protagonist.pt"
    h_final = scro_original / "best_antagonist.pt"
    if s_final.exists():
        scro_seekers.append(("final", load_scro_checkpoint(str(s_final), obs_dim, act_dim), True))
        print(f"  Loaded SCRO seeker final")
    if h_final.exists():
        scro_hiders.append(("final", load_scro_checkpoint(str(h_final), obs_dim, act_dim), True))
        print(f"  Loaded SCRO hider final")

    print(f"\nTotal: {len(vanilla_seekers)} vanilla seekers, {len(vanilla_hiders)} vanilla hiders")
    print(f"       {len(scro_seekers)} SCRO seekers, {len(scro_hiders)} SCRO hiders")

    episodes = 50

    # 1. Vanilla internal
    print("\n" + "="*60)
    print("Computing Vanilla internal cross-play...")
    v_win, v_s_labels, v_h_labels = compute_crossplay_matrix(
        vanilla_seekers, vanilla_hiders, env_config, episodes
    )

    # 2. SCRO internal
    print("\n" + "="*60)
    print("Computing SCRO internal cross-play...")
    s_win, s_s_labels, s_h_labels = compute_crossplay_matrix(
        scro_seekers, scro_hiders, env_config, episodes
    )

    # 3. Cross-method: Vanilla seekers vs SCRO hiders
    print("\n" + "="*60)
    print("Computing Vanilla Seekers vs SCRO Hiders...")
    vs_win, vs_s_labels, vs_h_labels = compute_crossplay_matrix(
        vanilla_seekers, scro_hiders, env_config, episodes
    )

    # 4. Cross-method: SCRO seekers vs Vanilla hiders
    print("\n" + "="*60)
    print("Computing SCRO Seekers vs Vanilla Hiders...")
    sv_win, sv_s_labels, sv_h_labels = compute_crossplay_matrix(
        scro_seekers, vanilla_hiders, env_config, episodes
    )

    # Create visualization
    print("\n" + "="*60)
    print("Creating visualization...")

    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)

    # Vanilla internal
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = plot_heatmap(ax1, v_win, v_s_labels, v_h_labels,
                      'Vanilla Self-Play Training Dynamics\nSeeker Win Rate (%)')
    ax1.set_xlabel('Vanilla Hider Checkpoint (PPO update #)')
    ax1.set_ylabel('Vanilla Seeker Checkpoint')

    # SCRO internal
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = plot_heatmap(ax2, s_win, s_s_labels, s_h_labels,
                      'SCRO Training Dynamics\nSeeker Win Rate (%)')
    ax2.set_xlabel('SCRO Hider Checkpoint (generation)')
    ax2.set_ylabel('SCRO Seeker Checkpoint')

    # Vanilla seeker vs SCRO hider
    ax3 = fig.add_subplot(gs[1, 0])
    im3 = plot_heatmap(ax3, vs_win, vs_s_labels, vs_h_labels,
                      'Cross-Method: Vanilla Seekers vs SCRO Hiders\nSeeker Win Rate (%)')
    ax3.set_xlabel('SCRO Hider Checkpoint')
    ax3.set_ylabel('Vanilla Seeker Checkpoint')

    # SCRO seeker vs Vanilla hider
    ax4 = fig.add_subplot(gs[1, 1])
    im4 = plot_heatmap(ax4, sv_win, sv_s_labels, sv_h_labels,
                      'Cross-Method: SCRO Seekers vs Vanilla Hiders\nSeeker Win Rate (%)')
    ax4.set_xlabel('Vanilla Hider Checkpoint')
    ax4.set_ylabel('SCRO Seeker Checkpoint')

    plt.suptitle('Checkpoint Cross-Play Comparison: Vanilla Self-Play vs SCRO\nTraining Dynamics Analysis',
                 fontsize=14, fontweight='bold')

    output_path = "experiments/results/checkpoint_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {output_path}")
    plt.close()

    # Save results
    results = {
        'vanilla_internal': {'win_matrix': v_win.tolist(), 'seeker_labels': v_s_labels, 'hider_labels': v_h_labels},
        'scro_internal': {'win_matrix': s_win.tolist(), 'seeker_labels': s_s_labels, 'hider_labels': s_h_labels},
        'vanilla_vs_scro': {'win_matrix': vs_win.tolist(), 'seeker_labels': vs_s_labels, 'hider_labels': vs_h_labels},
        'scro_vs_vanilla': {'win_matrix': sv_win.tolist(), 'seeker_labels': sv_s_labels, 'hider_labels': sv_h_labels},
    }

    json_path = "experiments/results/checkpoint_comparison.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {json_path}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    print("\nVanilla diagonal (contemporary matchups):")
    for i, label in enumerate(v_s_labels):
        if i < len(v_h_labels):
            print(f"  {label}: {v_win[i, i]:.1%}")

    print("\nSCRO diagonal (contemporary matchups):")
    for i, label in enumerate(s_s_labels):
        if i < len(s_h_labels):
            print(f"  {label}: {s_win[i, i]:.1%}")

    print("\nCross-method (final vs final):")
    print(f"  Vanilla final seeker vs SCRO final hider: {vs_win[-1, -1]:.1%}")
    print(f"  SCRO final seeker vs Vanilla final hider: {sv_win[-1, -1]:.1%}")

    print("\nKey insights:")
    # Vanilla seeker avg across SCRO hiders
    print(f"  Vanilla seekers avg vs SCRO hiders: {np.mean(vs_win):.1%}")
    # SCRO seeker avg across Vanilla hiders
    print(f"  SCRO seekers avg vs Vanilla hiders: {np.mean(sv_win):.1%}")


if __name__ == "__main__":
    main()
