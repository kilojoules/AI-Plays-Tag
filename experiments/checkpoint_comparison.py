#!/usr/bin/env python3
"""
Comprehensive checkpoint cross-play comparison.

Creates:
1. Heatmap for vanilla self-play training dynamics
2. Heatmap for SCRO training dynamics
3. Cross-method comparison (SCRO checkpoints vs vanilla checkpoints)
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


def find_vanilla_checkpoints(run_dir: Path) -> List[Tuple[str, Path]]:
    """Find vanilla checkpoints with labels."""
    checkpoints = []
    ckpt_dir = run_dir / "checkpoints"

    if ckpt_dir.exists():
        for ckpt in sorted(ckpt_dir.glob("seeker_*.pt")):
            num = int(ckpt.stem.split('_')[1])
            checkpoints.append((str(num), ckpt))

    final = run_dir / "policy_seeker_final.pt"
    if final.exists():
        checkpoints.append(("final", final))

    return checkpoints


def find_scro_checkpoints(run_dir: Path) -> List[Tuple[str, Path]]:
    """Find SCRO checkpoints with labels."""
    checkpoints = []
    ckpt_dir = run_dir / "checkpoints"

    if ckpt_dir.exists():
        for ckpt in sorted(ckpt_dir.glob("protagonist_gen*.pt")):
            gen = int(ckpt.stem.split('gen')[1])
            checkpoints.append((f"g{gen}", ckpt))

    final = run_dir / "best_protagonist.pt"
    if final.exists():
        checkpoints.append(("final", final))

    return checkpoints


def compute_crossplay_matrix(
    seeker_checkpoints: List[Tuple[str, any, bool]],  # (label, policy, is_scro)
    hider_checkpoints: List[Tuple[str, any, bool]],
    env_config: TagEnvConfig,
    episodes: int = 50,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Compute full cross-play matrix."""
    n_seekers = len(seeker_checkpoints)
    n_hiders = len(hider_checkpoints)

    win_matrix = np.zeros((n_seekers, n_hiders))
    length_matrix = np.zeros((n_seekers, n_hiders))

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
            length_matrix[i, j] = result['avg_episode_length']
            done += 1
            print(f"  [{done}/{total}] {s_label} vs {h_label}: {result['seeker_win_rate']:.1%}")

    return win_matrix, length_matrix, seeker_labels, hider_labels


def plot_heatmap(ax, matrix, row_labels, col_labels, title, cmap='RdYlGn',
                 vmin=0, vmax=100, fmt='.0f', is_percentage=True):
    """Plot a heatmap with annotations."""
    im = ax.imshow(matrix * (100 if is_percentage else 1), cmap=cmap,
                   aspect='auto', vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_title(title, fontsize=10)

    # Annotations
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = matrix[i, j] * (100 if is_percentage else 1)
            color = 'white' if val < 30 or val > 70 else 'black'
            ax.text(j, i, f'{val:{fmt}}', ha='center', va='center',
                   color=color, fontsize=7)

    return im


def main():
    parser = argparse.ArgumentParser(description="Checkpoint comparison analysis")
    parser.add_argument("--vanilla-dir", type=str, required=True,
                        help="Path to vanilla self-play run directory")
    parser.add_argument("--scro-dir", type=str, required=True,
                        help="Path to SCRO run directory")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Episodes per matchup")
    parser.add_argument("--layout", type=str, default="four_corners",
                        help="Arena layout")
    parser.add_argument("--output", type=str, default="checkpoint_comparison.png",
                        help="Output figure path")

    args = parser.parse_args()

    vanilla_dir = Path(args.vanilla_dir)
    scro_dir = Path(args.scro_dir)

    # Setup environment
    env_config = TagEnvConfig(layout=args.layout)
    env = SingleTagEnv(config=env_config)
    obs_dim = env.obs_dim
    act_dim = env.act_dim

    print("Finding checkpoints...")

    # Find vanilla checkpoints
    vanilla_seeker_ckpts = find_vanilla_checkpoints(vanilla_dir)
    vanilla_hider_ckpts = []
    ckpt_dir = vanilla_dir / "checkpoints"
    if ckpt_dir.exists():
        for ckpt in sorted(ckpt_dir.glob("hider_*.pt")):
            num = int(ckpt.stem.split('_')[1])
            vanilla_hider_ckpts.append((str(num), ckpt))
    final_h = vanilla_dir / "policy_hider_final.pt"
    if final_h.exists():
        vanilla_hider_ckpts.append(("final", final_h))

    # Find SCRO checkpoints
    scro_seeker_ckpts = find_scro_checkpoints(scro_dir)
    scro_hider_ckpts = []
    ckpt_dir = scro_dir / "checkpoints"
    if ckpt_dir.exists():
        for ckpt in sorted(ckpt_dir.glob("antagonist_gen*.pt")):
            gen = int(ckpt.stem.split('gen')[1])
            scro_hider_ckpts.append((f"g{gen}", ckpt))
    final_h = scro_dir / "best_antagonist.pt"
    if final_h.exists():
        scro_hider_ckpts.append(("final", final_h))

    print(f"  Vanilla: {len(vanilla_seeker_ckpts)} seekers, {len(vanilla_hider_ckpts)} hiders")
    print(f"  SCRO: {len(scro_seeker_ckpts)} seekers, {len(scro_hider_ckpts)} hiders")

    # Load all policies
    print("\nLoading policies...")

    vanilla_seekers = [(label, load_vanilla_checkpoint(str(path), obs_dim, act_dim), False)
                       for label, path in vanilla_seeker_ckpts]
    vanilla_hiders = [(label, load_vanilla_checkpoint(str(path), obs_dim, act_dim), False)
                      for label, path in vanilla_hider_ckpts]

    scro_seekers = [(label, load_scro_checkpoint(str(path), obs_dim, act_dim), True)
                    for label, path in scro_seeker_ckpts]
    scro_hiders = [(label, load_scro_checkpoint(str(path), obs_dim, act_dim), True)
                   for label, path in scro_hider_ckpts]

    print(f"  Loaded {len(vanilla_seekers)} vanilla seekers, {len(vanilla_hiders)} vanilla hiders")
    print(f"  Loaded {len(scro_seekers)} SCRO seekers, {len(scro_hiders)} SCRO hiders")

    results = {}

    # 1. Vanilla vs Vanilla (internal)
    print("\n" + "="*60)
    print("Computing Vanilla internal cross-play...")
    v_win, v_len, v_s_labels, v_h_labels = compute_crossplay_matrix(
        vanilla_seekers, vanilla_hiders, env_config, args.episodes
    )
    results['vanilla_internal'] = {
        'win_matrix': v_win.tolist(),
        'seeker_labels': v_s_labels,
        'hider_labels': v_h_labels,
    }

    # 2. SCRO vs SCRO (internal) - if we have SCRO checkpoints
    if scro_seekers and scro_hiders:
        print("\n" + "="*60)
        print("Computing SCRO internal cross-play...")
        s_win, s_len, s_s_labels, s_h_labels = compute_crossplay_matrix(
            scro_seekers, scro_hiders, env_config, args.episodes
        )
        results['scro_internal'] = {
            'win_matrix': s_win.tolist(),
            'seeker_labels': s_s_labels,
            'hider_labels': s_h_labels,
        }

    # 3. Cross-method: Vanilla seekers vs SCRO hiders
    if scro_hiders:
        print("\n" + "="*60)
        print("Computing Vanilla Seekers vs SCRO Hiders...")
        vs_win, vs_len, vs_s_labels, vs_h_labels = compute_crossplay_matrix(
            vanilla_seekers, scro_hiders, env_config, args.episodes
        )
        results['vanilla_seeker_vs_scro_hider'] = {
            'win_matrix': vs_win.tolist(),
            'seeker_labels': vs_s_labels,
            'hider_labels': vs_h_labels,
        }

    # 4. Cross-method: SCRO seekers vs Vanilla hiders
    if scro_seekers:
        print("\n" + "="*60)
        print("Computing SCRO Seekers vs Vanilla Hiders...")
        sv_win, sv_len, sv_s_labels, sv_h_labels = compute_crossplay_matrix(
            scro_seekers, vanilla_hiders, env_config, args.episodes
        )
        results['scro_seeker_vs_vanilla_hider'] = {
            'win_matrix': sv_win.tolist(),
            'seeker_labels': sv_s_labels,
            'hider_labels': sv_h_labels,
        }

    # Create visualization
    print("\n" + "="*60)
    print("Creating visualization...")

    n_plots = 2  # At minimum vanilla internal + one cross-method
    if scro_seekers and scro_hiders:
        n_plots = 4  # Full comparison

    if n_plots == 4:
        fig = plt.figure(figsize=(16, 14))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)

        # Vanilla internal
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = plot_heatmap(ax1, v_win, v_s_labels, v_h_labels,
                          'Vanilla Self-Play: Seeker Win Rate (%)')
        ax1.set_xlabel('Vanilla Hider Checkpoint')
        ax1.set_ylabel('Vanilla Seeker Checkpoint')

        # SCRO internal
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = plot_heatmap(ax2, s_win, s_s_labels, s_h_labels,
                          'SCRO: Seeker Win Rate (%)')
        ax2.set_xlabel('SCRO Hider Checkpoint')
        ax2.set_ylabel('SCRO Seeker Checkpoint')

        # Vanilla seeker vs SCRO hider
        ax3 = fig.add_subplot(gs[1, 0])
        im3 = plot_heatmap(ax3, vs_win, vs_s_labels, vs_h_labels,
                          'Vanilla Seekers vs SCRO Hiders (%)')
        ax3.set_xlabel('SCRO Hider Checkpoint')
        ax3.set_ylabel('Vanilla Seeker Checkpoint')

        # SCRO seeker vs Vanilla hider
        ax4 = fig.add_subplot(gs[1, 1])
        im4 = plot_heatmap(ax4, sv_win, sv_s_labels, sv_h_labels,
                          'SCRO Seekers vs Vanilla Hiders (%)')
        ax4.set_xlabel('Vanilla Hider Checkpoint')
        ax4.set_ylabel('SCRO Seeker Checkpoint')

    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        ax1 = axes[0]
        im1 = plot_heatmap(ax1, v_win, v_s_labels, v_h_labels,
                          'Vanilla Self-Play: Seeker Win Rate (%)')
        ax1.set_xlabel('Hider Checkpoint')
        ax1.set_ylabel('Seeker Checkpoint')

        if scro_seekers and scro_hiders:
            ax2 = axes[1]
            im2 = plot_heatmap(ax2, s_win, s_s_labels, s_h_labels,
                              'SCRO: Seeker Win Rate (%)')
            ax2.set_xlabel('Hider Checkpoint')
            ax2.set_ylabel('Seeker Checkpoint')

    plt.suptitle('Checkpoint Cross-Play Comparison: Vanilla vs SCRO Training Dynamics',
                 fontsize=14, fontweight='bold')

    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {args.output}")
    plt.close()

    # Save results JSON
    json_path = args.output.replace('.png', '.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {json_path}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    print("\nVanilla Self-Play diagonal (contemporary matchups):")
    for i, label in enumerate(v_s_labels):
        if i < len(v_h_labels):
            print(f"  {label}: {v_win[i, i]:.1%} seeker WR")

    if scro_seekers and scro_hiders:
        print("\nSCRO diagonal (contemporary matchups):")
        for i, label in enumerate(s_s_labels):
            if i < len(s_h_labels):
                print(f"  {label}: {s_win[i, i]:.1%} seeker WR")

    print("\nCross-method summary (final checkpoints):")
    if vanilla_seekers and scro_hiders:
        print(f"  Vanilla final seeker vs SCRO final hider: {vs_win[-1, -1]:.1%}")
    if scro_seekers and vanilla_hiders:
        print(f"  SCRO final seeker vs Vanilla final hider: {sv_win[-1, -1]:.1%}")


if __name__ == "__main__":
    main()
