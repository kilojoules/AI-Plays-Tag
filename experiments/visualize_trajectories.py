#!/usr/bin/env python3
"""
Create static trajectory visualization showing multiple episodes.
"""
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, str(Path(__file__).parent.parent))

from trainer.tag_env import SingleTagEnv, TagEnvConfig, LAYOUTS
from trainer.ppo import PPOAgent, PPOConfig


def load_policy(path: str, obs_dim: int, act_dim: int) -> PPOAgent:
    cfg = PPOConfig(obs_dim=obs_dim, act_dim=act_dim)
    policy = PPOAgent(cfg)
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    policy.pi.load_state_dict(ckpt["pi"])
    policy.vf.load_state_dict(ckpt["vf"])
    return policy


def run_episode(seeker, hider, env_config, max_steps=200):
    env = SingleTagEnv(config=env_config)
    obs = env.reset()

    seeker_traj = []
    hider_traj = []
    done = False
    step = 0

    while not done and step < max_steps:
        state = env.get_state()
        seeker_idx = state['seeker_idx']
        positions = state['positions']

        seeker_traj.append(positions[seeker_idx].copy())
        hider_traj.append(positions[1 - seeker_idx].copy())

        with torch.no_grad():
            s_act, _, _ = seeker.act(obs['seeker'])
            h_act, _, _ = hider.act(obs['hider'])

        obs, _, done, info = env.step({'seeker': s_act.squeeze(), 'hider': h_act.squeeze()})
        step += 1

    return {
        'seeker_traj': np.array(seeker_traj),
        'hider_traj': np.array(hider_traj),
        'tagged': info.get('tagged', False),
        'steps': step,
    }


def main():
    # Find latest checkpoint
    long_dir = Path("experiments/results/long_training")
    run_dir = sorted(long_dir.glob("*"))[-1]
    ckpt_dir = run_dir / "checkpoints"

    seeker_path = ckpt_dir / "seeker_00300.pt"
    hider_path = ckpt_dir / "hider_00300.pt"

    env_config = TagEnvConfig(layout="four_corners")
    env = SingleTagEnv(config=env_config)

    seeker = load_policy(str(seeker_path), env.obs_dim, env.act_dim)
    hider = load_policy(str(hider_path), env.obs_dim, env.act_dim)

    layout = LAYOUTS.get(env_config.layout, LAYOUTS['empty'])
    obstacles = layout.get('obstacles', [])
    safe_zone = layout.get('safe_zone', None)
    arena_size = env_config.arena_half

    # Run episodes
    np.random.seed(123)
    episodes = []
    for i in range(6):
        result = run_episode(seeker, hider, env_config)
        episodes.append(result)

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (ax, ep) in enumerate(zip(axes, episodes)):
        ax.set_xlim(-arena_size - 1, arena_size + 1)
        ax.set_ylim(-arena_size - 1, arena_size + 1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # Arena
        arena_rect = patches.Rectangle(
            (-arena_size, -arena_size), 2 * arena_size, 2 * arena_size,
            linewidth=2, edgecolor='black', facecolor='white'
        )
        ax.add_patch(arena_rect)

        # Obstacles
        for obs in obstacles:
            rect = patches.Rectangle(
                (obs.x - obs.half_width, obs.y - obs.half_height),
                2 * obs.half_width, 2 * obs.half_height,
                linewidth=1, edgecolor='black', facecolor='gray', alpha=0.7
            )
            ax.add_patch(rect)

        # Safe zone
        if safe_zone:
            circle = patches.Circle(
                (safe_zone.x, safe_zone.y), safe_zone.radius,
                linewidth=2, edgecolor='green', facecolor='lightgreen', alpha=0.3
            )
            ax.add_patch(circle)

        # Trajectories
        seeker_traj = ep['seeker_traj']
        hider_traj = ep['hider_traj']

        ax.plot(seeker_traj[:, 0], seeker_traj[:, 1], 'r-', linewidth=1.5, alpha=0.7, label='Seeker')
        ax.plot(hider_traj[:, 0], hider_traj[:, 1], 'b-', linewidth=1.5, alpha=0.7, label='Hider')

        # Start positions
        ax.scatter(seeker_traj[0, 0], seeker_traj[0, 1], c='red', s=100, marker='o', edgecolors='black', zorder=5)
        ax.scatter(hider_traj[0, 0], hider_traj[0, 1], c='blue', s=100, marker='o', edgecolors='black', zorder=5)

        # End positions
        ax.scatter(seeker_traj[-1, 0], seeker_traj[-1, 1], c='red', s=150, marker='*', edgecolors='black', zorder=5)
        ax.scatter(hider_traj[-1, 0], hider_traj[-1, 1], c='blue', s=150, marker='*', edgecolors='black', zorder=5)

        result = "TAGGED" if ep['tagged'] else "Survived"
        ax.set_title(f"Episode {idx+1}: {result} in {ep['steps']} steps", fontsize=11)

        if idx == 0:
            ax.legend(loc='upper right', fontsize=9)

    plt.suptitle("Long Training Results (Update 300, ~614k steps)\nSeeker (red) vs Hider (blue)", fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = "experiments/results/long_training_trajectories.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Summary
    tagged = sum(1 for ep in episodes if ep['tagged'])
    avg_steps = np.mean([ep['steps'] for ep in episodes if ep['tagged']])
    print(f"\nSummary: {tagged}/{len(episodes)} tagged, avg {avg_steps:.0f} steps when tagged")


if __name__ == "__main__":
    main()
