#!/usr/bin/env python3
"""
Create animations for all 9 matchup combinations from pretrained comparison.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, FFMpegWriter

sys.path.insert(0, str(Path(__file__).parent.parent))

from trainer.tag_env import SingleTagEnv, TagEnvConfig, LAYOUTS
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


def run_episode(seeker, hider, seeker_is_scro: bool, hider_is_scro: bool,
                env_config: TagEnvConfig, max_steps: int = 200) -> List[Dict]:
    """Run one episode and record trajectory."""
    env = SingleTagEnv(config=env_config)
    obs = env.reset()

    trajectory = []
    done = False
    step = 0

    while not done and step < max_steps:
        # Record state using get_state()
        state = env.get_state()
        seeker_idx = state['seeker_idx']
        positions = state['positions']

        trajectory.append({
            'seeker_pos': positions[seeker_idx].copy(),
            'hider_pos': positions[1 - seeker_idx].copy(),
            'step': step,
        })

        s_act = get_action(seeker, obs['seeker'], seeker_is_scro)
        h_act = get_action(hider, obs['hider'], hider_is_scro)
        obs, _, done, info = env.step({'seeker': s_act, 'hider': h_act})
        step += 1

    # Record final state
    state = env.get_state()
    seeker_idx = state['seeker_idx']
    positions = state['positions']

    trajectory.append({
        'seeker_pos': positions[seeker_idx].copy(),
        'hider_pos': positions[1 - seeker_idx].copy(),
        'step': step,
        'tagged': info.get('tagged', False),
    })

    return trajectory


def create_animation(trajectory: List[Dict], seeker_name: str, hider_name: str,
                     output_path: str, env_config: TagEnvConfig):
    """Create animation from trajectory."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Get layout info
    layout_name = env_config.layout
    layout = LAYOUTS.get(layout_name, LAYOUTS['empty'])
    obstacles = layout.get('obstacles', [])
    safe_zone = layout.get('safe_zone', None)

    arena_size = env_config.arena_half

    def init():
        ax.clear()
        ax.set_xlim(-arena_size - 1, arena_size + 1)
        ax.set_ylim(-arena_size - 1, arena_size + 1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        return []

    def animate(frame_idx):
        ax.clear()
        ax.set_xlim(-arena_size - 1, arena_size + 1)
        ax.set_ylim(-arena_size - 1, arena_size + 1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # Draw arena boundary
        arena_rect = patches.Rectangle(
            (-arena_size, -arena_size), 2 * arena_size, 2 * arena_size,
            linewidth=2, edgecolor='black', facecolor='white'
        )
        ax.add_patch(arena_rect)

        # Draw obstacles
        for obs in obstacles:
            rect = patches.Rectangle(
                (obs.x - obs.half_width, obs.y - obs.half_height),
                2 * obs.half_width, 2 * obs.half_height,
                linewidth=1, edgecolor='black', facecolor='gray', alpha=0.7
            )
            ax.add_patch(rect)

        # Draw safe zone
        if safe_zone:
            circle = patches.Circle(
                (safe_zone.x, safe_zone.y), safe_zone.radius,
                linewidth=2, edgecolor='green', facecolor='lightgreen', alpha=0.3
            )
            ax.add_patch(circle)

        # Get current frame
        frame = trajectory[min(frame_idx, len(trajectory) - 1)]
        seeker_pos = frame['seeker_pos']
        hider_pos = frame['hider_pos']

        # Draw trajectory trails
        trail_len = min(20, frame_idx)
        if trail_len > 0:
            start_idx = max(0, frame_idx - trail_len)
            seeker_trail = [trajectory[i]['seeker_pos'] for i in range(start_idx, frame_idx + 1)]
            hider_trail = [trajectory[i]['hider_pos'] for i in range(start_idx, frame_idx + 1)]

            if len(seeker_trail) > 1:
                sx = [p[0] for p in seeker_trail]
                sy = [p[1] for p in seeker_trail]
                ax.plot(sx, sy, 'r-', alpha=0.5, linewidth=1)

            if len(hider_trail) > 1:
                hx = [p[0] for p in hider_trail]
                hy = [p[1] for p in hider_trail]
                ax.plot(hx, hy, 'b-', alpha=0.5, linewidth=1)

        # Draw agents
        seeker_circle = patches.Circle(seeker_pos, 0.5, color='red', label='Seeker')
        hider_circle = patches.Circle(hider_pos, 0.5, color='blue', label='Hider')
        ax.add_patch(seeker_circle)
        ax.add_patch(hider_circle)

        # Title
        final_frame = trajectory[-1]
        result = "TAGGED!" if final_frame.get('tagged', False) else "Survived"
        ax.set_title(f"{seeker_name} (Seeker) vs {hider_name} (Hider)\n"
                    f"Step {frame['step']}/{len(trajectory)-1} | {result}",
                    fontsize=12)

        ax.legend(loc='upper right')
        return []

    # Create animation
    anim = FuncAnimation(fig, animate, init_func=init,
                        frames=len(trajectory), interval=50, blit=False)

    # Save
    writer = FFMpegWriter(fps=20, bitrate=1800)
    anim.save(output_path, writer=writer)
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    # Paths
    original_seeker = "experiments/results/obstacles/comparison_20260124_212516/vanilla_selfplay/seed_42/20260124_212519/policy_seeker_final.pt"
    original_hider = "experiments/results/obstacles/comparison_20260124_212516/vanilla_selfplay/seed_42/20260124_212519/policy_hider_final.pt"

    pretrained_vanilla_dir = Path("experiments/results/pretrained_vanilla")
    latest_vanilla = sorted(pretrained_vanilla_dir.glob("*"))[-1]
    pretrained_v_seeker = latest_vanilla / "policy_seeker_final.pt"
    pretrained_v_hider = latest_vanilla / "policy_hider_final.pt"

    pretrained_scro_dir = Path("experiments/results/pretrained_scro")
    latest_scro = sorted(pretrained_scro_dir.glob("*"))[-1]
    pretrained_s_seeker = latest_scro / "best_protagonist.pt"
    pretrained_s_hider = latest_scro / "best_antagonist.pt"

    # Setup
    env_config = TagEnvConfig(layout="four_corners")
    env = SingleTagEnv(config=env_config)
    obs_dim = env.obs_dim
    act_dim = env.act_dim

    # Load policies
    print("Loading policies...")
    orig_seeker = load_vanilla(original_seeker, obs_dim, act_dim)
    orig_hider = load_vanilla(original_hider, obs_dim, act_dim)

    pt_v_seeker = load_vanilla(str(pretrained_v_seeker), obs_dim, act_dim)
    pt_v_hider = load_vanilla(str(pretrained_v_hider), obs_dim, act_dim)

    pt_s_seeker = load_scro(str(pretrained_s_seeker), obs_dim, act_dim)
    pt_s_hider = load_scro(str(pretrained_s_hider), obs_dim, act_dim)

    # Define matchups
    seekers = [
        ("Original_Vanilla", orig_seeker, False),
        ("Continued_Vanilla", pt_v_seeker, False),
        ("Pretrained_SCRO", pt_s_seeker, True),
    ]

    hiders = [
        ("Original_Vanilla", orig_hider, False),
        ("Continued_Vanilla", pt_v_hider, False),
        ("Pretrained_SCRO", pt_s_hider, True),
    ]

    # Output directory
    output_dir = Path("experiments/results/matchup_animations")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run all matchups
    print(f"\nCreating animations for {len(seekers) * len(hiders)} matchups...")

    np.random.seed(42)

    for s_name, s_policy, s_scro in seekers:
        for h_name, h_policy, h_scro in hiders:
            print(f"\n{s_name} vs {h_name}...")

            # Run episode
            trajectory = run_episode(s_policy, h_policy, s_scro, h_scro, env_config)

            # Create animation
            output_path = output_dir / f"{s_name}_vs_{h_name}.mp4"
            create_animation(trajectory, s_name.replace('_', ' '),
                           h_name.replace('_', ' '), str(output_path), env_config)

            # Print result
            final = trajectory[-1]
            result = "Tagged" if final.get('tagged', False) else "Survived"
            print(f"    Result: {result} in {final['step']} steps")

    print(f"\nAll animations saved to: {output_dir}")


if __name__ == "__main__":
    main()
