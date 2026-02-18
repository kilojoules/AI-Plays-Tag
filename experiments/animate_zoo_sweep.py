#!/usr/bin/env python3
"""
Create 2D animations of one episode per zoo sweep experiment,
using the latest checkpoints. Outputs GIFs (no ffmpeg needed).
"""
import sys
from pathlib import Path
import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, str(Path(__file__).parent.parent))

from trainer.tag_env import SingleTagEnv, TagEnvConfig, LAYOUTS
from trainer.ppo import PPOAgent, PPOConfig


SWEEP_DIR = Path("experiments/results/zoo_sweep")
EXPERIMENTS = [
    "A00_hider_only", "A00_both",
    "A05_hider_only", "A05_both",
    "A10_hider_only", "A10_both",
    "A20_hider_only", "A20_both",
]

LABELS = {
    "A00_hider_only": "A=0 (no zoo), hider zoo only",
    "A00_both":       "A=0 (no zoo), both zoos",
    "A05_hider_only": "A=0.05, hider zoo only",
    "A05_both":       "A=0.05, both zoos",
    "A10_hider_only": "A=0.1, hider zoo only",
    "A10_both":       "A=0.1, both zoos",
    "A20_hider_only": "A=0.2, hider zoo only",
    "A20_both":       "A=0.2, both zoos",
}


def find_latest_checkpoints(exp_name):
    """Find the latest seeker/hider checkpoint pair for an experiment."""
    exp_dir = SWEEP_DIR / exp_name
    if not exp_dir.exists():
        return None, None

    # Find latest run subdir
    subdirs = sorted(exp_dir.glob("2*"), key=lambda p: p.name)
    if not subdirs:
        return None, None
    run_dir = subdirs[-1]
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None, None

    seeker_ckpts = sorted(ckpt_dir.glob("seeker_*.pt"))
    hider_ckpts = sorted(ckpt_dir.glob("hider_*.pt"))
    if not seeker_ckpts or not hider_ckpts:
        return None, None

    return str(seeker_ckpts[-1]), str(hider_ckpts[-1])


def load_policy(path, obs_dim, act_dim):
    cfg = PPOConfig(obs_dim=obs_dim, act_dim=act_dim)
    policy = PPOAgent(cfg)
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    policy.pi.load_state_dict(ckpt["pi"])
    policy.vf.load_state_dict(ckpt["vf"])
    return policy


def run_episode(seeker, hider, env_config, max_steps=200):
    env = SingleTagEnv(config=env_config)
    obs = env.reset()
    trajectory = []
    done = False
    step = 0

    while not done and step < max_steps:
        state = env.get_state()
        seeker_idx = state['seeker_idx']
        positions = state['positions']
        trajectory.append({
            'seeker_pos': positions[seeker_idx].copy(),
            'hider_pos': positions[1 - seeker_idx].copy(),
            'step': step,
            'safe_zone_exhausted': state['safe_zone_exhausted'],
        })
        with torch.no_grad():
            s_act, _, _ = seeker.act(obs['seeker'])
            h_act, _, _ = hider.act(obs['hider'])
        obs, _, done, info = env.step({'seeker': s_act.squeeze(), 'hider': h_act.squeeze()})
        step += 1

    # Final frame
    state = env.get_state()
    seeker_idx = state['seeker_idx']
    positions = state['positions']
    trajectory.append({
        'seeker_pos': positions[seeker_idx].copy(),
        'hider_pos': positions[1 - seeker_idx].copy(),
        'step': step,
        'tagged': info.get('tagged', False),
        'safe_zone_exhausted': state['safe_zone_exhausted'],
    })
    return trajectory


def create_animation(trajectory, title, output_path, env_config):
    fig, ax = plt.subplots(figsize=(8, 8))

    layout = LAYOUTS.get(env_config.layout, LAYOUTS['empty'])
    obstacles = layout.get('obstacles', [])
    safe_zone = layout.get('safe_zone', None)
    arena_size = env_config.arena_half

    final = trajectory[-1]
    result = "TAGGED!" if final.get('tagged', False) else "Survived"
    result_color = 'red' if final.get('tagged', False) else 'blue'
    n_frames = len(trajectory)

    def animate(frame_idx):
        ax.clear()
        ax.set_xlim(-arena_size - 1, arena_size + 1)
        ax.set_ylim(-arena_size - 1, arena_size + 1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)

        # Arena
        arena_rect = patches.Rectangle(
            (-arena_size, -arena_size), 2 * arena_size, 2 * arena_size,
            linewidth=2, edgecolor='black', facecolor='#f8f8f8'
        )
        ax.add_patch(arena_rect)

        # Obstacles
        for obs in obstacles:
            rect = patches.Rectangle(
                (obs.x - obs.half_width, obs.y - obs.half_height),
                2 * obs.half_width, 2 * obs.half_height,
                linewidth=1, edgecolor='#555', facecolor='#aaa', alpha=0.8
            )
            ax.add_patch(rect)

        # Safe zone
        if safe_zone:
            frame_data = trajectory[min(frame_idx, n_frames - 1)]
            exhausted = frame_data.get('safe_zone_exhausted', False)
            if exhausted:
                fc, ec = 'lightsalmon', 'red'
            else:
                fc, ec = 'lightgreen', 'green'
            circle = patches.Circle(
                (safe_zone.x, safe_zone.y), safe_zone.radius,
                linewidth=2, edgecolor=ec, facecolor=fc, alpha=0.35
            )
            ax.add_patch(circle)

        frame = trajectory[min(frame_idx, n_frames - 1)]
        seeker_pos = frame['seeker_pos']
        hider_pos = frame['hider_pos']

        # Trails (last 30 frames)
        trail_start = max(0, frame_idx - 30)
        if frame_idx > 0:
            seeker_trail = [trajectory[i]['seeker_pos'] for i in range(trail_start, frame_idx + 1)]
            hider_trail = [trajectory[i]['hider_pos'] for i in range(trail_start, frame_idx + 1)]

            # Fading trail
            for i in range(len(seeker_trail) - 1):
                alpha = 0.1 + 0.5 * (i / max(len(seeker_trail) - 1, 1))
                ax.plot([seeker_trail[i][0], seeker_trail[i+1][0]],
                        [seeker_trail[i][1], seeker_trail[i+1][1]],
                        'r-', alpha=alpha, linewidth=2)
                ax.plot([hider_trail[i][0], hider_trail[i+1][0]],
                        [hider_trail[i][1], hider_trail[i+1][1]],
                        'b-', alpha=alpha, linewidth=2)

        # Agents
        ax.add_patch(patches.Circle(seeker_pos, 0.6, color='#e74c3c', zorder=5))
        ax.add_patch(patches.Circle(hider_pos, 0.6, color='#3498db', zorder=5))
        ax.annotate('S', seeker_pos, ha='center', va='center',
                     fontsize=10, fontweight='bold', color='white', zorder=6)
        ax.annotate('H', hider_pos, ha='center', va='center',
                     fontsize=10, fontweight='bold', color='white', zorder=6)

        # Title
        step_str = f"Step {frame['step']}/{n_frames - 1}"
        if frame_idx == n_frames - 1:
            step_str += f"  |  {result}"
        ax.set_title(f"{title}\n{step_str}", fontsize=11, fontweight='bold')

        return []

    anim = FuncAnimation(fig, animate, frames=n_frames, interval=50, blit=False)
    writer = PillowWriter(fps=20)
    anim.save(output_path, writer=writer)
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    output_dir = Path("experiments/results/zoo_sweep/animations")
    output_dir.mkdir(parents=True, exist_ok=True)

    env_config = TagEnvConfig(layout="four_corners")
    env = SingleTagEnv(config=env_config)
    obs_dim, act_dim = env.obs_dim, env.act_dim

    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 60)
    print("ZOO SWEEP EPISODE ANIMATIONS")
    print("=" * 60)

    results_summary = []

    for exp_name in EXPERIMENTS:
        print(f"\n--- {exp_name} ---")
        seeker_path, hider_path = find_latest_checkpoints(exp_name)
        if seeker_path is None:
            print(f"  SKIP: no checkpoints found")
            continue

        print(f"  Seeker: {Path(seeker_path).name}")
        print(f"  Hider:  {Path(hider_path).name}")

        seeker = load_policy(seeker_path, obs_dim, act_dim)
        hider = load_policy(hider_path, obs_dim, act_dim)

        trajectory = run_episode(seeker, hider, env_config)

        final = trajectory[-1]
        tagged = final.get('tagged', False)
        result_str = "Tagged" if tagged else "Survived"
        steps = final['step']
        print(f"  Result: {result_str} in {steps} steps")
        results_summary.append((exp_name, result_str, steps))

        label = LABELS.get(exp_name, exp_name)
        gif_path = str(output_dir / f"{exp_name}.gif")
        create_animation(trajectory, label, gif_path, env_config)

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for name, result, steps in results_summary:
        print(f"  {name:20s}: {result:8s} ({steps} steps)")

    print(f"\nAnimations saved to: {output_dir}")


if __name__ == "__main__":
    main()
