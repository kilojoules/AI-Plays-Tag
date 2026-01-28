#!/usr/bin/env python3
"""
Create animations from long training checkpoints.
"""
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, FFMpegWriter

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
        })

        with torch.no_grad():
            s_act, _, _ = seeker.act(obs['seeker'])
            h_act, _, _ = hider.act(obs['hider'])

        obs, _, done, info = env.step({'seeker': s_act.squeeze(), 'hider': h_act.squeeze()})
        step += 1

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


def create_animation(trajectory, title, output_path, env_config):
    fig, ax = plt.subplots(figsize=(10, 10))

    layout = LAYOUTS.get(env_config.layout, LAYOUTS['empty'])
    obstacles = layout.get('obstacles', [])
    safe_zone = layout.get('safe_zone', None)
    arena_size = env_config.arena_half

    def animate(frame_idx):
        ax.clear()
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

        frame = trajectory[min(frame_idx, len(trajectory) - 1)]
        seeker_pos = frame['seeker_pos']
        hider_pos = frame['hider_pos']

        # Trails
        trail_len = min(30, frame_idx)
        if trail_len > 0:
            start_idx = max(0, frame_idx - trail_len)
            seeker_trail = [trajectory[i]['seeker_pos'] for i in range(start_idx, frame_idx + 1)]
            hider_trail = [trajectory[i]['hider_pos'] for i in range(start_idx, frame_idx + 1)]

            if len(seeker_trail) > 1:
                sx = [p[0] for p in seeker_trail]
                sy = [p[1] for p in seeker_trail]
                ax.plot(sx, sy, 'r-', alpha=0.5, linewidth=2)

            if len(hider_trail) > 1:
                hx = [p[0] for p in hider_trail]
                hy = [p[1] for p in hider_trail]
                ax.plot(hx, hy, 'b-', alpha=0.5, linewidth=2)

        # Agents
        seeker_circle = patches.Circle(seeker_pos, 0.5, color='red', label='Seeker')
        hider_circle = patches.Circle(hider_pos, 0.5, color='blue', label='Hider')
        ax.add_patch(seeker_circle)
        ax.add_patch(hider_circle)

        final = trajectory[-1]
        result = "TAGGED!" if final.get('tagged', False) else "Survived"
        ax.set_title(f"{title}\nStep {frame['step']}/{len(trajectory)-1} | {result}", fontsize=14)
        ax.legend(loc='upper right')

        return []

    anim = FuncAnimation(fig, animate, frames=len(trajectory), interval=50, blit=False)
    writer = FFMpegWriter(fps=20, bitrate=2400)
    anim.save(output_path, writer=writer)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    # Find latest checkpoint
    long_dir = Path("experiments/results/long_training")
    run_dir = sorted(long_dir.glob("*"))[-1]
    ckpt_dir = run_dir / "checkpoints"

    # Use checkpoint 300 (latest before crash)
    seeker_path = ckpt_dir / "seeker_00300.pt"
    hider_path = ckpt_dir / "hider_00300.pt"

    print(f"Using checkpoints from: {ckpt_dir}")
    print(f"  Seeker: {seeker_path}")
    print(f"  Hider: {hider_path}")

    env_config = TagEnvConfig(layout="four_corners")
    env = SingleTagEnv(config=env_config)

    seeker = load_policy(str(seeker_path), env.obs_dim, env.act_dim)
    hider = load_policy(str(hider_path), env.obs_dim, env.act_dim)

    output_dir = Path("experiments/results/long_training_animations")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run multiple episodes
    print("\nCreating animations...")
    np.random.seed(42)

    results = []
    for i in range(5):
        print(f"\nEpisode {i+1}...")
        trajectory = run_episode(seeker, hider, env_config)

        final = trajectory[-1]
        result = "Tagged" if final.get('tagged', False) else "Survived"
        results.append((result, final['step']))
        print(f"  Result: {result} in {final['step']} steps")

        output_path = output_dir / f"episode_{i+1}.mp4"
        create_animation(trajectory, f"Long Training (Update 300) - Episode {i+1}",
                        str(output_path), env_config)

    print(f"\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    tagged = sum(1 for r, _ in results if r == "Tagged")
    print(f"Tagged: {tagged}/5 ({tagged*20}%)")
    for i, (result, steps) in enumerate(results):
        print(f"  Episode {i+1}: {result} in {steps} steps")

    print(f"\nAnimations saved to: {output_dir}")


if __name__ == "__main__":
    main()
