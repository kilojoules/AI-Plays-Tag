#!/usr/bin/env python3
"""
Visualize results from stable long training.
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


def create_trajectory_plot(episodes, output_path, env_config, title):
    """Create static trajectory visualization."""
    layout = LAYOUTS.get(env_config.layout, LAYOUTS['empty'])
    obstacles = layout.get('obstacles', [])
    safe_zone = layout.get('safe_zone', None)
    arena_size = env_config.arena_half

    n_episodes = len(episodes)
    cols = min(3, n_episodes)
    rows = (n_episodes + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    if n_episodes == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (ax, traj) in enumerate(zip(axes, episodes)):
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

        # Extract trajectories
        seeker_traj = np.array([f['seeker_pos'] for f in traj])
        hider_traj = np.array([f['hider_pos'] for f in traj])

        ax.plot(seeker_traj[:, 0], seeker_traj[:, 1], 'r-', linewidth=1.5, alpha=0.7, label='Seeker')
        ax.plot(hider_traj[:, 0], hider_traj[:, 1], 'b-', linewidth=1.5, alpha=0.7, label='Hider')

        # Start positions
        ax.scatter(seeker_traj[0, 0], seeker_traj[0, 1], c='red', s=100, marker='o', edgecolors='black', zorder=5)
        ax.scatter(hider_traj[0, 0], hider_traj[0, 1], c='blue', s=100, marker='o', edgecolors='black', zorder=5)

        # End positions
        ax.scatter(seeker_traj[-1, 0], seeker_traj[-1, 1], c='red', s=150, marker='*', edgecolors='black', zorder=5)
        ax.scatter(hider_traj[-1, 0], hider_traj[-1, 1], c='blue', s=150, marker='*', edgecolors='black', zorder=5)

        final = traj[-1]
        result = "TAGGED" if final.get('tagged', False) else "Survived"
        steps = final['step']
        ax.set_title(f"Episode {idx+1}: {result} in {steps} steps", fontsize=11)

        if idx == 0:
            ax.legend(loc='upper right', fontsize=9)

    # Hide unused subplots
    for idx in range(len(episodes), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_training_curve(metrics_path, output_path):
    """Plot training metrics over time."""
    import pandas as pd

    df = pd.read_csv(metrics_path)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Win rate over time
    ax = axes[0, 0]
    ax.plot(df['timesteps'] / 1e6, df['seeker_win_rate'] * 100, 'r-', label='Seeker', alpha=0.7)
    ax.plot(df['timesteps'] / 1e6, df['hider_win_rate'] * 100, 'b-', label='Hider', alpha=0.7)
    ax.set_xlabel('Timesteps (millions)')
    ax.set_ylabel('Win Rate (%)')
    ax.set_title('Win Rate Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    # Episode length
    ax = axes[0, 1]
    ax.plot(df['timesteps'] / 1e6, df['episode_length_mean'], 'g-', alpha=0.7)
    ax.set_xlabel('Timesteps (millions)')
    ax.set_ylabel('Episode Length')
    ax.set_title('Average Episode Length')
    ax.grid(True, alpha=0.3)

    # Rewards
    ax = axes[1, 0]
    ax.plot(df['timesteps'] / 1e6, df['seeker_reward_mean'], 'r-', label='Seeker', alpha=0.7)
    ax.plot(df['timesteps'] / 1e6, df['hider_reward_mean'], 'b-', label='Hider', alpha=0.7)
    ax.set_xlabel('Timesteps (millions)')
    ax.set_ylabel('Mean Reward')
    ax.set_title('Mean Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Policy losses
    ax = axes[1, 1]
    ax.plot(df['timesteps'] / 1e6, df['seeker_policy_loss'], 'r-', label='Seeker', alpha=0.7)
    ax.plot(df['timesteps'] / 1e6, df['hider_policy_loss'], 'b-', label='Hider', alpha=0.7)
    ax.set_xlabel('Timesteps (millions)')
    ax.set_ylabel('Policy Loss')
    ax.set_title('Policy Losses')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Stable Long Training Progress (~8M steps)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    # Find the stable training run
    stable_dir = Path("experiments/results/stable_long_training")
    run_dir = sorted(stable_dir.glob("*"))[-1]
    ckpt_dir = run_dir / "checkpoints"
    metrics_path = run_dir / "metrics.csv"

    # Find latest checkpoint
    seeker_ckpts = sorted(ckpt_dir.glob("seeker_*.pt"))
    latest_update = int(seeker_ckpts[-1].stem.split('_')[1])

    seeker_path = ckpt_dir / f"seeker_{latest_update:05d}.pt"
    hider_path = ckpt_dir / f"hider_{latest_update:05d}.pt"

    print(f"Using checkpoints from: {ckpt_dir}")
    print(f"  Latest update: {latest_update}")
    print(f"  Seeker: {seeker_path}")
    print(f"  Hider: {hider_path}")

    env_config = TagEnvConfig(layout="four_corners")
    env = SingleTagEnv(config=env_config)

    seeker = load_policy(str(seeker_path), env.obs_dim, env.act_dim)
    hider = load_policy(str(hider_path), env.obs_dim, env.act_dim)

    output_dir = Path("experiments/results/stable_training_viz")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot training curve
    print("\nPlotting training curve...")
    plot_training_curve(str(metrics_path), str(output_dir / "training_curve.png"))

    # Run evaluation episodes
    print("\nRunning evaluation episodes...")
    np.random.seed(42)

    episodes = []
    results = []
    for i in range(6):
        print(f"  Episode {i+1}...")
        trajectory = run_episode(seeker, hider, env_config)
        episodes.append(trajectory)

        final = trajectory[-1]
        result = "Tagged" if final.get('tagged', False) else "Survived"
        results.append((result, final['step']))
        print(f"    Result: {result} in {final['step']} steps")

    # Create trajectory plot
    print("\nCreating trajectory visualization...")
    create_trajectory_plot(
        episodes,
        str(output_dir / "trajectories.png"),
        env_config,
        f"Stable Training Results (Update {latest_update}, ~{latest_update * 2048 // 1000}k steps)\nSeeker (red) vs Hider (blue)"
    )

    # Create animations for interesting episodes
    print("\nCreating animations...")
    for i, (result, steps) in enumerate(results):
        if i < 3:  # Animate first 3 episodes
            output_path = output_dir / f"episode_{i+1}.mp4"
            create_animation(
                episodes[i],
                f"Stable Training (Update {latest_update}) - Episode {i+1}",
                str(output_path),
                env_config
            )

    # Summary
    print(f"\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    tagged = sum(1 for r, _ in results if r == "Tagged")
    print(f"Tagged: {tagged}/{len(results)} ({tagged*100//len(results)}%)")
    avg_steps_tagged = np.mean([s for r, s in results if r == "Tagged"]) if tagged > 0 else 0
    avg_steps_survived = np.mean([s for r, s in results if r == "Survived"]) if tagged < len(results) else 0
    for i, (result, steps) in enumerate(results):
        print(f"  Episode {i+1}: {result} in {steps} steps")

    if tagged > 0:
        print(f"\nAvg steps when tagged: {avg_steps_tagged:.1f}")
    if tagged < len(results):
        print(f"Avg steps when survived: {avg_steps_survived:.1f}")

    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
