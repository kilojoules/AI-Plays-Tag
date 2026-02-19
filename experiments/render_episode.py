#!/usr/bin/env python3
"""Render a single episode from trained agents as MP4."""
import sys
from pathlib import Path
import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, FFMpegWriter

sys.path.insert(0, str(Path(__file__).parent.parent))

from trainer.tag_env import SingleTagEnv, TagEnvConfig, LAYOUTS
from trainer.ppo import PPOAgent, PPOConfig
from trainer.sac import SACAgent, SACConfig


def load_policy(path, obs_dim, act_dim):
    """Auto-detect checkpoint type (SAC or PPO) and load accordingly."""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict) and ckpt.get('type') == 'sac':
        hidden_dim = ckpt.get('config', {}).get('hidden_dim', 256)
        cfg = SACConfig(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=hidden_dim)
        policy = SACAgent(cfg)
        policy.actor.load_state_dict(ckpt['actor'])
        policy.critic.load_state_dict(ckpt['critic'])
        policy.critic_target.load_state_dict(ckpt['critic_target'])
        return policy
    else:
        # PPO checkpoint (default)
        cfg = PPOConfig(obs_dim=obs_dim, act_dim=act_dim)
        policy = PPOAgent(cfg)
        if isinstance(ckpt, dict) and 'pi' in ckpt:
            policy.pi.load_state_dict(ckpt["pi"])
            if 'vf' in ckpt:
                policy.vf.load_state_dict(ckpt["vf"])
        else:
            policy.pi.load_state_dict(ckpt)
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
        stamina = state.get('stamina', np.array([0.0, 0.0]))
        trajectory.append({
            'seeker_pos': positions[seeker_idx].copy(),
            'hider_pos': positions[1 - seeker_idx].copy(),
            'seeker_stamina': float(stamina[seeker_idx]),
            'hider_stamina': float(stamina[1 - seeker_idx]),
            'step': step,
            'safe_zone_exhausted': state['safe_zone_exhausted'],
        })
        with torch.no_grad():
            s_act, _, _ = seeker.act(obs['seeker'])
            h_act, _, _ = hider.act(obs['hider'])
        obs, _, done, info = env.step({'seeker': s_act.squeeze(), 'hider': h_act.squeeze()})
        step += 1

    state = env.get_state()
    seeker_idx = state['seeker_idx']
    positions = state['positions']
    stamina = state.get('stamina', np.array([0.0, 0.0]))
    trajectory.append({
        'seeker_pos': positions[seeker_idx].copy(),
        'hider_pos': positions[1 - seeker_idx].copy(),
        'seeker_stamina': float(stamina[seeker_idx]),
        'hider_stamina': float(stamina[1 - seeker_idx]),
        'step': step,
        'tagged': info.get('tagged', False),
        'safe_zone_exhausted': state['safe_zone_exhausted'],
    })
    return trajectory


def create_mp4(trajectory, title, output_path, env_config, enable_sprint=False):
    fig, ax = plt.subplots(figsize=(8, 8))
    max_stamina = env_config.max_stamina if hasattr(env_config, 'max_stamina') else 3.0

    layout = LAYOUTS.get(env_config.layout, LAYOUTS['empty'])
    obstacles = layout.get('obstacles', [])
    safe_zone = layout.get('safe_zone', None)
    arena_size = env_config.arena_half

    final = trajectory[-1]
    result = "TAGGED!" if final.get('tagged', False) else "Survived"
    n_frames = len(trajectory)

    def animate(frame_idx):
        ax.clear()
        ax.set_xlim(-arena_size - 1, arena_size + 1)
        ax.set_ylim(-arena_size - 1, arena_size + 1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)

        arena_rect = patches.Rectangle(
            (-arena_size, -arena_size), 2 * arena_size, 2 * arena_size,
            linewidth=2, edgecolor='black', facecolor='#f8f8f8'
        )
        ax.add_patch(arena_rect)

        for obs in obstacles:
            rect = patches.Rectangle(
                (obs.x - obs.half_width, obs.y - obs.half_height),
                2 * obs.half_width, 2 * obs.half_height,
                linewidth=1, edgecolor='#555', facecolor='#aaa', alpha=0.8
            )
            ax.add_patch(rect)

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

        trail_start = max(0, frame_idx - 30)
        if frame_idx > 0:
            seeker_trail = [trajectory[i]['seeker_pos'] for i in range(trail_start, frame_idx + 1)]
            hider_trail = [trajectory[i]['hider_pos'] for i in range(trail_start, frame_idx + 1)]
            for i in range(len(seeker_trail) - 1):
                alpha = 0.1 + 0.5 * (i / max(len(seeker_trail) - 1, 1))
                ax.plot([seeker_trail[i][0], seeker_trail[i+1][0]],
                        [seeker_trail[i][1], seeker_trail[i+1][1]],
                        'r-', alpha=alpha, linewidth=2)
                ax.plot([hider_trail[i][0], hider_trail[i+1][0]],
                        [hider_trail[i][1], hider_trail[i+1][1]],
                        'b-', alpha=alpha, linewidth=2)

        ax.add_patch(patches.Circle(seeker_pos, 0.6, color='#e74c3c', zorder=5))
        ax.add_patch(patches.Circle(hider_pos, 0.6, color='#3498db', zorder=5))
        ax.annotate('S', seeker_pos, ha='center', va='center',
                     fontsize=10, fontweight='bold', color='white', zorder=6)
        ax.annotate('H', hider_pos, ha='center', va='center',
                     fontsize=10, fontweight='bold', color='white', zorder=6)

        # Draw stamina bars below agents when sprint is enabled
        if enable_sprint:
            bar_width = 1.6
            bar_height = 0.25
            bar_y_offset = -1.1
            for agent_pos, stam_key, color in [
                (seeker_pos, 'seeker_stamina', '#e74c3c'),
                (hider_pos, 'hider_stamina', '#3498db'),
            ]:
                stam = frame.get(stam_key, max_stamina)
                frac = np.clip(stam / max_stamina, 0, 1)
                bx = agent_pos[0] - bar_width / 2
                by = agent_pos[1] + bar_y_offset
                # Background (empty bar)
                ax.add_patch(patches.Rectangle(
                    (bx, by), bar_width, bar_height,
                    linewidth=0.5, edgecolor='#333', facecolor='#ddd', zorder=7))
                # Fill
                if frac > 0:
                    fill_color = '#2ecc71' if frac > 0.3 else '#e67e22'
                    ax.add_patch(patches.Rectangle(
                        (bx, by), bar_width * frac, bar_height,
                        linewidth=0, facecolor=fill_color, zorder=8))

        step_str = f"Step {frame['step']}/{n_frames - 1}"
        if frame_idx == n_frames - 1:
            step_str += f"  |  {result}"
        ax.set_title(f"{title}\n{step_str}", fontsize=11, fontweight='bold')
        return []

    anim = FuncAnimation(fig, animate, frames=n_frames, interval=50, blit=False)
    writer = FFMpegWriter(fps=20, bitrate=2000)
    anim.save(output_path, writer=writer)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=None, help="Direct path to run dir with policy_*_final.pt")
    parser.add_argument("--exp", default="A05_both", help="Experiment name (in zoo_sweep)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--title", default=None, help="Title for the animation")
    parser.add_argument("--output", default=None, help="Output mp4 path")
    parser.add_argument("--layout", default="four_corners",
                        choices=["empty", "four_corners", "central_cross", "playground"],
                        help="Arena layout")
    parser.add_argument("--enable-sprint", action="store_true",
                        help="Enable stamina/sprint system")
    parser.add_argument("--hider-speed-mult", type=float, default=1.0,
                        help="Hider base speed multiplier")
    parser.add_argument("--sprint-speed-mult", type=float, default=1.5,
                        help="Max speed multiplier when sprinting")
    args = parser.parse_args()

    if args.dir:
        # Direct path to a run directory
        run_dir = Path(args.dir)
        if not run_dir.exists():
            print(f"Directory not found: {run_dir}")
            sys.exit(1)
    else:
        sweep_dir = Path("experiments/results/zoo_sweep")
        exp_dir = sweep_dir / args.exp
        subdirs = sorted(exp_dir.glob("2*"), key=lambda p: p.name)
        if not subdirs:
            print(f"No run dirs in {exp_dir}")
            sys.exit(1)
        run_dir = subdirs[-1]

    # Prefer final policies, fall back to latest checkpoints
    seeker_path = run_dir / "policy_seeker_final.pt"
    hider_path = run_dir / "policy_hider_final.pt"
    if not seeker_path.exists():
        ckpt_dir = run_dir / "checkpoints"
        seeker_path = sorted(ckpt_dir.glob("seeker_*.pt"))[-1]
        hider_path = sorted(ckpt_dir.glob("hider_*.pt"))[-1]

    print(f"Run dir: {run_dir}")
    print(f"Seeker: {seeker_path.name}")
    print(f"Hider:  {hider_path.name}")

    env_config = TagEnvConfig(
        layout=args.layout,
        enable_sprint=args.enable_sprint,
        hider_speed_mult=args.hider_speed_mult,
        sprint_speed_mult=args.sprint_speed_mult,
    )
    env = SingleTagEnv(config=env_config)
    obs_dim, act_dim = env.obs_dim, env.act_dim

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    seeker = load_policy(str(seeker_path), obs_dim, act_dim)
    hider = load_policy(str(hider_path), obs_dim, act_dim)

    trajectory = run_episode(seeker, hider, env_config)
    final = trajectory[-1]
    tagged = final.get('tagged', False)
    print(f"Result: {'Tagged' if tagged else 'Survived'} in {final['step']} steps")

    title = args.title or run_dir.parent.name
    output = args.output or str(run_dir / "episode.mp4")
    create_mp4(trajectory, title, output, env_config, enable_sprint=args.enable_sprint)


if __name__ == "__main__":
    main()
