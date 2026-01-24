#!/usr/bin/env python3
"""
Outcome visualization for trained tag agents.

Generates:
1. Agent trajectory plots showing movement patterns
2. Win rate statistics and charts
3. Episode replay data for Godot visualization
4. Optional animated MP4s of gameplay
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle
    from matplotlib.collections import LineCollection
    import matplotlib.animation as animation
except ImportError:
    print("matplotlib required. Install with: pip install matplotlib", file=sys.stderr)
    sys.exit(1)

try:
    import torch
except ImportError:
    print("PyTorch required. Install with: pip install torch", file=sys.stderr)
    sys.exit(1)

from tag_env import SingleTagEnv, TagEnvConfig
from ppo import PPOConfig, PPOAgent


class OutcomeVisualizer:
    """Visualizes outcomes from trained tag agents."""

    def __init__(self, seeker_policy_path: str, hider_policy_path: str,
                 output_dir: str = "trainer/visualizations"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.env = SingleTagEnv()

        # Load policies
        ppo_cfg = PPOConfig(obs_dim=self.env.obs_dim, act_dim=self.env.act_dim)
        self.policies = {
            'seeker': PPOAgent(ppo_cfg),
            'hider': PPOAgent(ppo_cfg),
        }

        if os.path.exists(seeker_policy_path):
            self.policies['seeker'].load_policy(seeker_policy_path)
            print(f"Loaded seeker policy: {seeker_policy_path}")
        else:
            print(f"Warning: Seeker policy not found at {seeker_policy_path}")

        if os.path.exists(hider_policy_path):
            self.policies['hider'].load_policy(hider_policy_path)
            print(f"Loaded hider policy: {hider_policy_path}")
        else:
            print(f"Warning: Hider policy not found at {hider_policy_path}")

    def run_episode(self) -> Dict[str, Any]:
        """Run a single episode and collect trajectory data."""
        obs = self.env.reset()
        state = self.env.get_state()

        trajectory = {
            'seeker_positions': [state['positions'][state['seeker_idx']].tolist()],
            'hider_positions': [state['positions'][1 - state['seeker_idx']].tolist()],
            'timestamps': [0.0],
            'distances': [],
            'tagged': False,
            'duration': 0.0,
        }

        done = False
        while not done:
            # Get actions from policies
            actions = {}
            for role in ['seeker', 'hider']:
                action, _, _ = self.policies[role].act(obs[role])
                actions[role] = action

            obs, rewards, done, info = self.env.step(actions)
            state = self.env.get_state()

            seeker_idx = state['seeker_idx']
            trajectory['seeker_positions'].append(state['positions'][seeker_idx].tolist())
            trajectory['hider_positions'].append(state['positions'][1 - seeker_idx].tolist())
            trajectory['timestamps'].append(state['time_elapsed'])
            trajectory['distances'].append(info['distances'])

        trajectory['tagged'] = info['tagged']
        trajectory['duration'] = state['time_elapsed']

        return trajectory

    def evaluate(self, num_episodes: int = 50) -> Dict[str, Any]:
        """Run multiple episodes and compute statistics."""
        print(f"\nRunning {num_episodes} evaluation episodes...")

        trajectories = []
        seeker_wins = 0
        hider_wins = 0
        durations = []

        for i in range(num_episodes):
            traj = self.run_episode()
            trajectories.append(traj)

            if traj['tagged']:
                seeker_wins += 1
            else:
                hider_wins += 1

            durations.append(traj['duration'])

            if (i + 1) % 10 == 0:
                print(f"  Episode {i + 1}/{num_episodes}")

        stats = {
            'num_episodes': num_episodes,
            'seeker_wins': seeker_wins,
            'hider_wins': hider_wins,
            'seeker_win_rate': seeker_wins / num_episodes,
            'hider_win_rate': hider_wins / num_episodes,
            'avg_duration': np.mean(durations),
            'min_duration': np.min(durations),
            'max_duration': np.max(durations),
            'std_duration': np.std(durations),
        }

        print(f"\nResults:")
        print(f"  Seeker win rate: {stats['seeker_win_rate']:.1%}")
        print(f"  Hider win rate: {stats['hider_win_rate']:.1%}")
        print(f"  Avg episode duration: {stats['avg_duration']:.2f}s")

        return {'trajectories': trajectories, 'stats': stats}

    def plot_trajectory(self, trajectory: Dict[str, Any], filename: str):
        """Plot a single episode trajectory."""
        fig, ax = plt.subplots(figsize=(8, 8))

        arena_half = 15.0
        arena = Rectangle((-arena_half, -arena_half), arena_half * 2, arena_half * 2,
                         fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(arena)

        # Convert to numpy arrays
        seeker_pos = np.array(trajectory['seeker_positions'])
        hider_pos = np.array(trajectory['hider_positions'])

        # Plot trajectories with color gradient (start=light, end=dark)
        n_points = len(seeker_pos)
        colors_seeker = plt.cm.Reds(np.linspace(0.3, 1.0, n_points - 1))
        colors_hider = plt.cm.Blues(np.linspace(0.3, 1.0, n_points - 1))

        # Plot seeker trajectory
        for i in range(n_points - 1):
            ax.plot(seeker_pos[i:i+2, 0], seeker_pos[i:i+2, 1],
                   color=colors_seeker[i], linewidth=2)

        # Plot hider trajectory
        for i in range(n_points - 1):
            ax.plot(hider_pos[i:i+2, 0], hider_pos[i:i+2, 1],
                   color=colors_hider[i], linewidth=2)

        # Mark start positions
        ax.scatter(seeker_pos[0, 0], seeker_pos[0, 1], c='red', s=100,
                  marker='o', zorder=5, label='Seeker start')
        ax.scatter(hider_pos[0, 0], hider_pos[0, 1], c='blue', s=100,
                  marker='o', zorder=5, label='Hider start')

        # Mark end positions
        ax.scatter(seeker_pos[-1, 0], seeker_pos[-1, 1], c='darkred', s=150,
                  marker='*', zorder=5, label='Seeker end')
        ax.scatter(hider_pos[-1, 0], hider_pos[-1, 1], c='darkblue', s=150,
                  marker='*', zorder=5, label='Hider end')

        # Add outcome annotation
        outcome = "TAGGED" if trajectory['tagged'] else "ESCAPED"
        color = 'red' if trajectory['tagged'] else 'blue'
        ax.text(0, arena_half + 1, f"Outcome: {outcome} ({trajectory['duration']:.1f}s)",
               ha='center', fontsize=12, fontweight='bold', color=color)

        ax.set_xlim(-arena_half - 2, arena_half + 2)
        ax.set_ylim(-arena_half - 2, arena_half + 2)
        ax.set_aspect('equal')
        ax.legend(loc='upper left')
        ax.set_title('Episode Trajectory')
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(filename, dpi=150)
        plt.close(fig)

    def plot_multi_trajectory(self, trajectories: List[Dict[str, Any]], filename: str,
                              max_episodes: int = 10):
        """Plot multiple trajectories overlaid."""
        fig, ax = plt.subplots(figsize=(10, 10))

        arena_half = 15.0
        arena = Rectangle((-arena_half, -arena_half), arena_half * 2, arena_half * 2,
                         fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(arena)

        # Separate by outcome
        tagged_trajs = [t for t in trajectories if t['tagged']][:max_episodes // 2]
        escaped_trajs = [t for t in trajectories if not t['tagged']][:max_episodes // 2]

        # Plot tagged episodes (seeker wins) in red tones
        for traj in tagged_trajs:
            seeker_pos = np.array(traj['seeker_positions'])
            hider_pos = np.array(traj['hider_positions'])
            ax.plot(seeker_pos[:, 0], seeker_pos[:, 1], 'r-', alpha=0.3, linewidth=1)
            ax.plot(hider_pos[:, 0], hider_pos[:, 1], 'b-', alpha=0.3, linewidth=1)

        # Plot escaped episodes (hider wins) in blue tones
        for traj in escaped_trajs:
            seeker_pos = np.array(traj['seeker_positions'])
            hider_pos = np.array(traj['hider_positions'])
            ax.plot(seeker_pos[:, 0], seeker_pos[:, 1], 'orange', alpha=0.3, linewidth=1)
            ax.plot(hider_pos[:, 0], hider_pos[:, 1], 'cyan', alpha=0.3, linewidth=1)

        # Legend
        ax.plot([], [], 'r-', label=f'Seeker (tagged: {len(tagged_trajs)})', alpha=0.5)
        ax.plot([], [], 'b-', label=f'Hider (tagged: {len(tagged_trajs)})', alpha=0.5)
        ax.plot([], [], 'orange', label=f'Seeker (escaped: {len(escaped_trajs)})', alpha=0.5)
        ax.plot([], [], 'cyan', label=f'Hider (escaped: {len(escaped_trajs)})', alpha=0.5)

        ax.set_xlim(-arena_half - 2, arena_half + 2)
        ax.set_ylim(-arena_half - 2, arena_half + 2)
        ax.set_aspect('equal')
        ax.legend(loc='upper left')
        ax.set_title(f'Multi-Episode Trajectories (n={len(tagged_trajs) + len(escaped_trajs)})')
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(filename, dpi=150)
        plt.close(fig)

    def plot_statistics(self, stats: Dict[str, Any], filename: str):
        """Plot evaluation statistics."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Win rate pie chart
        ax = axes[0]
        labels = ['Seeker Wins', 'Hider Wins']
        sizes = [stats['seeker_wins'], stats['hider_wins']]
        colors = ['#ff6b6b', '#4dabf7']
        explode = (0.05, 0.05)

        ax.pie(sizes, explode=explode, labels=labels, colors=colors,
               autopct='%1.1f%%', startangle=90)
        ax.set_title('Win Distribution')

        # Duration histogram
        ax = axes[1]
        ax.text(0.5, 0.5, f"Avg: {stats['avg_duration']:.2f}s\n"
                         f"Min: {stats['min_duration']:.2f}s\n"
                         f"Max: {stats['max_duration']:.2f}s\n"
                         f"Std: {stats['std_duration']:.2f}s",
               ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_title('Episode Duration Stats')
        ax.axis('off')

        # Summary metrics
        ax = axes[2]
        metrics_text = (
            f"Total Episodes: {stats['num_episodes']}\n\n"
            f"Seeker Win Rate: {stats['seeker_win_rate']:.1%}\n"
            f"Hider Win Rate: {stats['hider_win_rate']:.1%}\n\n"
            f"Average Duration: {stats['avg_duration']:.2f}s"
        )
        ax.text(0.5, 0.5, metrics_text, ha='center', va='center',
               fontsize=14, transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_title('Summary')
        ax.axis('off')

        fig.suptitle('Evaluation Results', fontsize=16, fontweight='bold')
        fig.tight_layout()
        fig.savefig(filename, dpi=150)
        plt.close(fig)

    def create_animation(self, trajectory: Dict[str, Any], filename: str, fps: int = 30):
        """Create an animated MP4 of a single episode."""
        fig, ax = plt.subplots(figsize=(8, 8))

        arena_half = 15.0

        def init():
            ax.clear()
            arena = Rectangle((-arena_half, -arena_half), arena_half * 2, arena_half * 2,
                             fill=False, edgecolor='black', linewidth=2)
            ax.add_patch(arena)
            ax.set_xlim(-arena_half - 2, arena_half + 2)
            ax.set_ylim(-arena_half - 2, arena_half + 2)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            return []

        seeker_pos = np.array(trajectory['seeker_positions'])
        hider_pos = np.array(trajectory['hider_positions'])
        n_frames = len(seeker_pos)

        seeker_circle = plt.Circle((0, 0), 0.5, color='red', zorder=5)
        hider_circle = plt.Circle((0, 0), 0.5, color='blue', zorder=5)
        seeker_trail, = ax.plot([], [], 'r-', alpha=0.3, linewidth=1)
        hider_trail, = ax.plot([], [], 'b-', alpha=0.3, linewidth=1)
        time_text = ax.text(0, arena_half + 1, '', ha='center', fontsize=12)

        ax.add_patch(seeker_circle)
        ax.add_patch(hider_circle)

        def animate(frame):
            # Update positions
            seeker_circle.center = seeker_pos[frame]
            hider_circle.center = hider_pos[frame]

            # Update trails
            seeker_trail.set_data(seeker_pos[:frame+1, 0], seeker_pos[:frame+1, 1])
            hider_trail.set_data(hider_pos[:frame+1, 0], hider_pos[:frame+1, 1])

            # Update time
            t = trajectory['timestamps'][frame]
            time_text.set_text(f'Time: {t:.2f}s')

            return [seeker_circle, hider_circle, seeker_trail, hider_trail, time_text]

        ani = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=n_frames, interval=1000/fps, blit=True)

        ani.save(filename, writer='ffmpeg', fps=fps)
        plt.close(fig)
        print(f"Animation saved: {filename}")

    def export_for_godot(self, trajectory: Dict[str, Any], filename: str):
        """Export trajectory in JSONL format for Godot replay."""
        with open(filename, 'w') as f:
            # Episode start
            f.write(json.dumps({
                'type': 'episode_start',
                'ts': 0
            }) + '\n')

            seeker_pos = trajectory['seeker_positions']
            hider_pos = trajectory['hider_positions']
            timestamps = trajectory['timestamps']

            for i, (sp, hp, ts) in enumerate(zip(seeker_pos, hider_pos, timestamps)):
                # Seeker step
                f.write(json.dumps({
                    'type': 'step',
                    'agent': 'Seeker',
                    'pos': [sp[0], 0.0, sp[1]],  # Convert 2D to 3D (y=0)
                    'is_it': True,
                    'ts': int(ts * 1000)
                }) + '\n')

                # Hider step
                f.write(json.dumps({
                    'type': 'step',
                    'agent': 'Hider',
                    'pos': [hp[0], 0.0, hp[1]],
                    'is_it': False,
                    'ts': int(ts * 1000)
                }) + '\n')

            # Episode end
            if trajectory['tagged']:
                f.write(json.dumps({
                    'type': 'tag',
                    'attacker': 'Seeker',
                    'target': 'Hider',
                    'ts': int(timestamps[-1] * 1000)
                }) + '\n')

            f.write(json.dumps({
                'type': 'episode_end',
                'ts': int(timestamps[-1] * 1000)
            }) + '\n')

    def visualize(self, num_episodes: int = 50, create_anim: bool = False):
        """Run full visualization pipeline."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(self.output_dir, timestamp)
        os.makedirs(run_dir, exist_ok=True)

        # Run evaluation
        results = self.evaluate(num_episodes)
        trajectories = results['trajectories']
        stats = results['stats']

        # Save statistics
        stats_path = os.path.join(run_dir, "stats.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\nStatistics saved: {stats_path}")

        # Plot statistics
        self.plot_statistics(stats, os.path.join(run_dir, "stats.png"))
        print(f"Statistics plot saved")

        # Plot multi-trajectory overview
        self.plot_multi_trajectory(trajectories, os.path.join(run_dir, "trajectories_overview.png"))
        print(f"Trajectory overview saved")

        # Plot individual trajectories for interesting episodes
        # Find shortest (quick tag) and longest (close escape) episodes
        tagged_trajs = [t for t in trajectories if t['tagged']]
        escaped_trajs = [t for t in trajectories if not t['tagged']]

        if tagged_trajs:
            quickest = min(tagged_trajs, key=lambda t: t['duration'])
            self.plot_trajectory(quickest, os.path.join(run_dir, "quickest_tag.png"))
            self.export_for_godot(quickest, os.path.join(run_dir, "quickest_tag.jsonl"))
            print(f"Quickest tag episode saved")

            if create_anim:
                self.create_animation(quickest, os.path.join(run_dir, "quickest_tag.mp4"))

        if escaped_trajs:
            longest_escape = max(escaped_trajs, key=lambda t: t['duration'])
            self.plot_trajectory(longest_escape, os.path.join(run_dir, "best_escape.png"))
            self.export_for_godot(longest_escape, os.path.join(run_dir, "best_escape.jsonl"))
            print(f"Best escape episode saved")

            if create_anim:
                self.create_animation(longest_escape, os.path.join(run_dir, "best_escape.mp4"))

        # Export a typical episode
        if trajectories:
            # Pick episode closest to average duration
            avg_dur = stats['avg_duration']
            typical = min(trajectories, key=lambda t: abs(t['duration'] - avg_dur))
            self.plot_trajectory(typical, os.path.join(run_dir, "typical_episode.png"))
            self.export_for_godot(typical, os.path.join(run_dir, "typical_episode.jsonl"))
            print(f"Typical episode saved")

            if create_anim:
                self.create_animation(typical, os.path.join(run_dir, "typical_episode.mp4"))

        print(f"\nAll visualizations saved to: {run_dir}")
        return run_dir


def main():
    parser = argparse.ArgumentParser(description="Visualize trained tag agent outcomes")
    parser.add_argument("--seeker-policy", type=str, default="trainer/policy_seeker.pt",
                        help="Path to seeker policy")
    parser.add_argument("--hider-policy", type=str, default="trainer/policy_hider.pt",
                        help="Path to hider policy")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Number of evaluation episodes")
    parser.add_argument("--output-dir", type=str, default="trainer/visualizations",
                        help="Output directory")
    parser.add_argument("--animate", action="store_true",
                        help="Create animated MP4 videos (requires ffmpeg)")

    args = parser.parse_args()

    visualizer = OutcomeVisualizer(
        seeker_policy_path=args.seeker_policy,
        hider_policy_path=args.hider_policy,
        output_dir=args.output_dir,
    )

    visualizer.visualize(num_episodes=args.episodes, create_anim=args.animate)


if __name__ == "__main__":
    main()
