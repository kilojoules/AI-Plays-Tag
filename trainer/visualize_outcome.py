#!/usr/bin/env python3
"""
Outcome visualization for trained tag agents.

Generates:
1. Agent trajectory plots showing movement patterns
2. Win rate statistics and charts
3. Optional animated MP4s of gameplay
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

from tag_env import SingleTagEnv, TagEnvConfig, Obstacle, SafeZone, LAYOUTS
from ppo import PPOConfig, PPOAgent


class OutcomeVisualizer:
    """Visualizes outcomes from trained tag agents."""

    def __init__(self, seeker_policy_path: str, hider_policy_path: str,
                 output_dir: str = "trainer/visualizations",
                 layout: str = "empty"):
        self.output_dir = output_dir
        self.layout = layout
        os.makedirs(output_dir, exist_ok=True)

        # Create env config with layout
        env_config = TagEnvConfig(layout=layout)
        self.env = SingleTagEnv(config=env_config)

        # Store layout info for rendering
        layout_data = LAYOUTS.get(layout, LAYOUTS['empty'])
        self.obstacles = layout_data['obstacles']
        self.safe_zone = layout_data['safe_zone']

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
            'obstacles': self.obstacles,
            'safe_zone': self.safe_zone,
            'safe_zone_states': [{
                'time': state['safe_zone_time'],
                'exhausted': state['safe_zone_exhausted'],
                'cooldown': state['safe_zone_cooldown'],
            }],
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
            trajectory['safe_zone_states'].append({
                'time': state['safe_zone_time'],
                'exhausted': state['safe_zone_exhausted'],
                'cooldown': state['safe_zone_cooldown'],
            })

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

    def _draw_arena_features(self, ax, trajectory: Optional[Dict[str, Any]] = None):
        """Draw obstacles and safe zone on the axis."""
        # Get obstacles and safe zone from trajectory or instance
        obstacles = trajectory.get('obstacles', self.obstacles) if trajectory else self.obstacles
        safe_zone = trajectory.get('safe_zone', self.safe_zone) if trajectory else self.safe_zone

        # Draw obstacles as gray rectangles
        for obs in obstacles:
            rect = Rectangle(
                (obs.x - obs.half_width, obs.y - obs.half_height),
                obs.half_width * 2, obs.half_height * 2,
                fill=True, facecolor='gray', edgecolor='darkgray',
                linewidth=2, alpha=0.8, zorder=2
            )
            ax.add_patch(rect)

        # Draw safe zone as semi-transparent green circle
        if safe_zone is not None:
            circle = Circle(
                (safe_zone.x, safe_zone.y), safe_zone.radius,
                fill=True, facecolor='lightgreen', edgecolor='green',
                linewidth=2, alpha=0.4, zorder=1
            )
            ax.add_patch(circle)

    def plot_trajectory(self, trajectory: Dict[str, Any], filename: str):
        """Plot a single episode trajectory."""
        fig, ax = plt.subplots(figsize=(8, 8))

        arena_half = 15.0
        arena = Rectangle((-arena_half, -arena_half), arena_half * 2, arena_half * 2,
                         fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(arena)

        # Draw obstacles and safe zone
        self._draw_arena_features(ax, trajectory)

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
                   color=colors_seeker[i], linewidth=2, zorder=3)

        # Plot hider trajectory
        for i in range(n_points - 1):
            ax.plot(hider_pos[i:i+2, 0], hider_pos[i:i+2, 1],
                   color=colors_hider[i], linewidth=2, zorder=3)

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

        # Draw obstacles and safe zone (use first trajectory or instance data)
        self._draw_arena_features(ax, trajectories[0] if trajectories else None)

        # Separate by outcome
        tagged_trajs = [t for t in trajectories if t['tagged']][:max_episodes // 2]
        escaped_trajs = [t for t in trajectories if not t['tagged']][:max_episodes // 2]

        # Plot tagged episodes (seeker wins) in red tones
        for traj in tagged_trajs:
            seeker_pos = np.array(traj['seeker_positions'])
            hider_pos = np.array(traj['hider_positions'])
            ax.plot(seeker_pos[:, 0], seeker_pos[:, 1], 'r-', alpha=0.3, linewidth=1, zorder=3)
            ax.plot(hider_pos[:, 0], hider_pos[:, 1], 'b-', alpha=0.3, linewidth=1, zorder=3)

        # Plot escaped episodes (hider wins) in blue tones
        for traj in escaped_trajs:
            seeker_pos = np.array(traj['seeker_positions'])
            hider_pos = np.array(traj['hider_positions'])
            ax.plot(seeker_pos[:, 0], seeker_pos[:, 1], 'orange', alpha=0.3, linewidth=1, zorder=3)
            ax.plot(hider_pos[:, 0], hider_pos[:, 1], 'cyan', alpha=0.3, linewidth=1, zorder=3)

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
        seeker_pos = np.array(trajectory['seeker_positions'])
        hider_pos = np.array(trajectory['hider_positions'])
        n_frames = len(seeker_pos)

        # Get obstacles and safe zone from trajectory
        obstacles = trajectory.get('obstacles', self.obstacles)
        safe_zone = trajectory.get('safe_zone', self.safe_zone)
        safe_zone_states = trajectory.get('safe_zone_states', [])

        # Debug: print trajectory info
        print(f"  Creating animation with {n_frames} frames")
        print(f"  Seeker start: ({seeker_pos[0][0]:.1f}, {seeker_pos[0][1]:.1f})")
        print(f"  Hider start: ({hider_pos[0][0]:.1f}, {hider_pos[0][1]:.1f})")
        print(f"  Seeker end: ({seeker_pos[-1][0]:.1f}, {seeker_pos[-1][1]:.1f})")
        print(f"  Hider end: ({hider_pos[-1][0]:.1f}, {hider_pos[-1][1]:.1f})")
        if obstacles:
            print(f"  Obstacles: {len(obstacles)}")
        if safe_zone:
            print(f"  Safe zone at ({safe_zone.x}, {safe_zone.y}), radius {safe_zone.radius}")

        arena_half = 15.0
        fig, ax = plt.subplots(figsize=(10, 10))

        def animate(frame):
            ax.clear()

            # Draw arena
            arena = Rectangle((-arena_half, -arena_half), arena_half * 2, arena_half * 2,
                             fill=False, edgecolor='black', linewidth=2)
            ax.add_patch(arena)

            # Draw obstacles as gray rectangles
            for obs in obstacles:
                rect = Rectangle(
                    (obs.x - obs.half_width, obs.y - obs.half_height),
                    obs.half_width * 2, obs.half_height * 2,
                    fill=True, facecolor='gray', edgecolor='darkgray',
                    linewidth=2, alpha=0.8, zorder=2
                )
                ax.add_patch(rect)

            # Draw safe zone with pulsing effect and exhaustion state
            if safe_zone is not None:
                # Determine safe zone color based on state
                sz_state = safe_zone_states[frame] if frame < len(safe_zone_states) else {}
                is_exhausted = sz_state.get('exhausted', False)

                # Pulsing effect: vary alpha and radius slightly
                t = trajectory['timestamps'][frame]
                pulse = 0.1 * np.sin(t * 4)  # Pulse frequency

                if is_exhausted:
                    # Red/orange when exhausted (no protection)
                    zone_color = 'lightsalmon'
                    edge_color = 'red'
                    base_alpha = 0.3
                else:
                    # Green when protected
                    zone_color = 'lightgreen'
                    edge_color = 'green'
                    base_alpha = 0.4

                circle = Circle(
                    (safe_zone.x, safe_zone.y), safe_zone.radius * (1 + pulse * 0.1),
                    fill=True, facecolor=zone_color, edgecolor=edge_color,
                    linewidth=2, alpha=base_alpha + pulse * 0.1, zorder=1
                )
                ax.add_patch(circle)

            # Draw trails (path history)
            if frame > 0:
                ax.plot(seeker_pos[:frame+1, 0], seeker_pos[:frame+1, 1],
                       'r-', alpha=0.4, linewidth=2, label='Seeker path', zorder=3)
                ax.plot(hider_pos[:frame+1, 0], hider_pos[:frame+1, 1],
                       'b-', alpha=0.4, linewidth=2, label='Hider path', zorder=3)

            # Draw agents as circles
            seeker = plt.Circle(seeker_pos[frame], 0.8, color='red', zorder=5, label='Seeker')
            hider = plt.Circle(hider_pos[frame], 0.8, color='blue', zorder=5, label='Hider')
            ax.add_patch(seeker)
            ax.add_patch(hider)

            # Add agent labels
            ax.annotate('S', seeker_pos[frame], ha='center', va='center',
                       fontsize=12, fontweight='bold', color='white', zorder=6)
            ax.annotate('H', hider_pos[frame], ha='center', va='center',
                       fontsize=12, fontweight='bold', color='white', zorder=6)

            # Time and frame info
            t = trajectory['timestamps'][frame]
            outcome = "TAGGED!" if trajectory['tagged'] else "ESCAPED!"
            status = outcome if frame == n_frames - 1 else f"Time: {t:.2f}s"
            ax.set_title(f'Tag Game - {status}', fontsize=14, fontweight='bold')

            # Set axis properties
            ax.set_xlim(-arena_half - 2, arena_half + 2)
            ax.set_ylim(-arena_half - 2, arena_half + 2)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')

            # Legend
            ax.legend(loc='upper right')

        ani = animation.FuncAnimation(fig, animate, frames=n_frames,
                                       interval=1000/fps, repeat=False)

        ani.save(filename, writer='ffmpeg', fps=fps, dpi=100)
        plt.close(fig)
        print(f"Animation saved: {filename}")

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
            print(f"Quickest tag episode saved")

            if create_anim:
                self.create_animation(quickest, os.path.join(run_dir, "quickest_tag.mp4"))

        if escaped_trajs:
            longest_escape = max(escaped_trajs, key=lambda t: t['duration'])
            self.plot_trajectory(longest_escape, os.path.join(run_dir, "best_escape.png"))
            print(f"Best escape episode saved")

            if create_anim:
                self.create_animation(longest_escape, os.path.join(run_dir, "best_escape.mp4"))

        # Export a typical episode
        if trajectories:
            # Pick episode closest to average duration
            avg_dur = stats['avg_duration']
            typical = min(trajectories, key=lambda t: abs(t['duration'] - avg_dur))
            self.plot_trajectory(typical, os.path.join(run_dir, "typical_episode.png"))
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
    parser.add_argument("--layout", type=str, default="empty",
                        choices=["empty", "four_corners", "central_cross"],
                        help="Arena layout (default: empty)")

    args = parser.parse_args()

    visualizer = OutcomeVisualizer(
        seeker_policy_path=args.seeker_policy,
        hider_policy_path=args.hider_policy,
        output_dir=args.output_dir,
        layout=args.layout,
    )

    visualizer.visualize(num_episodes=args.episodes, create_anim=args.animate)


if __name__ == "__main__":
    main()
