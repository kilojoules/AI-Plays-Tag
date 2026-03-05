#!/usr/bin/env python3
"""Render header animation GIF from saved episode trajectory data."""
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import PillowWriter

sys.path.insert(0, str(Path(__file__).parent))

ARENA_HALF = 15.0
TAG_DIST = 1.8

# four_corners layout (x, y, half_w, half_h)
OBSTACLES = [
    (-8.0, 8.0, 1.5, 1.5),   # Top-left
    (8.0, 8.0, 1.5, 1.5),    # Top-right
    (-3.0, -8.0, 1.5, 1.5),  # Bottom-left
    (3.0, -8.0, 1.5, 1.5),   # Bottom-right
]
SAFE_ZONE = (0.0, 0.0, 2.5)


def render_episode(episode_file, output_file, fps=20):
    with open(episode_file) as f:
        data = json.load(f)

    sp = np.array(data["seeker_positions"])
    hp = np.array(data["hider_positions"])
    n_steps = len(sp)
    tagged = data["tagged"]
    config = data["config"]

    # Dark theme
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(7, 7), dpi=100)

    # Config info for title — parse string values
    hsm_str = config["hsm"]  # e.g. "HSM115"
    hsm_val = int(hsm_str.replace("HSM", "")) / 100  # 1.15
    A_str = config["A"]  # e.g. "A10"
    A_pct = int(A_str.replace("A", ""))  # 10
    sampling = config["sampling"]

    def draw_frame(step):
        ax.clear()
        ax.set_xlim(-ARENA_HALF - 1, ARENA_HALF + 1)
        ax.set_ylim(-ARENA_HALF - 1, ARENA_HALF + 1)
        ax.set_aspect("equal")
        ax.set_facecolor("#0d1117")
        fig.set_facecolor("#0d1117")

        # Arena boundary
        arena = patches.Rectangle(
            (-ARENA_HALF, -ARENA_HALF), 2 * ARENA_HALF, 2 * ARENA_HALF,
            linewidth=1.5, edgecolor="#30363d", facecolor="none"
        )
        ax.add_patch(arena)

        # Obstacles (center coords -> corner coords for Rectangle)
        for ox, oy, hw, hh in OBSTACLES:
            rect = patches.Rectangle(
                (ox - hw, oy - hh), 2 * hw, 2 * hh,
                facecolor="#21262d", edgecolor="#30363d", linewidth=0.8
            )
            ax.add_patch(rect)

        # Safe zone
        safe = plt.Circle((SAFE_ZONE[0], SAFE_ZONE[1]), SAFE_ZONE[2],
                           facecolor="none", edgecolor="#238636",
                           linewidth=1.5, alpha=0.6)
        ax.add_patch(safe)

        # Trajectory trails (fading)
        trail_len = min(step + 1, 30)
        if trail_len > 1:
            s_trail = sp[max(0, step + 1 - trail_len):step + 1]
            h_trail = hp[max(0, step + 1 - trail_len):step + 1]
            alphas = np.linspace(0.05, 0.4, len(s_trail))
            for k in range(len(s_trail) - 1):
                ax.plot(s_trail[k:k+2, 0], s_trail[k:k+2, 1],
                        color="#f85149", alpha=alphas[k], linewidth=1.5)
                ax.plot(h_trail[k:k+2, 0], h_trail[k:k+2, 1],
                        color="#58a6ff", alpha=alphas[k], linewidth=1.5)

        # Distance line
        sx, sy = sp[step]
        hx, hy = hp[step]
        dist = np.sqrt((sx - hx)**2 + (sy - hy)**2)
        ax.plot([sx, hx], [sy, hy], color="#484f58", linewidth=0.8,
                linestyle="--", alpha=0.5)

        # Agents with glow
        for pos, color, label in [(sp[step], "#f85149", "S"), (hp[step], "#58a6ff", "H")]:
            # Glow
            glow = plt.Circle(pos, 1.2, facecolor=color, alpha=0.15)
            ax.add_patch(glow)
            glow2 = plt.Circle(pos, 0.8, facecolor=color, alpha=0.25)
            ax.add_patch(glow2)
            # Agent
            agent = plt.Circle(pos, 0.5, facecolor=color, edgecolor="white",
                               linewidth=1.2, alpha=0.95, zorder=10)
            ax.add_patch(agent)
            ax.text(pos[0], pos[1], label, ha="center", va="center",
                    fontsize=8, fontweight="bold", color="white", zorder=11)

        # Tag flash on final frame
        if step == n_steps - 1 and tagged:
            flash = plt.Circle(sp[step], 2.0, facecolor="#ffa657",
                               alpha=0.4, zorder=9)
            ax.add_patch(flash)
            ax.text(0, ARENA_HALF + 0.3, "TAGGED!", ha="center",
                    fontsize=14, fontweight="bold", color="#ffa657",
                    zorder=12)

        # HUD
        ax.text(-ARENA_HALF + 0.3, -ARENA_HALF - 0.5,
                f"Step {step}/{n_steps}",
                fontsize=9, color="#8b949e", va="top")
        ax.text(ARENA_HALF - 0.3, -ARENA_HALF - 0.5,
                f"d={dist:.1f}",
                fontsize=9, color="#8b949e", va="top", ha="right")

        # Title
        title = f"Zoo-Trained Tag  |  Hider Speed {hsm_val}x  |  A={A_pct}% {sampling.replace('_', '-').title()}"
        ax.set_title(title, fontsize=10, color="#c9d1d9", pad=8)

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Render
    print(f"Rendering {n_steps} frames to {output_file}...")
    writer = PillowWriter(fps=fps)
    with writer.saving(fig, str(output_file), dpi=100):
        for step in range(n_steps):
            draw_frame(step)
            writer.grab_frame()
            if (step + 1) % 30 == 0:
                print(f"  frame {step + 1}/{n_steps}")

        # Hold last frame for 1 second
        for _ in range(fps):
            writer.grab_frame()

    plt.close()
    print(f"Done: {output_file} ({Path(output_file).stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    base = Path("experiments/results/zoo_asweep/best_episodes")

    # Render episode 1 (wide chase, score 100.9)
    render_episode(
        base / "episode_STP01_HSM115_A10_uniform_s0_1.json",
        "docs/header_animation_ep1.gif",
    )

    # Render episode 3 (interior play, score 93.7)
    render_episode(
        base / "episode_STP01_HSM115_A30_uniform_s0_3.json",
        "docs/header_animation_ep3.gif",
    )
