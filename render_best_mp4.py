#!/usr/bin/env python3
"""Find the most visually interesting episode from a trained config and render as MP4."""
import sys
import json
from pathlib import Path

sys.path.insert(0, "/work/users/juqu/AI-Plays-Tag")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, FFMpegWriter

from trainer.tag_env import VecTagEnv, TagEnvConfig, LAYOUTS
from trainer.ppo import PPOAgent, PPOConfig


def load_policy(path, obs_dim, act_dim):
    cfg = PPOConfig(obs_dim=obs_dim, act_dim=act_dim)
    policy = PPOAgent(cfg)
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    policy.pi.load_state_dict(ckpt["pi"])
    policy.vf.load_state_dict(ckpt["vf"])
    return policy


def batch_act(policy, obs_batch):
    with torch.no_grad():
        x = torch.as_tensor(obs_batch, dtype=torch.float32)
        logits = policy.pi(x)
        mean, log_std = torch.chunk(logits, 2, dim=-1)
        log_std = torch.clamp(log_std, -2.0, 1.5)
        std = torch.exp(log_std)
        action = torch.tanh(torch.distributions.Normal(mean, std).sample())
        return action.cpu().numpy()


def simulate_and_score(seeker_policy, hider_policy, hsm, num_envs=200):
    """Simulate episodes and return them sorted by visual interest score."""
    cfg = TagEnvConfig(layout="four_corners", hider_speed_mult=hsm)
    env = VecTagEnv(num_envs=num_envs, config=cfg)
    obs = env.reset()

    max_steps = int(cfg.time_limit / (cfg.dt * cfg.steps_per_action))
    arena_half = cfg.arena_half

    all_seeker_pos = [[] for _ in range(num_envs)]
    all_hider_pos = [[] for _ in range(num_envs)]
    active = np.ones(num_envs, dtype=bool)
    episode_lengths = np.zeros(num_envs, dtype=int)
    episode_tagged = np.zeros(num_envs, dtype=bool)

    for step_i in range(max_steps):
        eids = np.arange(num_envs)
        seeker_idx = env.seeker_idx
        hider_idx = 1 - seeker_idx
        seeker_pos = env.positions[eids, seeker_idx].copy()
        hider_pos = env.positions[eids, hider_idx].copy()

        for i in range(num_envs):
            if active[i]:
                all_seeker_pos[i].append(seeker_pos[i].tolist())
                all_hider_pos[i].append(hider_pos[i].tolist())

        seeker_actions = batch_act(seeker_policy, obs["seeker"])
        hider_actions = batch_act(hider_policy, obs["hider"])
        obs, rewards, dones, infos = env.step({"seeker": seeker_actions, "hider": hider_actions})

        newly_done = dones & active
        if np.any(newly_done):
            for i in np.where(newly_done)[0]:
                episode_lengths[i] = step_i + 1
                episode_tagged[i] = infos["tagged"][i]
            active[newly_done] = False

        if not np.any(active):
            break

    for i in range(num_envs):
        if active[i]:
            episode_lengths[i] = max_steps
            episode_tagged[i] = False

    # Score episodes
    episodes = []
    for i in range(num_envs):
        sp = np.array(all_seeker_pos[i])
        hp = np.array(all_hider_pos[i])
        T = len(hp)
        if T < 10:
            continue

        hider_diffs = np.diff(hp, axis=0)
        hider_dist = np.sum(np.linalg.norm(hider_diffs, axis=1))
        wall_dist = arena_half - np.abs(hp)
        in_corner = (wall_dist[:, 0] < 3.0) & (wall_dist[:, 1] < 3.0)
        corner_pct = np.mean(in_corner) * 100
        in_interior = np.min(wall_dist, axis=1) > 5.0
        interior_pct = np.mean(in_interior) * 100
        if len(hider_diffs) > 1:
            angles = np.arctan2(hider_diffs[:, 1], hider_diffs[:, 0])
            angle_diffs = np.abs(np.diff(angles))
            angle_diffs = np.minimum(angle_diffs, 2 * np.pi - angle_diffs)
            direction_changes = int(np.sum(angle_diffs > np.radians(45)))
        else:
            direction_changes = 0
        inter_dist = np.linalg.norm(sp - hp, axis=1)
        dist_variance = float(np.var(inter_dist))

        episodes.append({
            "index": i,
            "length": int(episode_lengths[i]),
            "tagged": bool(episode_tagged[i]),
            "hider_dist": float(hider_dist),
            "corner_pct": float(corner_pct),
            "interior_pct": float(interior_pct),
            "direction_changes": direction_changes,
            "dist_variance": dist_variance,
            "seeker_positions": all_seeker_pos[i],
            "hider_positions": all_hider_pos[i],
        })

    if not episodes:
        return []

    # Normalize and score
    metrics_keys = ["hider_dist", "corner_pct", "interior_pct", "direction_changes", "dist_variance"]
    ranges = {}
    for key in metrics_keys:
        vals = [e[key] for e in episodes]
        mn, mx = min(vals), max(vals)
        ranges[key] = (mn, mx - mn if mx > mn else 1.0)

    for ep in episodes:
        nd = (ep["hider_dist"] - ranges["hider_dist"][0]) / ranges["hider_dist"][1] * 100
        nc = (ep["corner_pct"] - ranges["corner_pct"][0]) / ranges["corner_pct"][1] * 100
        ni = (ep["interior_pct"] - ranges["interior_pct"][0]) / ranges["interior_pct"][1] * 100
        ndc = (ep["direction_changes"] - ranges["direction_changes"][0]) / ranges["direction_changes"][1] * 100
        ndv = (ep["dist_variance"] - ranges["dist_variance"][0]) / ranges["dist_variance"][1] * 100

        score = nd * 0.30 + (100 - nc) * 0.25 + ni * 0.15 + ndc * 0.15 + ndv * 0.15
        if ep["tagged"]:
            score += 20.0
        if 100 <= ep["length"] <= 180:
            score += 10.0
        ep["score"] = score

    episodes.sort(key=lambda e: e["score"], reverse=True)
    return episodes


def render_mp4(episode, output_path, title, hsm, fps=20):
    """Render episode as MP4 with dark theme."""
    sp = np.array(episode["seeker_positions"])
    hp = np.array(episode["hider_positions"])
    n_steps = len(sp)
    tagged = episode["tagged"]

    ARENA_HALF = 15.0
    OBSTACLES = [(-8.0, 8.0, 1.5, 1.5), (8.0, 8.0, 1.5, 1.5),
                 (-3.0, -8.0, 1.5, 1.5), (3.0, -8.0, 1.5, 1.5)]
    SAFE_ZONE = (0.0, 0.0, 2.5)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 8), dpi=120)

    def animate(step):
        ax.clear()
        ax.set_xlim(-ARENA_HALF - 1, ARENA_HALF + 1)
        ax.set_ylim(-ARENA_HALF - 1, ARENA_HALF + 1)
        ax.set_aspect("equal")
        ax.set_facecolor("#0d1117")
        fig.set_facecolor("#0d1117")

        # Arena boundary
        arena = patches.Rectangle(
            (-ARENA_HALF, -ARENA_HALF), 2 * ARENA_HALF, 2 * ARENA_HALF,
            linewidth=1.5, edgecolor="#30363d", facecolor="none")
        ax.add_patch(arena)

        # Obstacles
        for ox, oy, hw, hh in OBSTACLES:
            rect = patches.Rectangle(
                (ox - hw, oy - hh), 2 * hw, 2 * hh,
                facecolor="#21262d", edgecolor="#30363d", linewidth=0.8)
            ax.add_patch(rect)

        # Safe zone
        safe = plt.Circle((SAFE_ZONE[0], SAFE_ZONE[1]), SAFE_ZONE[2],
                          facecolor="none", edgecolor="#238636",
                          linewidth=1.5, alpha=0.6)
        ax.add_patch(safe)

        # Trails
        trail_len = min(step + 1, 30)
        if trail_len > 1:
            s_trail = sp[max(0, step + 1 - trail_len):step + 1]
            h_trail = hp[max(0, step + 1 - trail_len):step + 1]
            alphas = np.linspace(0.05, 0.4, len(s_trail))
            for k in range(len(s_trail) - 1):
                ax.plot(s_trail[k:k+2, 0], s_trail[k:k+2, 1],
                        color="#f85149", alpha=alphas[k], linewidth=2)
                ax.plot(h_trail[k:k+2, 0], h_trail[k:k+2, 1],
                        color="#58a6ff", alpha=alphas[k], linewidth=2)

        # Distance line
        sx, sy = sp[step]
        hx, hy = hp[step]
        dist = np.sqrt((sx - hx)**2 + (sy - hy)**2)
        ax.plot([sx, hx], [sy, hy], color="#484f58", linewidth=0.8,
                linestyle="--", alpha=0.5)

        # Agents with glow
        for pos, color, label in [(sp[step], "#f85149", "S"), (hp[step], "#58a6ff", "H")]:
            glow = plt.Circle(pos, 1.2, facecolor=color, alpha=0.15)
            ax.add_patch(glow)
            glow2 = plt.Circle(pos, 0.8, facecolor=color, alpha=0.25)
            ax.add_patch(glow2)
            agent = plt.Circle(pos, 0.5, facecolor=color, edgecolor="white",
                              linewidth=1.2, alpha=0.95, zorder=10)
            ax.add_patch(agent)
            ax.text(pos[0], pos[1], label, ha="center", va="center",
                   fontsize=9, fontweight="bold", color="white", zorder=11)

        # Tag flash
        if step == n_steps - 1 and tagged:
            flash = plt.Circle(sp[step], 2.5, facecolor="#ffa657", alpha=0.4, zorder=9)
            ax.add_patch(flash)
            ax.text(0, ARENA_HALF + 0.3, "TAGGED!", ha="center",
                   fontsize=16, fontweight="bold", color="#ffa657", zorder=12)

        # HUD
        ax.text(-ARENA_HALF + 0.3, -ARENA_HALF - 0.5,
                "Step %d/%d" % (step, n_steps - 1),
                fontsize=9, color="#8b949e", va="top")
        ax.text(ARENA_HALF - 0.3, -ARENA_HALF - 0.5,
                "d=%.1f" % dist,
                fontsize=9, color="#8b949e", va="top", ha="right")
        ax.set_title(title, fontsize=11, color="#c9d1d9", pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        return []

    # Total frames: episode + 1 second hold at end
    hold_frames = fps
    total_frames = n_steps + hold_frames

    def animate_with_hold(frame_idx):
        step = min(frame_idx, n_steps - 1)
        return animate(step)

    print("Rendering %d frames (%d episode + %d hold)..." % (total_frames, n_steps, hold_frames))
    anim = FuncAnimation(fig, animate_with_hold, frames=total_frames, interval=1000//fps, blit=False)
    writer = FFMpegWriter(fps=fps, bitrate=3000)
    anim.save(str(output_path), writer=writer)
    plt.close()
    print("Saved: %s (%.1f MB)" % (output_path, output_path.stat().st_size / 1e6))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Run directory with checkpoints/")
    parser.add_argument("--hsm", type=float, default=1.15, help="Hider speed multiplier")
    parser.add_argument("--num-episodes", type=int, default=200, help="Episodes to simulate")
    parser.add_argument("--output", default="best_episode.mp4", help="Output path")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    run_dir = Path(args.dir)
    ckpt_dir = run_dir / "checkpoints"
    seeker_ckpts = sorted(ckpt_dir.glob("seeker_*.pt"))
    hider_ckpts = sorted(ckpt_dir.glob("hider_*.pt"))
    seeker_path = seeker_ckpts[-1]
    hider_path = hider_ckpts[-1]

    print("Run dir:", run_dir)
    print("Seeker:", seeker_path.name)
    print("Hider:", hider_path.name)
    print("Hider speed mult:", args.hsm)

    cfg = TagEnvConfig(layout="four_corners", hider_speed_mult=args.hsm)
    from trainer.tag_env import SingleTagEnv
    env = SingleTagEnv(config=cfg)
    obs_dim, act_dim = env.obs_dim, env.act_dim

    seeker = load_policy(str(seeker_path), obs_dim, act_dim)
    hider = load_policy(str(hider_path), obs_dim, act_dim)

    print("Simulating %d episodes..." % args.num_episodes)
    episodes = simulate_and_score(seeker, hider, args.hsm, num_envs=args.num_episodes)

    tagged_eps = [e for e in episodes if e["tagged"]]
    survived_eps = [e for e in episodes if not e["tagged"]]
    print("Results: %d tagged, %d survived" % (len(tagged_eps), len(survived_eps)))

    # Show top 5
    print("\nTop 5 episodes:")
    print("%-5s %-8s %-5s %-5s %-10s %-7s %-8s" % ("Rank", "Score", "Len", "Tag", "HiderDist", "DirChg", "DistVar"))
    for i, ep in enumerate(episodes[:5]):
        print("%-5d %-8.1f %-5d %-5s %-10.1f %-7d %-8.1f" % (
            i+1, ep["score"], ep["length"], "Y" if ep["tagged"] else "N",
            ep["hider_dist"], ep["direction_changes"], ep["dist_variance"]))

    best = episodes[0]
    print("\nRendering best episode (score=%.1f, len=%d, tagged=%s)" % (
        best["score"], best["length"], best["tagged"]))

    title = args.title or "Zoo-Trained Tag  |  Hider Speed %.2fx" % args.hsm
    output = Path(args.output)
    render_mp4(best, output, title, args.hsm)


if __name__ == "__main__":
    main()
