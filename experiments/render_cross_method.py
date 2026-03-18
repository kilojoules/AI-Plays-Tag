#!/usr/bin/env python3
"""Render showcase GIFs for the cross-method gauntlet's most interesting matchups."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import PillowWriter

from trainer.tag_env import VecTagEnv, TagEnvConfig
from trainer.ppo import PPOAgent, PPOConfig
from trainer.sac import SACAgent, SACConfig

ARENA_HALF = 15.0
OBSTACLES = [
    (-8.0, 8.0, 1.5, 1.5),
    (8.0, 8.0, 1.5, 1.5),
    (-3.0, -8.0, 1.5, 1.5),
    (3.0, -8.0, 1.5, 1.5),
]
SAFE_ZONE = (0.0, 0.0, 2.5)
OBS_DIM = 87
ACT_DIM = 3

# Matchups to render: (seeker_label, hider_label, short_name, caption)
MATCHUPS = [
    # 1. Top seeker vs top hider -- the ultimate showdown (12% WR, hider dominates)
    ("fr_v2/R5_escalating/A0.75_sac",
     "fr_v2/R3_both_shaped/A0.50_sac",
     "ultimate_showdown",
     "Best Seeker vs Best Hider (both SAC)"),

    # 2. Perfectly balanced cross-method (50% WR)
    ("fr/R3_both_shaped/A0.50_sac",
     "selfplay/STP01_HSM120",
     "perfect_rivals",
     "FR/SAC Seeker vs Selfplay Hider (50/50)"),

    # 3. SAC vs SAC -- close intra-tier battle (70% WR)
    ("fr_v2/R5_escalating/A0.75_sac",
     "fr/R1_seeker_pursuit/A0.50_sac",
     "sac_mirror",
     "Optuna SAC vs FR SAC (seeker edge)"),

    # 4. Tier gap: SAC seeker vs zoo hider (dominance)
    ("reward/R2_hider_active/sac",
     "zoo/STP01_HSM100/A50_thompson_loss",
     "sac_vs_zoo",
     "Reward/SAC Seeker vs Zoo/PPO Hider"),

    # 5. Selfplay holding its own vs reward-shaped
    ("selfplay/STP01_HSM120",
     "reward/R2_hider_active/ppo",
     "selfplay_vs_shaped",
     "Selfplay Seeker vs Reward-Shaped Hider"),
]


def load_agent(path, algo):
    if algo == "ppo":
        cfg = PPOConfig(obs_dim=OBS_DIM, act_dim=ACT_DIM)
        policy = PPOAgent(cfg)
        ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
        policy.pi.load_state_dict(ckpt["pi"])
        policy.vf.load_state_dict(ckpt["vf"])

        def act(obs, p=policy):
            with torch.no_grad():
                x = torch.as_tensor(obs, dtype=torch.float32)
                logits = p.pi(x)
                mean, log_std = torch.chunk(logits, 2, dim=-1)
                log_std = torch.clamp(log_std, -2.0, 1.5)
                std = torch.exp(log_std)
                return torch.tanh(
                    torch.distributions.Normal(mean, std).sample()
                ).cpu().numpy()
        return act
    else:
        cfg = SACConfig(obs_dim=OBS_DIM, act_dim=ACT_DIM)
        agent = SACAgent(cfg)
        agent.load_policy(str(path))

        def act(obs, a=agent):
            with torch.no_grad():
                x = torch.as_tensor(obs, dtype=torch.float32)
                actions, _ = a.actor.sample(x)
                return actions.cpu().numpy()
        return act


def simulate_episodes(seeker_fn, hider_fn, num_envs=50):
    """Simulate and return trajectory data for all episodes."""
    cfg = TagEnvConfig(layout="four_corners", hider_speed_mult=1.15)
    env = VecTagEnv(num_envs=num_envs, config=cfg)
    obs = env.reset()
    max_steps = int(cfg.time_limit / (cfg.dt * cfg.steps_per_action))

    all_sp = [[] for _ in range(num_envs)]
    all_hp = [[] for _ in range(num_envs)]
    active = np.ones(num_envs, dtype=bool)
    ep_lengths = np.zeros(num_envs, dtype=int)
    ep_tagged = np.zeros(num_envs, dtype=bool)
    eids = np.arange(num_envs)

    for step in range(max_steps):
        si = env.seeker_idx
        hi = 1 - si
        s_pos = env.positions[eids, si].copy()
        h_pos = env.positions[eids, hi].copy()
        for i in range(num_envs):
            if active[i]:
                all_sp[i].append(s_pos[i].tolist())
                all_hp[i].append(h_pos[i].tolist())

        s_acts = seeker_fn(obs["seeker"])
        h_acts = hider_fn(obs["hider"])
        obs, _, dones, infos = env.step({"seeker": s_acts, "hider": h_acts})

        newly_done = dones & active
        if np.any(newly_done):
            for i in np.where(newly_done)[0]:
                ep_lengths[i] = step + 1
                ep_tagged[i] = infos["tagged"][i]
            active[newly_done] = False
        if not np.any(active):
            break

    for i in range(num_envs):
        if active[i]:
            ep_lengths[i] = max_steps

    episodes = []
    for i in range(num_envs):
        sp = np.array(all_sp[i])
        hp = np.array(all_hp[i])
        if len(hp) < 5:
            continue

        hider_dist_traveled = float(np.sum(np.linalg.norm(np.diff(hp, axis=0), axis=1)))
        wall_dist = ARENA_HALF - np.abs(hp)
        in_corner = (wall_dist[:, 0] < 3) & (wall_dist[:, 1] < 3)
        corner_pct = float(np.mean(in_corner))

        inter_dist = np.linalg.norm(sp - hp, axis=1)
        dist_var = float(np.var(inter_dist))

        diffs = np.diff(hp, axis=0)
        if len(diffs) > 1:
            angles = np.arctan2(diffs[:, 1], diffs[:, 0])
            adiffs = np.abs(np.diff(angles))
            adiffs = np.minimum(adiffs, 2 * np.pi - adiffs)
            dir_changes = int(np.sum(adiffs > np.radians(45)))
        else:
            dir_changes = 0

        # Score: prefer tags, dynamic play, interior movement, medium length
        score = 0.0
        score += hider_dist_traveled * 0.01
        score += (1 - corner_pct) * 30
        score += dir_changes * 0.5
        score += dist_var * 0.01
        if ep_tagged[i]:
            score += 25.0
        if 80 <= ep_lengths[i] <= 180:
            score += 15.0

        episodes.append({
            "index": i,
            "length": int(ep_lengths[i]),
            "tagged": bool(ep_tagged[i]),
            "score": score,
            "seeker_pos": all_sp[i],
            "hider_pos": all_hp[i],
        })

    return sorted(episodes, key=lambda e: e["score"], reverse=True)


def render_episode(episode, output_file, title, fps=20):
    """Render a single episode to GIF."""
    sp = np.array(episode["seeker_pos"])
    hp = np.array(episode["hider_pos"])
    n_steps = len(sp)
    tagged = episode["tagged"]

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(7, 7), dpi=100)

    def draw_frame(step):
        ax.clear()
        ax.set_xlim(-ARENA_HALF - 1, ARENA_HALF + 1)
        ax.set_ylim(-ARENA_HALF - 1, ARENA_HALF + 1)
        ax.set_aspect("equal")
        ax.set_facecolor("#0d1117")
        fig.set_facecolor("#0d1117")

        arena = patches.Rectangle(
            (-ARENA_HALF, -ARENA_HALF), 2 * ARENA_HALF, 2 * ARENA_HALF,
            linewidth=1.5, edgecolor="#30363d", facecolor="none")
        ax.add_patch(arena)

        for ox, oy, hw, hh in OBSTACLES:
            rect = patches.Rectangle(
                (ox - hw, oy - hh), 2 * hw, 2 * hh,
                facecolor="#21262d", edgecolor="#30363d", linewidth=0.8)
            ax.add_patch(rect)

        safe = plt.Circle((SAFE_ZONE[0], SAFE_ZONE[1]), SAFE_ZONE[2],
                           facecolor="none", edgecolor="#238636",
                           linewidth=1.5, alpha=0.6)
        ax.add_patch(safe)

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

        sx, sy = sp[step]
        hx, hy = hp[step]
        dist = np.sqrt((sx - hx)**2 + (sy - hy)**2)
        ax.plot([sx, hx], [sy, hy], color="#484f58", linewidth=0.8,
                linestyle="--", alpha=0.5)

        for pos, color, label in [(sp[step], "#f85149", "S"),
                                  (hp[step], "#58a6ff", "H")]:
            glow = plt.Circle(pos, 1.2, facecolor=color, alpha=0.15)
            ax.add_patch(glow)
            glow2 = plt.Circle(pos, 0.8, facecolor=color, alpha=0.25)
            ax.add_patch(glow2)
            agent = plt.Circle(pos, 0.5, facecolor=color, edgecolor="white",
                               linewidth=1.2, alpha=0.95, zorder=10)
            ax.add_patch(agent)
            ax.text(pos[0], pos[1], label, ha="center", va="center",
                    fontsize=8, fontweight="bold", color="white", zorder=11)

        if step == n_steps - 1 and tagged:
            flash = plt.Circle(sp[step], 2.0, facecolor="#ffa657",
                               alpha=0.4, zorder=9)
            ax.add_patch(flash)
            ax.text(0, ARENA_HALF + 0.3, "TAGGED!", ha="center",
                    fontsize=14, fontweight="bold", color="#ffa657", zorder=12)

        ax.text(-ARENA_HALF + 0.3, -ARENA_HALF - 0.5,
                f"Step {step}/{n_steps}", fontsize=9, color="#8b949e", va="top")
        ax.text(ARENA_HALF - 0.3, -ARENA_HALF - 0.5,
                f"d={dist:.1f}", fontsize=9, color="#8b949e", va="top", ha="right")

        ax.set_title(title, fontsize=10, color="#c9d1d9", pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    print(f"  Rendering {n_steps} frames -> {output_file}")
    writer = PillowWriter(fps=fps)
    with writer.saving(fig, str(output_file), dpi=100):
        for step in range(n_steps):
            draw_frame(step)
            writer.grab_frame()
        # Hold last frame
        for _ in range(fps):
            writer.grab_frame()
    plt.close()
    print(f"    Done ({Path(output_file).stat().st_size / 1024:.0f} KB)")


def main():
    results_path = Path("experiments/results/cross_method_gauntlet/gauntlet_results.json")
    with open(results_path) as f:
        results = json.load(f)

    labels = results["labels"]
    selected = results["selected"]
    output_dir = Path("docs/reward_shaping")

    # Build label -> (seeker_path, hider_path, algo) map
    agent_info = {}
    for entry in selected:
        label = entry["label"]
        algo = entry["algo"]
        agent_info[label] = {
            "seeker_path": entry["seeker_path"],
            "hider_path": entry["hider_path"],
            "algo": algo,
        }

    for seeker_label, hider_label, short_name, caption in MATCHUPS:
        print(f"\n=== {caption} ===")
        print(f"  Seeker: {seeker_label}")
        print(f"  Hider:  {hider_label}")

        s_info = agent_info[seeker_label]
        h_info = agent_info[hider_label]

        seeker_fn = load_agent(s_info["seeker_path"], s_info["algo"])
        hider_fn = load_agent(h_info["hider_path"], h_info["algo"])

        episodes = simulate_episodes(seeker_fn, hider_fn, num_envs=50)

        if not episodes:
            print("  No valid episodes!")
            continue

        best = episodes[0]
        tag_str = "Tagged" if best["tagged"] else "Survived"
        print(f"  Best episode: len={best['length']}, {tag_str}, score={best['score']:.1f}")

        gif_path = output_dir / f"xmethod_{short_name}.gif"
        render_episode(best, str(gif_path), caption)

    print(f"\nAll GIFs saved to {output_dir}/")


if __name__ == "__main__":
    main()
