#!/usr/bin/env python3
"""Render a hero GIF: best seeker vs best hider across 10 rounds in one animation."""
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

# Best seeker & best hider from cross-method gauntlet
SEEKER_LABEL = "fr_v2/R5_escalating/A0.75_sac"
HIDER_LABEL = "fr_v2/R3_both_shaped/A0.50_sac"

N_ROUNDS = 10
N_CANDIDATE_EPISODES = 80
FPS = 20
OUTPUT = Path("docs/header_animation.gif")


def load_sac_agent(path):
    cfg = SACConfig(obs_dim=OBS_DIM, act_dim=ACT_DIM)
    agent = SACAgent(cfg)
    agent.load_policy(str(path))

    def act(obs, a=agent):
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32)
            actions, _ = a.actor.sample(x)
            return actions.cpu().numpy()
    return act


def simulate_episodes(seeker_fn, hider_fn, num_envs):
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
        if len(hp) < 10:
            continue

        hider_dist = float(np.sum(np.linalg.norm(np.diff(hp, axis=0), axis=1)))
        wall_dist = ARENA_HALF - np.abs(hp)
        corner_pct = float(np.mean((wall_dist[:, 0] < 3) & (wall_dist[:, 1] < 3)))
        inter_dist = np.linalg.norm(sp - hp, axis=1)

        diffs = np.diff(hp, axis=0)
        if len(diffs) > 1:
            angles = np.arctan2(diffs[:, 1], diffs[:, 0])
            adiffs = np.abs(np.diff(angles))
            adiffs = np.minimum(adiffs, 2 * np.pi - adiffs)
            dir_changes = int(np.sum(adiffs > np.radians(45)))
        else:
            dir_changes = 0

        # Score: prefer dynamic play, tags, varied distance, medium length
        score = hider_dist * 0.01 + (1 - corner_pct) * 30 + dir_changes * 0.5
        score += float(np.var(inter_dist)) * 0.01
        if ep_tagged[i]:
            score += 20.0
        if 60 <= ep_lengths[i] <= 180:
            score += 15.0

        episodes.append({
            "length": int(ep_lengths[i]),
            "tagged": bool(ep_tagged[i]),
            "score": score,
            "seeker_pos": sp.tolist(),
            "hider_pos": hp.tolist(),
        })

    return sorted(episodes, key=lambda e: e["score"], reverse=True)


def select_rounds(episodes, n_rounds):
    """Pick n_rounds episodes: mix of tags and survivals, diverse lengths."""
    # Separate tags and survivals
    tags = [e for e in episodes if e["tagged"]]
    survs = [e for e in episodes if not e["tagged"]]

    # Want roughly proportional mix, but ensure at least 1 survival if available
    n_surv = max(1, min(len(survs), n_rounds // 3))
    n_tag = n_rounds - n_surv

    selected = tags[:n_tag] + survs[:n_surv]

    # If not enough of either, fill from the other
    while len(selected) < n_rounds and (tags or survs):
        pool = tags if len(tags) > len([s for s in selected if s["tagged"]]) else survs
        for e in pool:
            if e not in selected:
                selected.append(e)
                break
        else:
            break

    # Shuffle so tags and survivals are interleaved
    rng = np.random.RandomState(42)
    rng.shuffle(selected)
    return selected[:n_rounds]


def render_multi_round(rounds, output_file, fps=FPS):
    """Render multiple rounds into a single GIF with round counter and score."""
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(7, 7.8), dpi=100)

    total_tags = 0
    total_survivals = 0

    def draw_frame(sp, hp, step, n_steps, tagged, round_num, tags_so_far, survs_so_far, is_last_frame):
        ax.clear()
        ax.set_xlim(-ARENA_HALF - 1, ARENA_HALF + 1)
        ax.set_ylim(-ARENA_HALF - 2, ARENA_HALF + 1)
        ax.set_aspect("equal")
        ax.set_facecolor("#0d1117")
        fig.set_facecolor("#0d1117")

        # Arena
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

        # Trails
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

        # Agents
        for pos, color, label in [(sp[step], "#f85149", "S"),
                                  (hp[step], "#58a6ff", "H")]:
            glow = plt.Circle(pos, 1.2, facecolor=color, alpha=0.15)
            ax.add_patch(glow)
            glow2 = plt.Circle(pos, 0.8, facecolor=color, alpha=0.25)
            ax.add_patch(glow2)
            agent_circle = plt.Circle(pos, 0.5, facecolor=color,
                                      edgecolor="white", linewidth=1.2,
                                      alpha=0.95, zorder=10)
            ax.add_patch(agent_circle)
            ax.text(pos[0], pos[1], label, ha="center", va="center",
                    fontsize=8, fontweight="bold", color="white", zorder=11)

        # Tag flash
        if is_last_frame and tagged:
            flash = plt.Circle(sp[-1], 2.0, facecolor="#ffa657",
                               alpha=0.4, zorder=9)
            ax.add_patch(flash)
            ax.text(0, 0, "TAGGED!", ha="center", va="center",
                    fontsize=18, fontweight="bold", color="#ffa657",
                    zorder=12, alpha=0.9)

        # Survival text
        if is_last_frame and not tagged:
            ax.text(0, 0, "SURVIVED!", ha="center", va="center",
                    fontsize=18, fontweight="bold", color="#58a6ff",
                    zorder=12, alpha=0.9)

        # HUD: round counter + score
        ax.text(0, ARENA_HALF + 0.5,
                f"ROUND {round_num}/{N_ROUNDS}",
                ha="center", va="bottom", fontsize=13,
                fontweight="bold", color="#c9d1d9", zorder=15)

        score_text = f"Seeker {tags_so_far}  -  {survs_so_far} Hider"
        ax.text(0, -ARENA_HALF - 0.7,
                score_text, ha="center", va="top", fontsize=11,
                fontweight="bold", color="#8b949e", zorder=15)

        # Step counter + distance
        ax.text(-ARENA_HALF + 0.3, -ARENA_HALF - 1.5,
                f"Step {step}/{n_steps}", fontsize=8, color="#484f58", va="top")
        ax.text(ARENA_HALF - 0.3, -ARENA_HALF - 1.5,
                f"d={dist:.1f}", fontsize=8, color="#484f58", va="top", ha="right")

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    print(f"Rendering {N_ROUNDS} rounds to {output_file}...")

    writer = PillowWriter(fps=fps)
    with writer.saving(fig, str(output_file), dpi=100):
        for round_idx, ep in enumerate(rounds):
            sp = np.array(ep["seeker_pos"])
            hp = np.array(ep["hider_pos"])
            n_steps = len(sp)

            # Subsample long episodes to keep GIF reasonable (~3s per round)
            target_frames = fps * 3  # 3 seconds per round
            if n_steps > target_frames:
                indices = np.linspace(0, n_steps - 1, target_frames, dtype=int)
            else:
                indices = np.arange(n_steps)

            for frame_i, step in enumerate(indices):
                is_last = (frame_i == len(indices) - 1)
                draw_frame(sp, hp, step, n_steps, ep["tagged"],
                           round_idx + 1, total_tags, total_survivals,
                           is_last)
                writer.grab_frame()

            # Update score after round
            if ep["tagged"]:
                total_tags += 1
            else:
                total_survivals += 1

            # Hold last frame for 1 second
            for _ in range(fps):
                draw_frame(sp, hp, len(sp) - 1, n_steps, ep["tagged"],
                           round_idx + 1, total_tags, total_survivals,
                           True)
                writer.grab_frame()

    plt.close()
    size_kb = output_file.stat().st_size / 1024
    print(f"Done: {output_file} ({size_kb:.0f} KB)")
    print(f"Final score: Seeker {total_tags} - {total_survivals} Hider")


def main():
    # Load gauntlet results to find policy paths
    results_path = Path("experiments/results/cross_method_gauntlet/gauntlet_results.json")
    with open(results_path) as f:
        results = json.load(f)

    agent_info = {e["label"]: e for e in results["selected"]}
    s_info = agent_info[SEEKER_LABEL]
    h_info = agent_info[HIDER_LABEL]

    print(f"Seeker: {SEEKER_LABEL}")
    print(f"  path: {s_info['seeker_path']}")
    print(f"Hider:  {HIDER_LABEL}")
    print(f"  path: {h_info['hider_path']}")

    seeker_fn = load_sac_agent(s_info["seeker_path"])
    hider_fn = load_sac_agent(h_info["hider_path"])

    print(f"\nSimulating {N_CANDIDATE_EPISODES} episodes...")
    episodes = simulate_episodes(seeker_fn, hider_fn, N_CANDIDATE_EPISODES)

    n_tags = sum(1 for e in episodes if e["tagged"])
    print(f"  {n_tags}/{len(episodes)} tagged ({n_tags/len(episodes):.0%})")

    rounds = select_rounds(episodes, N_ROUNDS)
    round_tags = sum(1 for r in rounds if r["tagged"])
    round_lengths = [r["length"] for r in rounds]
    print(f"\nSelected {len(rounds)} rounds: {round_tags} tags, "
          f"{len(rounds) - round_tags} survivals")
    print(f"  Lengths: {round_lengths}")

    render_multi_round(rounds, OUTPUT)


if __name__ == "__main__":
    main()
