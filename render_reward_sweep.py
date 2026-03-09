#!/usr/bin/env python3
"""Render showcase animations from reward sweep results.

Picks the best episode from each preset×algo combo and renders GIFs.
"""
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


def load_ppo_policy(path, obs_dim, act_dim):
    cfg = PPOConfig(obs_dim=obs_dim, act_dim=act_dim)
    policy = PPOAgent(cfg)
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    policy.pi.load_state_dict(ckpt["pi"])
    policy.vf.load_state_dict(ckpt["vf"])
    return policy


def load_sac_policy(path, obs_dim, act_dim):
    cfg = SACConfig(obs_dim=obs_dim, act_dim=act_dim)
    agent = SACAgent(cfg)
    agent.load_policy(path)
    return agent


def act_batch_ppo(policy, obs_batch):
    with torch.no_grad():
        x = torch.as_tensor(obs_batch, dtype=torch.float32)
        logits = policy.pi(x)
        mean, log_std = torch.chunk(logits, 2, dim=-1)
        log_std = torch.clamp(log_std, -2.0, 1.5)
        std = torch.exp(log_std)
        action = torch.tanh(torch.distributions.Normal(mean, std).sample())
        return action.cpu().numpy()


def act_batch_sac(agent, obs_batch):
    with torch.no_grad():
        x = torch.as_tensor(obs_batch, dtype=torch.float32)
        actions, _ = agent.actor.sample(x)
        return actions.cpu().numpy()


def find_latest_run(base_dir):
    """Find the latest run directory with final policies."""
    base = Path(base_dir)
    if not base.exists():
        return None
    runs = sorted(base.glob("2026*"))
    for run in reversed(runs):
        if (run / "policy_seeker_final.pt").exists():
            return run
    return None


def simulate_episodes(seeker_act_fn, hider_act_fn, num_envs=50):
    """Simulate episodes and collect trajectory data."""
    cfg = TagEnvConfig(layout="four_corners", hider_speed_mult=1.15)
    env = VecTagEnv(num_envs=num_envs, config=cfg)
    obs = env.reset()

    max_steps = int(cfg.time_limit / (cfg.dt * cfg.steps_per_action))
    eids = np.arange(num_envs)

    all_seeker_pos = [[] for _ in range(num_envs)]
    all_hider_pos = [[] for _ in range(num_envs)]
    active = np.ones(num_envs, dtype=bool)
    episode_lengths = np.zeros(num_envs, dtype=int)
    episode_tagged = np.zeros(num_envs, dtype=bool)

    for step_i in range(max_steps):
        seeker_idx = env.seeker_idx
        hider_idx = 1 - seeker_idx

        seeker_pos = env.positions[eids, seeker_idx].copy()
        hider_pos = env.positions[eids, hider_idx].copy()

        for i in range(num_envs):
            if active[i]:
                all_seeker_pos[i].append(seeker_pos[i].tolist())
                all_hider_pos[i].append(hider_pos[i].tolist())

        seeker_actions = seeker_act_fn(obs['seeker'])
        hider_actions = hider_act_fn(obs['hider'])

        obs, rewards, dones, infos = env.step({
            'seeker': seeker_actions,
            'hider': hider_actions,
        })

        newly_done = dones & active
        if np.any(newly_done):
            for i in np.where(newly_done)[0]:
                episode_lengths[i] = step_i + 1
                episode_tagged[i] = infos['tagged'][i]
            active[newly_done] = False

        if not np.any(active):
            break

    for i in range(num_envs):
        if active[i]:
            episode_lengths[i] = max_steps
            episode_tagged[i] = False

    episodes = []
    for i in range(num_envs):
        sp = np.array(all_seeker_pos[i])
        hp = np.array(all_hider_pos[i])
        T = len(hp)
        if T < 5:
            continue

        hider_diffs = np.diff(hp, axis=0)
        hider_dist = np.sum(np.linalg.norm(hider_diffs, axis=1))

        wall_dist = ARENA_HALF - np.abs(hp)
        in_corner = (wall_dist[:, 0] < 3.0) & (wall_dist[:, 1] < 3.0)
        corner_pct = np.mean(in_corner) * 100

        near_any_wall = np.min(wall_dist, axis=1)
        interior_pct = np.mean(near_any_wall > 5.0) * 100

        if len(hider_diffs) > 1:
            angles = np.arctan2(hider_diffs[:, 1], hider_diffs[:, 0])
            angle_diffs = np.abs(np.diff(angles))
            angle_diffs = np.minimum(angle_diffs, 2 * np.pi - angle_diffs)
            direction_changes = int(np.sum(angle_diffs > np.radians(45)))
        else:
            direction_changes = 0

        inter_dist = np.linalg.norm(sp - hp, axis=1)
        dist_variance = float(np.var(inter_dist))

        # Near-wall fraction
        near_wall_frac = float(np.mean(near_any_wall < 2.0))

        episodes.append({
            'index': i,
            'length': int(episode_lengths[i]),
            'tagged': bool(episode_tagged[i]),
            'hider_dist': float(hider_dist),
            'corner_pct': float(corner_pct),
            'interior_pct': float(interior_pct),
            'direction_changes': direction_changes,
            'dist_variance': dist_variance,
            'near_wall_frac': near_wall_frac,
            'seeker_positions': all_seeker_pos[i],
            'hider_positions': all_hider_pos[i],
        })

    return episodes


def score_episodes(episodes):
    """Score episodes for visual interest."""
    if not episodes:
        return []

    for key in ['hider_dist', 'corner_pct', 'interior_pct', 'direction_changes', 'dist_variance']:
        vals = [e[key] for e in episodes]
        mn, mx = min(vals), max(vals)
        rng = mx - mn if mx > mn else 1.0
        for e in episodes:
            e[f'_n_{key}'] = (e[key] - mn) / rng * 100

    for ep in episodes:
        score = 0.0
        score += ep['_n_hider_dist'] * 0.30
        score += (100 - ep['_n_corner_pct']) * 0.25
        score += ep['_n_interior_pct'] * 0.15
        score += ep['_n_direction_changes'] * 0.15
        score += ep['_n_dist_variance'] * 0.15
        if ep['tagged']:
            score += 20.0
        if 100 <= ep['length'] <= 180:
            score += 10.0
        ep['score'] = score

    return sorted(episodes, key=lambda e: e['score'], reverse=True)


def render_episode(episode, output_file, title, fps=20):
    """Render a single episode to GIF."""
    sp = np.array(episode['seeker_positions'])
    hp = np.array(episode['hider_positions'])
    n_steps = len(sp)
    tagged = episode['tagged']

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

        for pos, color, label in [(sp[step], "#f85149", "S"), (hp[step], "#58a6ff", "H")]:
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

    print(f"Rendering {n_steps} frames to {output_file}...")
    writer = PillowWriter(fps=fps)
    with writer.saving(fig, str(output_file), dpi=100):
        for step in range(n_steps):
            draw_frame(step)
            writer.grab_frame()
        for _ in range(fps):
            writer.grab_frame()

    plt.close()
    print(f"  Done: {output_file} ({Path(output_file).stat().st_size / 1024:.0f} KB)")


def main():
    base = Path("experiments/results/reward_sweep")
    output_dir = base / "animations"
    output_dir.mkdir(parents=True, exist_ok=True)

    presets = [
        "R0_baseline", "R1_seeker_pursuit", "R2_hider_active", "R3_both_shaped",
        "R4_sparse", "R5_escalating", "R6_coverage", "R7_kitchen_sink",
    ]
    algos = ["ppo", "sac"]

    obs_dim = 87  # 12 + 36*2 + 3
    act_dim = 3

    best_per_config = []

    for preset in presets:
        for algo in algos:
            # Try all seeds, pick seed with best episode
            best_ep = None
            best_seed = None

            for seed in [0, 1, 2]:
                run_dir = find_latest_run(base / preset / algo / f"seed_{seed}")
                if run_dir is None:
                    continue

                seeker_path = run_dir / "policy_seeker_final.pt"
                hider_path = run_dir / "policy_hider_final.pt"
                if not seeker_path.exists() or not hider_path.exists():
                    continue

                # Load policies
                if algo == "ppo":
                    seeker = load_ppo_policy(str(seeker_path), obs_dim, act_dim)
                    hider = load_ppo_policy(str(hider_path), obs_dim, act_dim)
                    seeker_fn = lambda obs, s=seeker: act_batch_ppo(s, obs)
                    hider_fn = lambda obs, h=hider: act_batch_ppo(h, obs)
                else:
                    seeker = load_sac_policy(str(seeker_path), obs_dim, act_dim)
                    hider = load_sac_policy(str(hider_path), obs_dim, act_dim)
                    seeker_fn = lambda obs, s=seeker: act_batch_sac(s, obs)
                    hider_fn = lambda obs, h=hider: act_batch_sac(h, obs)

                episodes = simulate_episodes(seeker_fn, hider_fn, num_envs=30)
                scored = score_episodes(episodes)

                if scored and (best_ep is None or scored[0]['score'] > best_ep['score']):
                    best_ep = scored[0]
                    best_seed = seed

            if best_ep is not None:
                best_ep['preset'] = preset
                best_ep['algo'] = algo
                best_ep['seed'] = best_seed
                best_per_config.append(best_ep)
                tag_str = "TAG" if best_ep['tagged'] else "timeout"
                print(f"{preset}/{algo} seed={best_seed}: "
                      f"score={best_ep['score']:.1f}, len={best_ep['length']}, "
                      f"{tag_str}, wall={best_ep['near_wall_frac']:.0%}, "
                      f"dirs={best_ep['direction_changes']}")

    # Sort by score, render top 8 (one per preset, best algo)
    best_per_config.sort(key=lambda e: e['score'], reverse=True)

    print(f"\n{'='*60}")
    print(f"Rendering top episodes")
    print(f"{'='*60}")

    rendered = set()
    count = 0
    for ep in best_per_config:
        key = ep['preset']
        if key in rendered:
            continue
        rendered.add(key)
        count += 1

        label = ep['preset'].replace('_', ' ')
        algo_upper = ep['algo'].upper()
        tag_str = "Tagged" if ep['tagged'] else "Timeout"
        title = f"{label} | {algo_upper} | {tag_str} @ step {ep['length']}"

        filename = f"{ep['preset']}_{ep['algo']}.gif"
        render_episode(ep, str(output_dir / filename), title)

        if count >= 8:
            break

    # Also render all 16 combos (smaller, for comparison)
    print(f"\nRendering all preset x algo combos...")
    for ep in best_per_config:
        label = ep['preset'].replace('_', ' ')
        algo_upper = ep['algo'].upper()
        tag_str = "Tagged" if ep['tagged'] else "Timeout"
        title = f"{label} | {algo_upper} | {tag_str} @ step {ep['length']}"

        filename = f"{ep['preset']}_{ep['algo']}.gif"
        outpath = output_dir / filename
        if outpath.exists():
            continue
        render_episode(ep, str(outpath), title)

    print(f"\nAll animations saved to {output_dir}/")


if __name__ == "__main__":
    main()
