#!/usr/bin/env python3
"""Find visually compelling episodes from trained RL tag game for header animation."""

import sys
import json
import os
from pathlib import Path

sys.path.insert(0, "/work/users/juqu/AI-Plays-Tag")

import numpy as np
import torch

from trainer.tag_env import VecTagEnv, TagEnvConfig
from trainer.ppo import PPOAgent, PPOConfig


def load_policy(path, obs_dim, act_dim):
    """Load a trained policy from checkpoint."""
    cfg = PPOConfig(obs_dim=obs_dim, act_dim=act_dim)
    policy = PPOAgent(cfg)
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    policy.pi.load_state_dict(ckpt["pi"])
    policy.vf.load_state_dict(ckpt["vf"])
    return policy


def batch_act(policy, obs_batch):
    """Get batch actions from policy."""
    with torch.no_grad():
        x = torch.as_tensor(obs_batch, dtype=torch.float32)
        logits = policy.pi(x)
        mean, log_std = torch.chunk(logits, 2, dim=-1)
        log_std = torch.clamp(log_std, -2.0, 1.5)
        std = torch.exp(log_std)
        action = torch.tanh(torch.distributions.Normal(mean, std).sample())
        return action.cpu().numpy()


def find_latest_checkpoint_dir(base_dir):
    """Find the latest timestamp directory and highest checkpoint number."""
    base = Path(base_dir)
    if not base.exists():
        return None, None

    # Find latest timestamp subdirectory
    timestamp_dirs = sorted([d for d in base.iterdir() if d.is_dir() and d.name != "checkpoints"])
    if not timestamp_dirs:
        return None, None

    latest_ts = timestamp_dirs[-1]
    ckpt_dir = latest_ts / "checkpoints"
    if not ckpt_dir.exists():
        return None, None

    # Find highest numbered seeker and hider checkpoints
    seeker_ckpts = sorted(ckpt_dir.glob("seeker_*.pt"))
    hider_ckpts = sorted(ckpt_dir.glob("hider_*.pt"))

    if not seeker_ckpts or not hider_ckpts:
        return None, None

    seeker_path = seeker_ckpts[-1]
    hider_path = hider_ckpts[-1]

    return str(seeker_path), str(hider_path)


def simulate_episodes(seeker_policy, hider_policy, num_envs=100, config_name=""):
    """Simulate episodes and collect trajectory data."""
    cfg = TagEnvConfig(layout="four_corners", hider_speed_mult=1.15)
    env = VecTagEnv(num_envs=num_envs, config=cfg)
    obs = env.reset()

    max_steps = int(cfg.time_limit / (cfg.dt * cfg.steps_per_action))  # 200
    arena_half = cfg.arena_half  # 15.0

    # Storage for trajectories
    all_seeker_pos = [[] for _ in range(num_envs)]
    all_hider_pos = [[] for _ in range(num_envs)]
    active = np.ones(num_envs, dtype=bool)
    episode_lengths = np.zeros(num_envs, dtype=int)
    episode_tagged = np.zeros(num_envs, dtype=bool)

    for step_i in range(max_steps):
        # Record positions before stepping
        eids = np.arange(num_envs)
        seeker_idx = env.seeker_idx  # (num_envs,)
        hider_idx = 1 - seeker_idx

        seeker_pos = env.positions[eids, seeker_idx].copy()  # (num_envs, 2)
        hider_pos = env.positions[eids, hider_idx].copy()    # (num_envs, 2)

        for i in range(num_envs):
            if active[i]:
                all_seeker_pos[i].append(seeker_pos[i].tolist())
                all_hider_pos[i].append(hider_pos[i].tolist())

        # Get actions
        seeker_actions = batch_act(seeker_policy, obs['seeker'])
        hider_actions = batch_act(hider_policy, obs['hider'])

        # Step
        obs, rewards, dones, infos = env.step({
            'seeker': seeker_actions,
            'hider': hider_actions,
        })

        # Track which episodes just finished
        newly_done = dones & active
        if np.any(newly_done):
            # Record final positions for newly done episodes
            for i in np.where(newly_done)[0]:
                episode_lengths[i] = step_i + 1
                episode_tagged[i] = infos['tagged'][i]
            active[newly_done] = False

        if not np.any(active):
            break

    # Mark any still-active episodes as timed out at max_steps
    for i in range(num_envs):
        if active[i]:
            episode_lengths[i] = max_steps
            episode_tagged[i] = False

    # Compute per-episode metrics
    episodes = []
    for i in range(num_envs):
        sp = np.array(all_seeker_pos[i])  # (T, 2)
        hp = np.array(all_hider_pos[i])   # (T, 2)
        T = len(hp)

        if T < 5:
            continue

        # 1. Hider distance traveled
        hider_diffs = np.diff(hp, axis=0)
        hider_dist = np.sum(np.linalg.norm(hider_diffs, axis=1))

        # 2. Corner time: within 3 units of TWO walls at +/-15
        wall_dist = arena_half - np.abs(hp)  # distance to nearest wall in each axis
        in_corner = (wall_dist[:, 0] < 3.0) & (wall_dist[:, 1] < 3.0)
        corner_pct = np.mean(in_corner) * 100

        # 3. Interior time: more than 5 units from any wall
        near_any_wall = np.min(wall_dist, axis=1)
        in_interior = near_any_wall > 5.0
        interior_pct = np.mean(in_interior) * 100

        # 4. Direction changes (angle changes > 45 degrees)
        if len(hider_diffs) > 1:
            # Compute direction angles
            angles = np.arctan2(hider_diffs[:, 1], hider_diffs[:, 0])
            angle_diffs = np.abs(np.diff(angles))
            angle_diffs = np.minimum(angle_diffs, 2 * np.pi - angle_diffs)
            direction_changes = np.sum(angle_diffs > np.radians(45))
        else:
            direction_changes = 0

        # 5. Distance variance between agents
        inter_dist = np.linalg.norm(sp - hp, axis=1)
        dist_variance = np.var(inter_dist)

        episodes.append({
            'index': i,
            'length': int(episode_lengths[i]),
            'tagged': bool(episode_tagged[i]),
            'hider_dist': float(hider_dist),
            'corner_pct': float(corner_pct),
            'interior_pct': float(interior_pct),
            'direction_changes': int(direction_changes),
            'dist_variance': float(dist_variance),
            'seeker_positions': [p for p in all_seeker_pos[i]],
            'hider_positions': [p for p in all_hider_pos[i]],
        })

    return episodes


def score_episodes(episodes):
    """Score episodes on visual interest."""
    if not episodes:
        return []

    # Normalize each metric to [0, 100]
    metrics = {
        'hider_dist': [e['hider_dist'] for e in episodes],
        'corner_pct': [e['corner_pct'] for e in episodes],
        'interior_pct': [e['interior_pct'] for e in episodes],
        'direction_changes': [e['direction_changes'] for e in episodes],
        'dist_variance': [e['dist_variance'] for e in episodes],
    }

    # Compute min/max for normalization
    ranges = {}
    for key, vals in metrics.items():
        mn, mx = min(vals), max(vals)
        ranges[key] = (mn, mx - mn if mx > mn else 1.0)

    for ep in episodes:
        # Normalize each to 0-100
        norm_hider_dist = (ep['hider_dist'] - ranges['hider_dist'][0]) / ranges['hider_dist'][1] * 100
        norm_corner_pct = (ep['corner_pct'] - ranges['corner_pct'][0]) / ranges['corner_pct'][1] * 100
        norm_interior_pct = (ep['interior_pct'] - ranges['interior_pct'][0]) / ranges['interior_pct'][1] * 100
        norm_dir_changes = (ep['direction_changes'] - ranges['direction_changes'][0]) / ranges['direction_changes'][1] * 100
        norm_dist_var = (ep['dist_variance'] - ranges['dist_variance'][0]) / ranges['dist_variance'][1] * 100

        score = 0.0
        # High hider distance traveled (30%)
        score += norm_hider_dist * 0.30
        # Low corner time (25%) - invert: less corner = higher score
        score += (100 - norm_corner_pct) * 0.25
        # High interior time (15%)
        score += norm_interior_pct * 0.15
        # High direction changes (15%)
        score += norm_dir_changes * 0.15
        # High distance variance (15%)
        score += norm_dist_var * 0.15

        # BONUS: episode ends in a tag (dramatic ending)
        if ep['tagged']:
            score += 20.0
        # BONUS: episode length between 100-180 steps
        if 100 <= ep['length'] <= 180:
            score += 10.0

        ep['score'] = score
        ep['score_components'] = {
            'hider_dist_norm': round(norm_hider_dist, 1),
            'corner_pct_norm': round(norm_corner_pct, 1),
            'interior_pct_norm': round(norm_interior_pct, 1),
            'dir_changes_norm': round(norm_dir_changes, 1),
            'dist_var_norm': round(norm_dist_var, 1),
        }

    return sorted(episodes, key=lambda e: e['score'], reverse=True)


def main():
    obs_dim = 87  # 12 + 36*2 + 3
    act_dim = 3

    configs = [
        ("STP01_HSM115", "A10_uniform", 0),
        ("STP01_HSM115", "A30_uniform", 0),
        ("STP05_HSM115", "A10_uniform", 0),
    ]

    base_results = Path("/work/users/juqu/AI-Plays-Tag/experiments/results/zoo_asweep")
    output_dir = base_results / "best_episodes"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_top_episodes = []

    for game_cfg, a_cfg, seed in configs:
        config_name = f"{game_cfg}/{a_cfg}/seed_{seed}"
        exp_dir = base_results / game_cfg / a_cfg / f"seed_{seed}"
        print(f"\n{'='*70}")
        print(f"Config: {config_name}")
        print(f"{'='*70}")

        seeker_path, hider_path = find_latest_checkpoint_dir(exp_dir)
        if seeker_path is None:
            print(f"  SKIP: no checkpoints found in {exp_dir}")
            continue

        print(f"  Seeker checkpoint: {Path(seeker_path).name}")
        print(f"  Hider checkpoint:  {Path(hider_path).name}")

        seeker_policy = load_policy(seeker_path, obs_dim, act_dim)
        hider_policy = load_policy(hider_path, obs_dim, act_dim)

        print(f"  Simulating 100 episodes...")
        episodes = simulate_episodes(seeker_policy, hider_policy, num_envs=100, config_name=config_name)
        print(f"  Collected {len(episodes)} valid episodes")

        # Score episodes
        scored = score_episodes(episodes)

        # Print summary stats
        tag_count = sum(1 for e in scored if e['tagged'])
        timeout_count = len(scored) - tag_count
        avg_len = np.mean([e['length'] for e in scored])
        print(f"  Tags: {tag_count}, Timeouts: {timeout_count}, Avg length: {avg_len:.1f}")

        # Print top 10
        print(f"\n  {'Rank':<5} {'Score':<8} {'Len':<5} {'Tag':<4} {'HiderDist':<10} {'Corner%':<9} {'Interior%':<10} {'DirChg':<7} {'DistVar':<8}")
        print(f"  {'-'*65}")
        for rank, ep in enumerate(scored[:10], 1):
            print(f"  {rank:<5} {ep['score']:<8.1f} {ep['length']:<5} {'Y' if ep['tagged'] else 'N':<4} "
                  f"{ep['hider_dist']:<10.1f} {ep['corner_pct']:<9.1f} {ep['interior_pct']:<10.1f} "
                  f"{ep['direction_changes']:<7} {ep['dist_variance']:<8.1f}")

        # Store top episodes with config info for cross-config comparison
        for ep in scored[:5]:
            ep['config'] = config_name
            ep['game_cfg'] = game_cfg
            ep['a_cfg'] = a_cfg
            ep['seed'] = seed
            all_top_episodes.append(ep)

    # Global ranking across all configs
    all_top_episodes.sort(key=lambda e: e['score'], reverse=True)

    print(f"\n{'='*70}")
    print(f"GLOBAL TOP 10 (across all configs)")
    print(f"{'='*70}")
    print(f"{'Rank':<5} {'Config':<35} {'Score':<8} {'Len':<5} {'Tag':<4} {'HiderDist':<10} {'Corner%':<9} {'Interior%':<10} {'DirChg':<7} {'DistVar':<8}")
    print(f"{'-'*95}")
    for rank, ep in enumerate(all_top_episodes[:10], 1):
        print(f"{rank:<5} {ep['config']:<35} {ep['score']:<8.1f} {ep['length']:<5} {'Y' if ep['tagged'] else 'N':<4} "
              f"{ep['hider_dist']:<10.1f} {ep['corner_pct']:<9.1f} {ep['interior_pct']:<10.1f} "
              f"{ep['direction_changes']:<7} {ep['dist_variance']:<8.1f}")

    # Save top 3 globally
    print(f"\nSaving top 3 episodes to {output_dir}/")
    for rank, ep in enumerate(all_top_episodes[:3], 1):
        # Parse config into filename-friendly format
        cfg_label = ep['config'].replace('/', '_').replace('seed_', 's')

        save_data = {
            'config': {
                'game_cfg': ep['game_cfg'],
                'a_cfg': ep['a_cfg'],
                'seed': ep['seed'],
                'stp': ep['game_cfg'].split('_')[0],
                'hsm': ep['game_cfg'].split('_')[1],
                'A': ep['a_cfg'].split('_')[0],
                'sampling': ep['a_cfg'].split('_')[1],
            },
            'episode_length': ep['length'],
            'tagged': ep['tagged'],
            'seeker_positions': ep['seeker_positions'],
            'hider_positions': ep['hider_positions'],
            'score': round(ep['score'], 2),
            'score_components': ep['score_components'],
            'metrics': {
                'hider_distance_traveled': round(ep['hider_dist'], 2),
                'corner_time_pct': round(ep['corner_pct'], 2),
                'interior_time_pct': round(ep['interior_pct'], 2),
                'direction_changes': ep['direction_changes'],
                'distance_variance': round(ep['dist_variance'], 2),
            },
        }

        filename = f"episode_{cfg_label}_{rank}.json"
        filepath = output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
        print(f"  Saved: {filename} (score={ep['score']:.1f}, len={ep['length']}, tagged={ep['tagged']})")

    print("\nDone!")


if __name__ == "__main__":
    main()
