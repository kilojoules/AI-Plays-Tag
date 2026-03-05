#!/usr/bin/env python3
"""
Analyze hider behavior across trained zoo agents.
Loads final checkpoints from HSM=1.15 and HSM=1.20 configs (fast hiders)
and examines whether hiders develop interesting evasive strategies
or just degenerate into corner-hugging.
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from trainer.tag_env import VecTagEnv, TagEnvConfig
from trainer.ppo import PPOAgent, PPOConfig


# ── Helpers ──────────────────────────────────────────────────────────────

def batch_act(policy, obs_batch):
    """Vectorized deterministic-ish inference (sample from learned distribution)."""
    with torch.no_grad():
        x = torch.as_tensor(obs_batch, dtype=torch.float32)
        logits = policy.pi(x)
        mean, log_std = torch.chunk(logits, 2, dim=-1)
        log_std = torch.clamp(log_std, -2.0, 1.5)
        std = torch.exp(log_std)
        action = torch.tanh(torch.distributions.Normal(mean, std).sample())
        return action.cpu().numpy()


def batch_act_deterministic(policy, obs_batch):
    """Vectorized deterministic inference (use mean only)."""
    with torch.no_grad():
        x = torch.as_tensor(obs_batch, dtype=torch.float32)
        logits = policy.pi(x)
        mean, log_std = torch.chunk(logits, 2, dim=-1)
        action = torch.tanh(mean)
        return action.cpu().numpy()


def find_checkpoint_dir(base_dir: Path, stp_str: str, hsm_pct: str,
                        a_str: str, sampling: str, seed: int = 0) -> Optional[Path]:
    """Find the checkpoint directory for a given config."""
    exp_dir = base_dir / f"STP{stp_str}_HSM{hsm_pct}" / f"{a_str}_{sampling}" / f"seed_{seed}"
    if not exp_dir.exists():
        return None
    # Find timestamp directory (should be one)
    ts_dirs = sorted([d for d in exp_dir.iterdir() if d.is_dir()])
    if not ts_dirs:
        return None
    ckpt_dir = ts_dirs[-1] / "checkpoints"
    if not ckpt_dir.exists():
        return None
    return ckpt_dir


def find_latest_checkpoint(ckpt_dir: Path, role: str) -> Optional[Path]:
    """Find the latest checkpoint file for a given role."""
    files = sorted(ckpt_dir.glob(f"{role}_*.pt"))
    return files[-1] if files else None


@dataclass
class EpisodeStats:
    """Statistics for a single episode's hider behavior."""
    total_distance: float        # Total distance traveled by hider
    wall_time_frac: float        # Fraction of time within 2 units of boundary
    corner_time_frac: float      # Fraction of time within 3 units of two walls
    direction_changes: int       # Number of acceleration sign changes
    survived: bool               # Whether hider survived full episode
    episode_length: int          # Number of steps
    avg_agent_distance: float    # Average distance between agents
    std_agent_distance: float    # Std of distance (evasive = high variance)
    min_agent_distance: float    # Minimum distance (close calls)
    max_agent_distance: float    # Maximum distance
    interior_time_frac: float    # Fraction of time in interior (>5 units from any wall)
    near_obstacle_frac: float    # Fraction of time near an obstacle (within 3 units)
    avg_speed: float             # Average speed of hider


def simulate_episodes(
    seeker_policy, hider_policy,
    env_config: TagEnvConfig,
    n_episodes: int = 50,
    num_envs: int = 50,
    stochastic: bool = True,
) -> List[EpisodeStats]:
    """
    Simulate episodes and collect per-episode hider behavior stats.
    Uses VecTagEnv for speed.
    """
    act_fn = batch_act if stochastic else batch_act_deterministic
    env = VecTagEnv(num_envs=num_envs, config=env_config)
    obs = env.reset()

    arena_half = env_config.arena_half  # 15.0
    wall_threshold = 2.0
    corner_threshold = 3.0
    interior_threshold = 5.0
    obstacle_threshold = 3.0

    # Pre-compute obstacle centers for proximity check
    obs_centers = np.array([[o.x, o.y] for o in env.obstacles], dtype=np.float32) if env.obstacles else np.zeros((0, 2))

    # Per-env tracking
    hider_positions_history = [[] for _ in range(num_envs)]
    agent_distances_history = [[] for _ in range(num_envs)]
    hider_velocities_history = [[] for _ in range(num_envs)]
    step_counts = np.zeros(num_envs, dtype=int)

    completed_episodes: List[EpisodeStats] = []
    max_steps = 250  # safety limit (time_limit/dt/steps_per_action ~ 200)

    for step_i in range(max_steps):
        # Get actions
        seeker_actions = act_fn(seeker_policy, obs['seeker'])
        hider_actions = act_fn(hider_policy, obs['hider'])

        # Record hider positions before step
        eids = np.arange(num_envs)
        hider_idx = 1 - env.seeker_idx  # [E]
        hider_pos = env.positions[eids, hider_idx].copy()  # [E, 2]
        hider_vel = env.velocities[eids, hider_idx].copy()  # [E, 2]
        distances = env._compute_distances()

        for i in range(num_envs):
            hider_positions_history[i].append(hider_pos[i].copy())
            hider_velocities_history[i].append(hider_vel[i].copy())
            agent_distances_history[i].append(distances[i])

        step_counts += 1

        # Step
        actions = {
            'seeker': seeker_actions,
            'hider': hider_actions,
        }
        obs, rewards, dones, infos = env.step(actions)

        # Check done envs
        done_ids = np.where(dones)[0]
        for eid in done_ids:
            if len(hider_positions_history[eid]) < 3:
                # Skip degenerate short episodes
                hider_positions_history[eid] = []
                hider_velocities_history[eid] = []
                agent_distances_history[eid] = []
                step_counts[eid] = 0
                continue

            positions = np.array(hider_positions_history[eid])  # [T, 2]
            velocities = np.array(hider_velocities_history[eid])  # [T, 2]
            dist_arr = np.array(agent_distances_history[eid])  # [T]
            T = len(positions)

            # 1. Total distance traveled
            displacements = np.diff(positions, axis=0)
            step_distances = np.linalg.norm(displacements, axis=1)
            total_distance = float(np.sum(step_distances))

            # 2. Wall time: within wall_threshold of boundary at +/-arena_half
            near_wall_x = (np.abs(positions[:, 0]) > arena_half - wall_threshold)
            near_wall_y = (np.abs(positions[:, 1]) > arena_half - wall_threshold)
            near_wall = near_wall_x | near_wall_y
            wall_time_frac = float(np.mean(near_wall))

            # 3. Corner time: within corner_threshold of two walls
            near_corner_x = (np.abs(positions[:, 0]) > arena_half - corner_threshold)
            near_corner_y = (np.abs(positions[:, 1]) > arena_half - corner_threshold)
            in_corner = near_corner_x & near_corner_y
            corner_time_frac = float(np.mean(in_corner))

            # 4. Direction changes (sign changes in velocity components)
            if len(velocities) > 2:
                vx_signs = np.sign(velocities[:, 0])
                vy_signs = np.sign(velocities[:, 1])
                vx_changes = np.sum(np.abs(np.diff(vx_signs)) > 0)
                vy_changes = np.sum(np.abs(np.diff(vy_signs)) > 0)
                direction_changes = int(vx_changes + vy_changes)
            else:
                direction_changes = 0

            # 5. Survived
            survived = bool(infos['timed_out'][eid] and not infos['tagged'][eid])

            # 6. Episode length
            episode_length = T

            # 7. Agent distance stats
            avg_agent_distance = float(np.mean(dist_arr))
            std_agent_distance = float(np.std(dist_arr))
            min_agent_distance = float(np.min(dist_arr))
            max_agent_distance = float(np.max(dist_arr))

            # 8. Interior time (>5 units from any wall)
            in_interior = (np.abs(positions[:, 0]) < arena_half - interior_threshold) & \
                         (np.abs(positions[:, 1]) < arena_half - interior_threshold)
            interior_time_frac = float(np.mean(in_interior))

            # 9. Near obstacle
            if len(obs_centers) > 0:
                # [T, O] distances from hider to each obstacle center
                dists_to_obs = np.linalg.norm(
                    positions[:, None, :] - obs_centers[None, :, :], axis=2
                )
                near_any_obstacle = np.any(dists_to_obs < obstacle_threshold, axis=1)
                near_obstacle_frac = float(np.mean(near_any_obstacle))
            else:
                near_obstacle_frac = 0.0

            # 10. Average speed
            speeds = np.linalg.norm(velocities, axis=1)
            avg_speed = float(np.mean(speeds))

            completed_episodes.append(EpisodeStats(
                total_distance=total_distance,
                wall_time_frac=wall_time_frac,
                corner_time_frac=corner_time_frac,
                direction_changes=direction_changes,
                survived=survived,
                episode_length=episode_length,
                avg_agent_distance=avg_agent_distance,
                std_agent_distance=std_agent_distance,
                min_agent_distance=min_agent_distance,
                max_agent_distance=max_agent_distance,
                interior_time_frac=interior_time_frac,
                near_obstacle_frac=near_obstacle_frac,
                avg_speed=avg_speed,
            ))

            # Reset tracking for this env
            hider_positions_history[eid] = []
            hider_velocities_history[eid] = []
            agent_distances_history[eid] = []
            step_counts[eid] = 0

        # Auto-reset done envs
        if np.any(dones):
            obs = env.auto_reset()

        if len(completed_episodes) >= n_episodes:
            break

    return completed_episodes[:n_episodes]


def summarize_episodes(episodes: List[EpisodeStats]) -> Dict:
    """Compute aggregate statistics from episode list."""
    if not episodes:
        return {}

    n = len(episodes)
    return {
        'n_episodes': n,
        'survival_rate': sum(1 for e in episodes if e.survived) / n,
        'avg_ep_length': np.mean([e.episode_length for e in episodes]),
        'avg_total_distance': np.mean([e.total_distance for e in episodes]),
        'avg_wall_frac': np.mean([e.wall_time_frac for e in episodes]),
        'avg_corner_frac': np.mean([e.corner_time_frac for e in episodes]),
        'avg_interior_frac': np.mean([e.interior_time_frac for e in episodes]),
        'avg_obstacle_frac': np.mean([e.near_obstacle_frac for e in episodes]),
        'avg_direction_changes': np.mean([e.direction_changes for e in episodes]),
        'avg_agent_distance': np.mean([e.avg_agent_distance for e in episodes]),
        'avg_distance_std': np.mean([e.std_agent_distance for e in episodes]),
        'avg_min_distance': np.mean([e.min_agent_distance for e in episodes]),
        'avg_speed': np.mean([e.avg_speed for e in episodes]),
    }


def behavior_score(summary: Dict) -> Tuple[float, str]:
    """
    Score hider behavior from 0 (degenerate corner-hugger) to 100 (interesting evasion).
    Returns score and a short description.
    """
    if not summary:
        return 0.0, "no data"

    score = 0.0
    reasons = []

    # Reward survival (0-20 pts)
    sr = summary['survival_rate']
    pts = sr * 20
    score += pts
    if sr > 0.7:
        reasons.append(f"high survival ({sr:.0%})")
    elif sr < 0.3:
        reasons.append(f"low survival ({sr:.0%})")

    # Reward movement (0-20 pts): more distance = more interesting
    dist = summary['avg_total_distance']
    # A hider that traverses ~100+ units in an episode is quite active
    pts = min(dist / 100.0, 1.0) * 20
    score += pts
    if dist > 80:
        reasons.append(f"very mobile ({dist:.0f} units)")
    elif dist < 20:
        reasons.append(f"sedentary ({dist:.0f} units)")

    # Penalize corner hugging (0-20 pts): less corner time = better
    corner = summary['avg_corner_frac']
    pts = (1.0 - corner) * 20
    score += pts
    if corner > 0.5:
        reasons.append(f"corner-hugger ({corner:.0%} in corners)")
    elif corner < 0.1:
        reasons.append(f"avoids corners ({corner:.0%})")

    # Reward interior usage (0-15 pts)
    interior = summary['avg_interior_frac']
    pts = interior * 15
    score += pts
    if interior > 0.4:
        reasons.append(f"uses interior ({interior:.0%})")

    # Reward distance variation (0-15 pts): evasive = varying distance
    std_d = summary['avg_distance_std']
    pts = min(std_d / 5.0, 1.0) * 15
    score += pts
    if std_d > 4:
        reasons.append(f"evasive (dist std={std_d:.1f})")
    elif std_d < 1.5:
        reasons.append(f"static distance (std={std_d:.1f})")

    # Reward direction changes (0-10 pts)
    dc = summary['avg_direction_changes']
    pts = min(dc / 50.0, 1.0) * 10
    score += pts
    if dc > 40:
        reasons.append(f"agile ({dc:.0f} dir changes)")

    return score, "; ".join(reasons)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    BASE_DIR = PROJECT_ROOT / "experiments" / "results" / "zoo_asweep"

    # Configs to test: focus on HSM=1.15 and HSM=1.20 (fast hiders)
    # across all STP values, and a representative subset of A values
    configs = []
    stp_values = [
        ("005", 0.005),
        ("01", 0.01),
        ("02", 0.02),
        ("05", 0.05),
    ]
    hsm_values = [
        ("115", 1.15),
        ("120", 1.20),
    ]
    # Try a spread of A values (zoo size) and both sampling strategies
    a_values = ["A05", "A10", "A20", "A30", "A50", "A75", "A90"]
    sampling_modes = ["thompson_loss", "uniform"]

    print("=" * 90)
    print("HIDER BEHAVIOR ANALYSIS - Zoo A-Sweep (HSM=1.15, 1.20)")
    print("=" * 90)
    print()

    # Discover available configs
    found_configs = []
    for stp_str, stp_val in stp_values:
        for hsm_str, hsm_val in hsm_values:
            for a_str in a_values:
                for sampling in sampling_modes:
                    ckpt_dir = find_checkpoint_dir(BASE_DIR, stp_str, hsm_str, a_str, sampling)
                    if ckpt_dir is None:
                        continue
                    seeker_ckpt = find_latest_checkpoint(ckpt_dir, "seeker")
                    hider_ckpt = find_latest_checkpoint(ckpt_dir, "hider")
                    if seeker_ckpt and hider_ckpt:
                        found_configs.append({
                            'stp_str': stp_str, 'stp_val': stp_val,
                            'hsm_str': hsm_str, 'hsm_val': hsm_val,
                            'a_str': a_str, 'sampling': sampling,
                            'seeker_ckpt': seeker_ckpt,
                            'hider_ckpt': hider_ckpt,
                        })

    print(f"Found {len(found_configs)} configs with checkpoints.")
    print()

    # To keep runtime manageable, pick a representative subset:
    # For each (STP, HSM) pair, test 3 A values x 2 sampling = 6 configs
    # Pick A10 (small zoo), A30 (medium), A75 (large)
    priority_a = {"A10", "A30", "A75"}
    selected = [c for c in found_configs if c['a_str'] in priority_a]

    # If we don't have enough, fall back to all
    if len(selected) < 6:
        selected = found_configs

    print(f"Selected {len(selected)} configs for detailed analysis.")
    print()

    # Create env config template (will vary per test)
    all_results = []

    for i, cfg in enumerate(selected):
        label = f"STP{cfg['stp_str']}_HSM{cfg['hsm_str']} / {cfg['a_str']}_{cfg['sampling']}"
        print(f"[{i+1}/{len(selected)}] Evaluating: {label}")

        # Build env config
        env_config = TagEnvConfig(
            layout="four_corners",
            hider_speed_mult=cfg['hsm_val'],
            seeker_time_penalty=-cfg['stp_val'],
        )

        # Create agents and load checkpoints
        # First, create a temp env to get obs_dim
        temp_env = VecTagEnv(num_envs=1, config=env_config)
        obs_dim = temp_env.obs_dim
        act_dim = temp_env.act_dim

        ppo_cfg = PPOConfig(obs_dim=obs_dim, act_dim=act_dim)
        seeker_agent = PPOAgent(ppo_cfg)
        hider_agent = PPOAgent(ppo_cfg)

        seeker_agent.load_policy(str(cfg['seeker_ckpt']))
        hider_agent.load_policy(str(cfg['hider_ckpt']))

        # Simulate episodes
        episodes = simulate_episodes(
            seeker_policy=seeker_agent,
            hider_policy=hider_agent,
            env_config=env_config,
            n_episodes=50,
            num_envs=50,
            stochastic=True,
        )

        summary = summarize_episodes(episodes)
        score, reasons = behavior_score(summary)

        all_results.append({
            'label': label,
            'config': cfg,
            'summary': summary,
            'score': score,
            'reasons': reasons,
        })

        # Brief per-config output
        print(f"  Score: {score:.1f}/100  |  Survival: {summary['survival_rate']:.0%}  "
              f"|  Distance: {summary['avg_total_distance']:.0f}  "
              f"|  Corner: {summary['avg_corner_frac']:.0%}  "
              f"|  Interior: {summary['avg_interior_frac']:.0%}")
        print(f"  Traits: {reasons}")
        print()

    # ── Summary Table ────────────────────────────────────────────────────
    print()
    print("=" * 90)
    print("SUMMARY: ALL CONFIGS RANKED BY BEHAVIOR SCORE")
    print("=" * 90)
    print()

    # Sort by score
    all_results.sort(key=lambda r: r['score'], reverse=True)

    header = f"{'Config':<48} {'Score':>5} {'Surv%':>5} {'Dist':>5} {'Corn%':>5} {'Intr%':>5} {'Wall%':>5} {'DirCh':>5} {'Speed':>5} {'DStd':>5}"
    print(header)
    print("-" * len(header))

    for r in all_results:
        s = r['summary']
        print(f"{r['label']:<48} {r['score']:>5.1f} {s['survival_rate']*100:>5.0f} "
              f"{s['avg_total_distance']:>5.0f} {s['avg_corner_frac']*100:>5.0f} "
              f"{s['avg_interior_frac']*100:>5.0f} {s['avg_wall_frac']*100:>5.0f} "
              f"{s['avg_direction_changes']:>5.0f} {s['avg_speed']:>5.1f} "
              f"{s['avg_distance_std']:>5.1f}")

    # ── Interpretation ───────────────────────────────────────────────────
    print()
    print("=" * 90)
    print("INTERPRETATION")
    print("=" * 90)
    print()

    best = all_results[0] if all_results else None
    worst = all_results[-1] if all_results else None

    if best:
        print(f"BEST hider:  {best['label']}  (score {best['score']:.1f})")
        print(f"  {best['reasons']}")
        bs = best['summary']
        print(f"  Avg distance traveled: {bs['avg_total_distance']:.1f} units")
        print(f"  Time in corners: {bs['avg_corner_frac']:.1%}")
        print(f"  Time in interior: {bs['avg_interior_frac']:.1%}")
        print(f"  Avg agent separation: {bs['avg_agent_distance']:.1f} +/- {bs['avg_distance_std']:.1f}")
        print(f"  Direction changes: {bs['avg_direction_changes']:.0f}")
        print()

    if worst:
        print(f"WORST hider: {worst['label']}  (score {worst['score']:.1f})")
        print(f"  {worst['reasons']}")
        ws = worst['summary']
        print(f"  Avg distance traveled: {ws['avg_total_distance']:.1f} units")
        print(f"  Time in corners: {ws['avg_corner_frac']:.1%}")
        print(f"  Time in interior: {ws['avg_interior_frac']:.1%}")
        print()

    # Check if ANY config produces interesting behavior
    interesting = [r for r in all_results if r['score'] >= 50]
    corner_huggers = [r for r in all_results if r['summary']['avg_corner_frac'] > 0.4]
    mobile = [r for r in all_results if r['summary']['avg_total_distance'] > 60]
    survivors = [r for r in all_results if r['summary']['survival_rate'] > 0.7]

    print(f"Configs with INTERESTING behavior (score >= 50): {len(interesting)}/{len(all_results)}")
    for r in interesting:
        print(f"  * {r['label']}  (score {r['score']:.1f})")

    print(f"\nConfigs where hiders are CORNER-HUGGERS (>40% in corners): {len(corner_huggers)}/{len(all_results)}")
    for r in corner_huggers:
        print(f"  * {r['label']}  (corner frac {r['summary']['avg_corner_frac']:.0%})")

    print(f"\nConfigs where hiders are MOBILE (>60 units traveled): {len(mobile)}/{len(all_results)}")
    for r in mobile:
        print(f"  * {r['label']}  (distance {r['summary']['avg_total_distance']:.0f})")

    print(f"\nConfigs with HIGH SURVIVAL (>70%): {len(survivors)}/{len(all_results)}")
    for r in survivors:
        print(f"  * {r['label']}  (survival {r['summary']['survival_rate']:.0%})")

    print()
    print("=" * 90)
    if interesting:
        print("VERDICT: Some configs produce genuinely interesting hider behavior!")
        print("The project IS worth showcasing with the right config selection.")
        print(f"Best candidate for header animation: {interesting[0]['label']}")
    elif mobile:
        print("VERDICT: Some hiders move around, but behavior is mixed.")
        print("Consider cherry-picking the most mobile config for the showcase.")
    else:
        print("VERDICT: Most hiders appear to be degenerate corner-huggers.")
        print("The header animation may need a different approach (e.g., playground layout).")
    print("=" * 90)


if __name__ == "__main__":
    main()
