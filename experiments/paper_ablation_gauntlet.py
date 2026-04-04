#!/usr/bin/env python3
"""
Cross-evaluation gauntlet for AAMAS paper ablation experiments.

Loads all trained agents from paper_ablations/ and runs every seeker
against every hider. Outputs a JSON with win-rate matrix and behavioral
metrics, plus a quick summary to stdout.

Covers:
  1A: alpha=0, alpha=0.1, control (auto) on R4_sparse — 3 conditions x 3 seeds = 9
  1B: 8 presets x 3 seeds = 24

Uses best seed per condition (highest SWR balance from training).
"""
import sys
import json
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainer.tag_env import VecTagEnv, TagEnvConfig
from trainer.sac import SACAgent, SACConfig


def load_sac_policy(path, obs_dim, act_dim):
    cfg = SACConfig(obs_dim=obs_dim, act_dim=act_dim)
    agent = SACAgent(cfg)
    agent.load_policy(str(path))
    return agent


def act_batch_sac(agent, obs_batch):
    with torch.no_grad():
        x = torch.as_tensor(obs_batch, dtype=torch.float32)
        actions, _ = agent.actor.sample(x)
        return actions.cpu().numpy()


def find_latest_run(base_dir):
    base = Path(base_dir)
    if not base.exists():
        return None
    runs = sorted(base.glob("2026*"))
    for run in reversed(runs):
        if (run / "policy_seeker_final.pt").exists():
            return run
    return None


def get_final_swr(run_dir):
    """Get average SWR from last 10 log entries."""
    csv_path = run_dir / "metrics.csv"
    if not csv_path.exists():
        return 0.5
    lines = csv_path.read_text().strip().split('\n')
    if len(lines) < 2:
        return 0.5
    wrs = []
    for line in lines[-10:]:
        parts = line.split(',')
        try:
            wrs.append(float(parts[7]))  # seeker_win_rate column
        except (IndexError, ValueError):
            continue
    return np.mean(wrs) if wrs else 0.5


def find_best_seed(base_dir):
    """Find seed with SWR closest to 0.5 (most balanced)."""
    best_balance = -1
    best_seed = 0
    for seed in [0, 1, 2]:
        run_dir = find_latest_run(Path(base_dir) / f"seed_{seed}")
        if run_dir is None:
            continue
        swr = get_final_swr(run_dir)
        balance = 1.0 - abs(swr - 0.5) * 2  # 1.0 = perfect balance
        if balance > best_balance:
            best_balance = balance
            best_seed = seed
    return best_seed


def evaluate_matchup(seeker_fn, hider_fn, num_episodes=50):
    """Run episodes and return seeker win rate + behavioral metrics."""
    cfg = TagEnvConfig(layout="four_corners", hider_speed_mult=1.15)
    env = VecTagEnv(num_envs=num_episodes, config=cfg)
    obs = env.reset()

    max_steps = int(cfg.time_limit / (cfg.dt * cfg.steps_per_action))
    active = np.ones(num_episodes, dtype=bool)
    tagged = np.zeros(num_episodes, dtype=bool)
    lengths = np.zeros(num_episodes, dtype=int)

    wall_samples = []
    speed_samples = []
    corner_samples = []
    wall_speed_samples = []

    for step in range(max_steps):
        seeker_actions = seeker_fn(obs['seeker'])
        hider_actions = hider_fn(obs['hider'])

        obs, rewards, dones, infos = env.step({
            'seeker': seeker_actions,
            'hider': hider_actions,
        })

        wall_samples.append(infos['hider_near_wall_frac'])
        speed_samples.append(infos['hider_speed_mean'])
        corner_samples.append(infos['hider_corner_frac'])
        wall_speed_samples.append(infos['hider_wall_speed_mean'])

        newly_done = dones & active
        if np.any(newly_done):
            for i in np.where(newly_done)[0]:
                lengths[i] = step + 1
                tagged[i] = infos['tagged'][i]
            active[newly_done] = False

        if not np.any(active):
            break

    for i in range(num_episodes):
        if active[i]:
            lengths[i] = max_steps

    return {
        'seeker_win_rate': float(tagged.mean()),
        'mean_episode_length': float(lengths.mean()),
        'mean_hider_wall_frac': float(np.mean(wall_samples)),
        'mean_hider_corner_frac': float(np.mean(corner_samples)),
        'mean_hider_speed': float(np.mean(speed_samples)),
        'mean_hider_wall_speed': float(np.mean(wall_speed_samples)),
    }


def main():
    base = Path("experiments/results/paper_ablations")
    output_dir = base / "gauntlet"
    output_dir.mkdir(parents=True, exist_ok=True)

    obs_dim = 87
    act_dim = 3

    # --- Discover all configs ---
    configs = []
    agents = {}  # key -> (seeker_fn, hider_fn)

    # 1A: counterfactual conditions
    for condition in ["alpha0", "alpha01", "control"]:
        config_dir = base / "1A_counterfactual" / "R4_sparse" / condition
        best_seed = find_best_seed(config_dir)
        run_dir = find_latest_run(config_dir / f"seed_{best_seed}")
        if run_dir is None:
            print(f"SKIP 1A/{condition}: no run found")
            continue

        key = f"1A/{condition}"
        seeker = load_sac_policy(run_dir / "policy_seeker_final.pt", obs_dim, act_dim)
        hider = load_sac_policy(run_dir / "policy_hider_final.pt", obs_dim, act_dim)
        configs.append(key)
        agents[key] = (
            lambda obs, s=seeker: act_batch_sac(s, obs),
            lambda obs, h=hider: act_batch_sac(h, obs),
        )
        swr = get_final_swr(run_dir)
        print(f"Loaded {key} (seed={best_seed}, train_SWR={swr:.1%})")

    # 1B: all presets
    presets = [
        "R0_baseline", "R1_seeker_pursuit", "R2_hider_active", "R3_both_shaped",
        "R4_sparse", "R5_escalating", "R6_coverage", "R7_kitchen_sink",
    ]
    for preset in presets:
        config_dir = base / "1B_alpha_dynamics" / preset / "auto"
        best_seed = find_best_seed(config_dir)
        run_dir = find_latest_run(config_dir / f"seed_{best_seed}")
        if run_dir is None:
            print(f"SKIP 1B/{preset}: no run found")
            continue

        key = f"1B/{preset}"
        seeker = load_sac_policy(run_dir / "policy_seeker_final.pt", obs_dim, act_dim)
        hider = load_sac_policy(run_dir / "policy_hider_final.pt", obs_dim, act_dim)
        configs.append(key)
        agents[key] = (
            lambda obs, s=seeker: act_batch_sac(s, obs),
            lambda obs, h=hider: act_batch_sac(h, obs),
        )
        swr = get_final_swr(run_dir)
        print(f"Loaded {key} (seed={best_seed}, train_SWR={swr:.1%})")

    n = len(configs)
    print(f"\n{n} configs loaded. Running {n}x{n} = {n*n} matchups (50 eps each)...\n")

    # --- Run all matchups ---
    win_matrix = np.zeros((n, n), dtype=np.float32)
    length_matrix = np.zeros((n, n), dtype=np.float32)
    wall_matrix = np.zeros((n, n), dtype=np.float32)
    corner_matrix = np.zeros((n, n), dtype=np.float32)
    speed_matrix = np.zeros((n, n), dtype=np.float32)
    wall_speed_matrix = np.zeros((n, n), dtype=np.float32)

    for i, s_cfg in enumerate(configs):
        for j, h_cfg in enumerate(configs):
            result = evaluate_matchup(
                agents[s_cfg][0], agents[h_cfg][1], num_episodes=50)
            win_matrix[i, j] = result['seeker_win_rate']
            length_matrix[i, j] = result['mean_episode_length']
            wall_matrix[i, j] = result['mean_hider_wall_frac']
            corner_matrix[i, j] = result['mean_hider_corner_frac']
            speed_matrix[i, j] = result['mean_hider_speed']
            wall_speed_matrix[i, j] = result['mean_hider_wall_speed']

            wr = result['seeker_win_rate']
            el = result['mean_episode_length']
            cr = result['mean_hider_corner_frac']
            print(f"  S:{s_cfg:<25s} vs H:{h_cfg:<25s} -> WR={wr:.0%}  EL={el:.0f}  Corner={cr:.2f}")

    # --- Save results ---
    results = {
        'configs': configs,
        'win_matrix': win_matrix.tolist(),
        'length_matrix': length_matrix.tolist(),
        'wall_matrix': wall_matrix.tolist(),
        'corner_matrix': corner_matrix.tolist(),
        'speed_matrix': speed_matrix.tolist(),
        'wall_speed_matrix': wall_speed_matrix.tolist(),
        'num_episodes_per_matchup': 50,
    }

    out_path = output_dir / "gauntlet_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print("SEEKER STRENGTH (mean WR as seeker across all hiders)")
    print(f"{'='*60}")
    for i, cfg in enumerate(configs):
        mean_wr = win_matrix[i, :].mean()
        print(f"  {cfg:<30s} {mean_wr:.1%}")

    print(f"\n{'='*60}")
    print("HIDER STRENGTH (mean survival across all seekers)")
    print(f"{'='*60}")
    for j, cfg in enumerate(configs):
        mean_surv = 1.0 - win_matrix[:, j].mean()
        print(f"  {cfg:<30s} {mean_surv:.1%}")

    # 1A-specific comparison
    print(f"\n{'='*60}")
    print("1A COUNTERFACTUAL: alpha=0 vs alpha=0.1 vs auto")
    print(f"{'='*60}")
    for cfg in configs:
        if cfg.startswith("1A/"):
            idx = configs.index(cfg)
            s_str = f"seeker={win_matrix[idx, :].mean():.1%}"
            h_str = f"hider_surv={1.0 - win_matrix[:, idx].mean():.1%}"
            w_str = f"wall={wall_matrix[:, idx].mean():.2f}"
            c_str = f"corner={corner_matrix[:, idx].mean():.2f}"
            ws_str = f"wall_spd={wall_speed_matrix[:, idx].mean():.1f}"
            print(f"  {cfg:<20s} {s_str}  {h_str}  {w_str}  {c_str}  {ws_str}")


if __name__ == "__main__":
    main()
