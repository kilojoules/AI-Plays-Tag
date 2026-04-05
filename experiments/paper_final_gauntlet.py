#!/usr/bin/env python3
"""
Cross-evaluation gauntlet for final paper experiments.

Part 1: Geometry study — evaluate each layout's agents within their own layout.
  9 layouts, best seed per layout, self-play eval (50 episodes).

Part 2: Init-alpha — cross-eval all 4 init-alpha conditions on four_corners.
  4 conditions, best seed each, 4x4 matrix (50 episodes per matchup).
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
    csv_path = run_dir / "metrics.csv"
    if not csv_path.exists():
        return 0.5
    lines = csv_path.read_text().strip().split('\n')
    wrs = []
    for line in lines[-10:]:
        parts = line.split(',')
        try:
            wrs.append(float(parts[7]))
        except (IndexError, ValueError):
            continue
    return np.mean(wrs) if wrs else 0.5


def find_best_seed(base_dir):
    best_balance = -1
    best_seed = 0
    for seed in [0, 1, 2]:
        run_dir = find_latest_run(Path(base_dir) / f"seed_{seed}")
        if run_dir is None:
            continue
        swr = get_final_swr(run_dir)
        balance = 1.0 - abs(swr - 0.5) * 2
        if balance > best_balance:
            best_balance = balance
            best_seed = seed
    return best_seed


def evaluate_matchup(seeker_fn, hider_fn, layout, num_episodes=50):
    cfg = TagEnvConfig(layout=layout, hider_speed_mult=1.15)
    env = VecTagEnv(num_envs=num_episodes, config=cfg)
    obs = env.reset()

    max_steps = int(cfg.time_limit / (cfg.dt * cfg.steps_per_action))
    active = np.ones(num_episodes, dtype=bool)
    tagged = np.zeros(num_episodes, dtype=bool)
    lengths = np.zeros(num_episodes, dtype=int)
    wall_samples, corner_samples, speed_samples = [], [], []

    for step in range(max_steps):
        seeker_actions = seeker_fn(obs['seeker'])
        hider_actions = hider_fn(obs['hider'])
        obs, rewards, dones, infos = env.step({
            'seeker': seeker_actions, 'hider': hider_actions})

        wall_samples.append(infos['hider_near_wall_frac'])
        corner_samples.append(infos['hider_corner_frac'])
        speed_samples.append(infos['hider_speed_mean'])

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
    }


def main():
    base = Path("experiments/results/paper_final")
    output_dir = base / "gauntlet"
    output_dir.mkdir(parents=True, exist_ok=True)

    obs_dim = 87
    act_dim = 3

    # =====================================================================
    # Part 1: Geometry study
    # =====================================================================
    print("=" * 60)
    print("PART 1: Geometry Study")
    print("=" * 60)

    layouts = ["empty", "one_corner", "two_corners", "central_cross",
               "wall_midpoints", "four_corners", "corner_tight",
               "center_cluster", "playground"]

    geo_results = {}

    for layout in layouts:
        config_dir = base / "geometry" / f"geo_{layout}"
        best_seed = find_best_seed(config_dir)
        run_dir = find_latest_run(config_dir / f"seed_{best_seed}")
        if run_dir is None:
            print(f"  SKIP {layout}: no run found")
            continue

        seeker = load_sac_policy(run_dir / "policy_seeker_final.pt", obs_dim, act_dim)
        hider = load_sac_policy(run_dir / "policy_hider_final.pt", obs_dim, act_dim)
        seeker_fn = lambda obs, s=seeker: act_batch_sac(s, obs)
        hider_fn = lambda obs, h=hider: act_batch_sac(h, obs)

        # Evaluate within this layout
        result = evaluate_matchup(seeker_fn, hider_fn, layout=layout, num_episodes=100)
        geo_results[layout] = result
        swr = result['seeker_win_rate']
        cr = result['mean_hider_corner_frac']
        wf = result['mean_hider_wall_frac']
        sp = result['mean_hider_speed']
        el = result['mean_episode_length']
        print(f"  {layout:<18s} seed={best_seed} WR={swr:.0%} Corner={cr:.2f} "
              f"Wall={wf:.2f} Speed={sp:.1f} EpLen={el:.0f}")

    # =====================================================================
    # Part 2: Init-alpha cross-eval
    # =====================================================================
    print(f"\n{'=' * 60}")
    print("PART 2: Init-Alpha Cross-Evaluation")
    print("=" * 60)

    alphas = [0.05, 0.2, 0.607, 2.0]
    alpha_agents = {}
    alpha_configs = []

    for ia in alphas:
        name = f"initalpha_{ia}"
        config_dir = base / "init_alpha" / name
        best_seed = find_best_seed(config_dir)
        run_dir = find_latest_run(config_dir / f"seed_{best_seed}")
        if run_dir is None:
            print(f"  SKIP {name}")
            continue

        seeker = load_sac_policy(run_dir / "policy_seeker_final.pt", obs_dim, act_dim)
        hider = load_sac_policy(run_dir / "policy_hider_final.pt", obs_dim, act_dim)
        alpha_configs.append(name)
        alpha_agents[name] = (
            lambda obs, s=seeker: act_batch_sac(s, obs),
            lambda obs, h=hider: act_batch_sac(h, obs),
        )
        swr = get_final_swr(run_dir)
        print(f"  Loaded {name} (seed={best_seed}, train_SWR={swr:.1%})")

    n = len(alpha_configs)
    win_matrix = np.zeros((n, n), dtype=np.float32)
    corner_matrix = np.zeros((n, n), dtype=np.float32)
    speed_matrix = np.zeros((n, n), dtype=np.float32)

    print(f"\n  Running {n}x{n} = {n*n} matchups...")
    for i, s_cfg in enumerate(alpha_configs):
        for j, h_cfg in enumerate(alpha_configs):
            r = evaluate_matchup(
                alpha_agents[s_cfg][0], alpha_agents[h_cfg][1],
                layout="four_corners", num_episodes=50)
            win_matrix[i, j] = r['seeker_win_rate']
            corner_matrix[i, j] = r['mean_hider_corner_frac']
            speed_matrix[i, j] = r['mean_hider_speed']
            print(f"    S:{s_cfg:<20s} vs H:{h_cfg:<20s} WR={r['seeker_win_rate']:.0%} "
                  f"Corner={r['mean_hider_corner_frac']:.2f}")

    # Summary
    print(f"\n  {'Config':<20s} {'Seeker':>7s} {'Surv':>6s} {'Comb':>6s} {'Corner':>7s}")
    print("  " + "-" * 50)
    for j, cfg in enumerate(alpha_configs):
        sk = win_matrix[j, :].mean()
        surv = 1.0 - win_matrix[:, j].mean()
        comb = (sk + surv) / 2
        cr = corner_matrix[:, j].mean()
        print(f"  {cfg:<20s} {sk:>6.1%} {surv:>5.1%} {comb:>5.1%} {cr:>7.2f}")

    # Save everything
    results = {
        'geometry': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in geo_results.items()},
        'init_alpha': {
            'configs': alpha_configs,
            'win_matrix': win_matrix.tolist(),
            'corner_matrix': corner_matrix.tolist(),
            'speed_matrix': speed_matrix.tolist(),
        },
    }

    out_path = output_dir / "paper_final_gauntlet.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
