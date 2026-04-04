#!/usr/bin/env python3
"""
Cross-evaluation gauntlet for responsiveness intrinsic reward experiments.

Loads best seed per condition, runs all-vs-all, outputs JSON with
win-rate matrix and behavioral metrics (including corner camping).
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
    if len(lines) < 2:
        return 0.5
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


def evaluate_matchup(seeker_fn, hider_fn, num_episodes=50):
    cfg = TagEnvConfig(layout="four_corners", hider_speed_mult=1.15)
    env = VecTagEnv(num_envs=num_episodes, config=cfg)
    obs = env.reset()

    max_steps = int(cfg.time_limit / (cfg.dt * cfg.steps_per_action))
    active = np.ones(num_episodes, dtype=bool)
    tagged = np.zeros(num_episodes, dtype=bool)
    lengths = np.zeros(num_episodes, dtype=int)

    wall_samples, corner_samples, speed_samples, wall_speed_samples = [], [], [], []

    for step in range(max_steps):
        seeker_actions = seeker_fn(obs['seeker'])
        hider_actions = hider_fn(obs['hider'])

        obs, rewards, dones, infos = env.step({
            'seeker': seeker_actions, 'hider': hider_actions,
        })

        wall_samples.append(infos['hider_near_wall_frac'])
        corner_samples.append(infos['hider_corner_frac'])
        speed_samples.append(infos['hider_speed_mean'])
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
    base = Path("experiments/results/responsiveness_sweep")
    output_dir = base / "gauntlet"
    output_dir.mkdir(parents=True, exist_ok=True)

    obs_dim = 87
    act_dim = 3

    condition_names = [
        "R4_sparse_baseline", "R2_active_baseline",
        "R4_te_005", "R4_te_01", "R4_te_02",
        "R4_kl_005", "R4_kl_01", "R4_kl_02",
        "R4_both_005", "R4_both_01", "R4_both_02",
    ]

    configs = []
    agents = {}

    for name in condition_names:
        config_dir = base / name
        best_seed = find_best_seed(config_dir)
        run_dir = find_latest_run(config_dir / f"seed_{best_seed}")
        if run_dir is None:
            print(f"SKIP {name}: no run found")
            continue

        seeker = load_sac_policy(run_dir / "policy_seeker_final.pt", obs_dim, act_dim)
        hider = load_sac_policy(run_dir / "policy_hider_final.pt", obs_dim, act_dim)
        configs.append(name)
        agents[name] = (
            lambda obs, s=seeker: act_batch_sac(s, obs),
            lambda obs, h=hider: act_batch_sac(h, obs),
        )
        swr = get_final_swr(run_dir)
        print(f"Loaded {name} (seed={best_seed}, train_SWR={swr:.1%})")

    n = len(configs)
    print(f"\n{n} configs. Running {n}x{n} = {n*n} matchups (50 eps each)...\n")

    matrices = {k: np.zeros((n, n), dtype=np.float32)
                for k in ['win', 'length', 'wall', 'corner', 'speed', 'wall_speed']}

    for i, s_cfg in enumerate(configs):
        for j, h_cfg in enumerate(configs):
            r = evaluate_matchup(agents[s_cfg][0], agents[h_cfg][1], num_episodes=50)
            matrices['win'][i, j] = r['seeker_win_rate']
            matrices['length'][i, j] = r['mean_episode_length']
            matrices['wall'][i, j] = r['mean_hider_wall_frac']
            matrices['corner'][i, j] = r['mean_hider_corner_frac']
            matrices['speed'][i, j] = r['mean_hider_speed']
            matrices['wall_speed'][i, j] = r['mean_hider_wall_speed']
            print(f"  S:{s_cfg:<25s} vs H:{h_cfg:<25s} -> "
                  f"WR={r['seeker_win_rate']:.0%} Corner={r['mean_hider_corner_frac']:.2f}")

    results = {
        'configs': configs,
        **{f'{k}_matrix': v.tolist() for k, v in matrices.items()},
        'num_episodes_per_matchup': 50,
    }

    out_path = output_dir / "gauntlet_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")

    # Summary
    W = matrices['win']
    C = matrices['corner']
    S = matrices['speed']

    print(f"\n{'Config':<25s} {'Seeker':>7s} {'Surv':>6s} {'Comb':>6s} {'Corner':>7s} {'Speed':>6s}")
    print('-' * 62)
    for j, cfg in enumerate(configs):
        sk = W[j, :].mean()
        surv = 1.0 - W[:, j].mean()
        comb = (sk + surv) / 2
        cr = C[:, j].mean()
        sp = S[:, j].mean()
        print(f"{cfg:<25s} {sk:>6.1%} {surv:>5.1%} {comb:>5.1%} {cr:>7.2f} {sp:>6.1f}")


if __name__ == "__main__":
    main()
