#!/usr/bin/env python3
"""
Cross-config gauntlet for the reward shaping study.

Every seeker (from each preset×algo, best seed) vs every hider.
Produces a win-rate matrix across all 16 configs (or however many are complete).
"""
import sys
import json
from pathlib import Path
from itertools import product

sys.path.insert(0, "/work/users/juqu/AI-Plays-Tag")

import numpy as np
import torch

from trainer.tag_env import VecTagEnv, TagEnvConfig
from trainer.ppo import PPOAgent, PPOConfig
from trainer.sac import SACAgent, SACConfig


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
    base = Path(base_dir)
    if not base.exists():
        return None
    runs = sorted(base.glob("2026*"))
    for run in reversed(runs):
        if (run / "policy_seeker_final.pt").exists():
            return run
    return None


def find_best_seed(base_dir, algo, obs_dim, act_dim):
    """Find the seed with highest seeker win rate from training metrics."""
    best_wr = -1
    best_seed = 0
    for seed in [0, 1, 2]:
        run_dir = find_latest_run(Path(base_dir) / f"seed_{seed}")
        if run_dir is None:
            continue
        csv_path = run_dir / "metrics.csv"
        if not csv_path.exists():
            continue
        # Read last few lines, get seeker win rate
        lines = csv_path.read_text().strip().split('\n')
        if len(lines) < 2:
            continue
        # Average win rate over last 5 entries
        wrs = []
        for line in lines[-5:]:
            parts = line.split(',')
            try:
                # PPO: col 8 is seeker_win_rate; SAC: col 7
                if algo == "ppo":
                    wr = float(parts[8])
                else:
                    wr = float(parts[7])
                wrs.append(wr)
            except (IndexError, ValueError):
                continue
        if wrs:
            avg_wr = np.mean(wrs)
            if avg_wr > best_wr:
                best_wr = avg_wr
                best_seed = seed
    return best_seed


def evaluate_matchup(seeker_fn, hider_fn, num_episodes=20):
    """Run episodes and return seeker win rate + mean episode length."""
    cfg = TagEnvConfig(layout="four_corners", hider_speed_mult=1.15)
    env = VecTagEnv(num_envs=num_episodes, config=cfg)
    obs = env.reset()

    max_steps = int(cfg.time_limit / (cfg.dt * cfg.steps_per_action))
    active = np.ones(num_episodes, dtype=bool)
    tagged = np.zeros(num_episodes, dtype=bool)
    lengths = np.zeros(num_episodes, dtype=int)

    # Behavioral metrics
    wall_samples = []
    speed_samples = []

    for step in range(max_steps):
        seeker_actions = seeker_fn(obs['seeker'])
        hider_actions = hider_fn(obs['hider'])

        obs, rewards, dones, infos = env.step({
            'seeker': seeker_actions,
            'hider': hider_actions,
        })

        wall_samples.append(infos['hider_near_wall_frac'])
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
        'mean_hider_speed': float(np.mean(speed_samples)),
        'num_episodes': num_episodes,
    }


def main():
    base = Path("experiments/results/reward_sweep")
    output_dir = base / "gauntlet"
    output_dir.mkdir(parents=True, exist_ok=True)

    presets = [
        "R0_baseline", "R1_seeker_pursuit", "R2_hider_active", "R3_both_shaped",
        "R4_sparse", "R5_escalating", "R6_coverage", "R7_kitchen_sink",
    ]
    algos = ["ppo", "sac"]
    obs_dim = 87
    act_dim = 3

    # Load all policies (best seed per config)
    configs = []
    seekers = {}
    hiders = {}

    for preset in presets:
        for algo in algos:
            config_key = f"{preset}/{algo}"
            config_dir = base / preset / algo

            best_seed = find_best_seed(config_dir, algo, obs_dim, act_dim)
            run_dir = find_latest_run(config_dir / f"seed_{best_seed}")
            if run_dir is None:
                print(f"SKIP {config_key}: no completed run")
                continue

            seeker_path = run_dir / "policy_seeker_final.pt"
            hider_path = run_dir / "policy_hider_final.pt"
            if not seeker_path.exists() or not hider_path.exists():
                print(f"SKIP {config_key}: missing final policies")
                continue

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

            configs.append(config_key)
            seekers[config_key] = seeker_fn
            hiders[config_key] = hider_fn
            print(f"Loaded {config_key} (seed={best_seed})")

    n = len(configs)
    print(f"\n{n} configs loaded. Running {n}x{n} = {n*n} matchups (20 eps each)...\n")

    # Run all matchups
    win_matrix = np.zeros((n, n), dtype=np.float32)
    length_matrix = np.zeros((n, n), dtype=np.float32)
    wall_matrix = np.zeros((n, n), dtype=np.float32)

    for i, seeker_cfg in enumerate(configs):
        for j, hider_cfg in enumerate(configs):
            result = evaluate_matchup(seekers[seeker_cfg], hiders[hider_cfg],
                                      num_episodes=20)
            win_matrix[i, j] = result['seeker_win_rate']
            length_matrix[i, j] = result['mean_episode_length']
            wall_matrix[i, j] = result['mean_hider_wall_frac']

            wr = result['seeker_win_rate']
            el = result['mean_episode_length']
            print(f"  S:{seeker_cfg:<30s} vs H:{hider_cfg:<30s} -> "
                  f"WR={wr:.0%}  EL={el:.0f}")

    # Save results
    results = {
        'configs': configs,
        'win_matrix': win_matrix.tolist(),
        'length_matrix': length_matrix.tolist(),
        'wall_matrix': wall_matrix.tolist(),
        'num_episodes_per_matchup': 20,
    }

    out_path = output_dir / "gauntlet_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("SEEKER STRENGTH (mean win rate as seeker across all hiders)")
    print(f"{'='*60}")
    for i, cfg in enumerate(configs):
        mean_wr = win_matrix[i, :].mean()
        print(f"  {cfg:<30s} {mean_wr:.1%}")

    print(f"\n{'='*60}")
    print("HIDER STRENGTH (mean survival rate across all seekers)")
    print(f"{'='*60}")
    for j, cfg in enumerate(configs):
        mean_surv = 1.0 - win_matrix[:, j].mean()
        print(f"  {cfg:<30s} {mean_surv:.1%}")


if __name__ == "__main__":
    main()
