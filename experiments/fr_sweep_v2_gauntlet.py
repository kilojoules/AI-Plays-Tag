#!/usr/bin/env python3
"""
Cross-config gauntlet for FR sweep v2.

Picks best seed per (preset, algo, A) config, then runs all-vs-all matchups.
Answers: does A (zoo mixing) produce stronger agents?
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, "/work/users/juqu/AI-Plays-Tag")

import numpy as np
import torch

from trainer.tag_env import VecTagEnv, TagEnvConfig
from trainer.ppo import PPOAgent, PPOConfig
from trainer.sac import SACAgent, SACConfig


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


def load_policy(path, algo, obs_dim, act_dim):
    if algo == "ppo":
        cfg = PPOConfig(obs_dim=obs_dim, act_dim=act_dim)
        policy = PPOAgent(cfg)
        ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
        policy.pi.load_state_dict(ckpt["pi"])
        policy.vf.load_state_dict(ckpt["vf"])
        return policy, lambda obs, p=policy: act_batch_ppo(p, obs)
    else:
        cfg = SACConfig(obs_dim=obs_dim, act_dim=act_dim)
        agent = SACAgent(cfg)
        agent.load_policy(str(path))
        return agent, lambda obs, a=agent: act_batch_sac(a, obs)


def evaluate_matchup(seeker_fn, hider_fn, num_episodes=50):
    cfg = TagEnvConfig(layout="four_corners", hider_speed_mult=1.15)
    env = VecTagEnv(num_envs=num_episodes, config=cfg)
    obs = env.reset()
    max_steps = int(cfg.time_limit / (cfg.dt * cfg.steps_per_action))
    active = np.ones(num_episodes, dtype=bool)
    tagged = np.zeros(num_episodes, dtype=bool)
    lengths = np.zeros(num_episodes, dtype=int)

    for step in range(max_steps):
        s_acts = seeker_fn(obs['seeker'])
        h_acts = hider_fn(obs['hider'])
        obs, rewards, dones, infos = env.step({'seeker': s_acts, 'hider': h_acts})
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

    return float(tagged.mean()), float(lengths.mean())


def find_best_seed(base_dir, algo):
    """Pick seed with SWR closest to 0.5 (most balanced = likely strongest both sides)."""
    best_score = -1
    best_run = None
    for seed_dir in sorted(Path(base_dir).iterdir()):
        if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
            continue
        for run_dir in sorted(seed_dir.iterdir()):
            mpath = run_dir / "metrics.csv"
            if not mpath.exists():
                continue
            with open(mpath) as f:
                rows = list(csv.reader(f))
                if len(rows) < 2:
                    continue
                swr = float(rows[-1][8])
                balance = min(swr, 1 - swr)
                if balance > best_score:
                    best_score = balance
                    best_run = run_dir
    return best_run, best_score


def main():
    base = Path("experiments/results/fr_sweep_v2")
    output_dir = base / "gauntlet"
    output_dir.mkdir(parents=True, exist_ok=True)

    presets = ["R0_baseline", "R1_seeker_pursuit", "R3_both_shaped",
               "R4_sparse", "R5_escalating"]
    algos = ["ppo", "sac"]
    a_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    obs_dim = 87
    act_dim = 3

    # Load best-seed policy per config
    configs = []
    seeker_fns = {}
    hider_fns = {}

    for preset in presets:
        for algo in algos:
            for a_val in a_values:
                key = f"{preset}/{algo}/A={a_val:.2f}"
                config_dir = base / preset / f"A{a_val:.2f}_{algo}"
                if not config_dir.exists():
                    continue

                run_dir, balance = find_best_seed(config_dir, algo)
                if run_dir is None:
                    print(f"SKIP {key}: no run found")
                    continue

                seeker_path = run_dir / "policy_seeker_final.pt"
                hider_path = run_dir / "policy_hider_final.pt"
                if not seeker_path.exists() or not hider_path.exists():
                    print(f"SKIP {key}: missing policies")
                    continue

                _, s_fn = load_policy(seeker_path, algo, obs_dim, act_dim)
                _, h_fn = load_policy(hider_path, algo, obs_dim, act_dim)
                configs.append(key)
                seeker_fns[key] = s_fn
                hider_fns[key] = h_fn
                print(f"Loaded {key} (balance={balance:.3f})")

    n = len(configs)
    print(f"\n{n} configs loaded. Running {n}x{n} = {n*n} matchups...\n")

    # Run gauntlet
    win_matrix = np.zeros((n, n), dtype=np.float32)
    len_matrix = np.zeros((n, n), dtype=np.float32)

    for i, s_cfg in enumerate(configs):
        for j, h_cfg in enumerate(configs):
            wr, el = evaluate_matchup(seeker_fns[s_cfg], hider_fns[h_cfg])
            win_matrix[i, j] = wr
            len_matrix[i, j] = el
        # Progress
        s_mean = win_matrix[i, :].mean()
        print(f"  [{i+1}/{n}] Seeker {s_cfg}: mean WR={s_mean:.1%}")

    # Save raw results
    results = {
        'configs': configs,
        'win_matrix': win_matrix.tolist(),
        'length_matrix': len_matrix.tolist(),
    }
    with open(output_dir / "gauntlet_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # === Analysis ===
    print(f"\n{'='*70}")
    print("SEEKER STRENGTH (mean win rate across ALL hiders)")
    print(f"{'='*70}")
    seeker_ranks = sorted(range(n), key=lambda i: win_matrix[i, :].mean(), reverse=True)
    for rank, i in enumerate(seeker_ranks):
        print(f"  {rank+1:2d}. {configs[i]:40s} WR={win_matrix[i,:].mean():.1%}")

    print(f"\n{'='*70}")
    print("HIDER STRENGTH (mean survival rate across ALL seekers)")
    print(f"{'='*70}")
    hider_ranks = sorted(range(n), key=lambda j: (1 - win_matrix[:, j].mean()), reverse=True)
    for rank, j in enumerate(hider_ranks):
        surv = 1 - win_matrix[:, j].mean()
        print(f"  {rank+1:2d}. {configs[j]:40s} SURV={surv:.1%}")

    # Group by A value (across presets/algos)
    print(f"\n{'='*70}")
    print("A VALUE EFFECT ON STRENGTH")
    print(f"{'='*70}")
    for a_val in a_values:
        # Seeker strength for this A
        s_indices = [i for i, c in enumerate(configs) if f"A={a_val:.2f}" in c]
        if s_indices:
            s_mean = np.mean([win_matrix[i, :].mean() for i in s_indices])
            h_mean = np.mean([1 - win_matrix[:, j].mean() for j in s_indices])
            print(f"  A={a_val:.2f}: seeker_strength={s_mean:.1%}, hider_strength={h_mean:.1%}")

    # Group by A value per algo
    for algo in algos:
        print(f"\n  {algo.upper()}:")
        for a_val in a_values:
            s_indices = [i for i, c in enumerate(configs) if f"/{algo}/A={a_val:.2f}" in c]
            if s_indices:
                s_mean = np.mean([win_matrix[i, :].mean() for i in s_indices])
                h_mean = np.mean([1 - win_matrix[:, j].mean() for j in s_indices])
                print(f"    A={a_val:.2f}: seeker={s_mean:.1%}, hider={h_mean:.1%}")

    # Group by preset
    print(f"\n{'='*70}")
    print("PRESET EFFECT ON STRENGTH")
    print(f"{'='*70}")
    for preset in presets:
        s_indices = [i for i, c in enumerate(configs) if c.startswith(preset)]
        if s_indices:
            s_mean = np.mean([win_matrix[i, :].mean() for i in s_indices])
            h_mean = np.mean([1 - win_matrix[:, j].mean() for j in s_indices])
            print(f"  {preset:25s}: seeker={s_mean:.1%}, hider={h_mean:.1%}")


if __name__ == "__main__":
    main()
