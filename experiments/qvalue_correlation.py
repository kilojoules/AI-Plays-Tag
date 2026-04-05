#!/usr/bin/env python3
"""
Q-value correlation analysis for the init-alpha mechanism study.

For each init_alpha condition and checkpoint, measures how well the
critic's Q-predictions correlate with actual Monte Carlo returns.

The hypothesis (H5): init_alpha=0.607 produces a critic with better
Q-value accuracy during the bootstrapping window because the entropy
term in the TD target provides dense gradient signal proportional to
actual value signal.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainer.tag_env import VecTagEnv, TagEnvConfig
from trainer.sac import SACAgent, SACConfig


def load_agent(path, obs_dim, act_dim):
    cfg = SACConfig(obs_dim=obs_dim, act_dim=act_dim)
    agent = SACAgent(cfg)
    agent.load_policy(str(path))
    return agent


def collect_mc_returns(seeker, hider, layout, num_episodes=100, gamma=0.969):
    """Run episodes and compute actual discounted returns for each (state, action)."""
    cfg = TagEnvConfig(layout=layout, hider_speed_mult=1.15)
    env = VecTagEnv(num_envs=num_episodes, config=cfg)
    obs = env.reset()

    max_steps = int(cfg.time_limit / (cfg.dt * cfg.steps_per_action))

    # Collect full trajectories
    trajectories = {role: {'obs': [], 'actions': [], 'rewards': []}
                    for role in ['seeker', 'hider']}
    active = np.ones(num_episodes, dtype=bool)

    for step in range(max_steps):
        acts = {}
        for role, agent in [('seeker', seeker), ('hider', hider)]:
            with torch.no_grad():
                x = torch.as_tensor(obs[role], dtype=torch.float32)
                actions, _ = agent.actor.sample(x)
                acts[role] = actions.cpu().numpy()

            trajectories[role]['obs'].append(obs[role].copy())
            trajectories[role]['actions'].append(acts[role].copy())

        next_obs, rewards, dones, infos = env.step(acts)

        for role in ['seeker', 'hider']:
            trajectories[role]['rewards'].append(rewards[role].copy())

        newly_done = dones & active
        active[newly_done] = False
        if not active.any():
            break

        obs = env.auto_reset()

    # Compute discounted MC returns (backward pass)
    results = {}
    for role in ['seeker', 'hider']:
        T = len(trajectories[role]['rewards'])
        all_obs = np.array(trajectories[role]['obs'])       # [T, E, obs_dim]
        all_acts = np.array(trajectories[role]['actions'])   # [T, E, act_dim]
        all_rews = np.array(trajectories[role]['rewards'])   # [T, E]

        # Compute returns G_t = sum_{k=0}^{T-t-1} gamma^k * r_{t+k}
        returns = np.zeros_like(all_rews)
        returns[-1] = all_rews[-1]
        for t in range(T - 2, -1, -1):
            returns[t] = all_rews[t] + gamma * returns[t + 1]

        results[role] = {
            'obs': all_obs,         # [T, E, obs_dim]
            'actions': all_acts,    # [T, E, act_dim]
            'returns': returns,     # [T, E]
        }

    return results


def compute_q_correlation(agent, obs, actions, mc_returns):
    """Compute correlation between critic Q-predictions and MC returns."""
    T, E = mc_returns.shape
    obs_flat = obs.reshape(-1, obs.shape[-1])
    act_flat = actions.reshape(-1, actions.shape[-1])
    ret_flat = mc_returns.reshape(-1)

    # Sample at most 2000 points to keep it fast
    n = min(len(ret_flat), 2000)
    idx = np.random.choice(len(ret_flat), n, replace=False)

    with torch.no_grad():
        o = torch.as_tensor(obs_flat[idx], dtype=torch.float32)
        a = torch.as_tensor(act_flat[idx], dtype=torch.float32)
        q1, q2 = agent.critic(o, a)
        q_pred = torch.min(q1, q2).squeeze(-1).numpy()

    mc = ret_flat[idx]

    # Filter out constant arrays
    if np.std(q_pred) < 1e-8 or np.std(mc) < 1e-8:
        return 0.0, float(np.mean(q_pred)), float(np.mean(mc)), float(np.std(q_pred))

    corr = float(np.corrcoef(q_pred, mc)[0, 1])
    return corr, float(np.mean(q_pred)), float(np.mean(mc)), float(np.std(q_pred))


def main():
    base = Path("experiments/results/paper_final/init_alpha")
    output_dir = base / "qvalue_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    obs_dim = 87
    act_dim = 3
    layout = "four_corners"

    init_alphas = [0.05, 0.2, 0.607, 2.0]
    # Checkpoints at 500K intervals
    checkpoint_steps = [500032, 1000064, 1500096, 2000128, 3000192, 5000000]

    all_results = {}

    for ia in init_alphas:
        name = f"initalpha_{ia}"
        run_dir = sorted((base / name / "seed_0").glob("2026*"))[-1]
        ckpt_dir = run_dir / "checkpoints"

        print(f"\n{'='*60}")
        print(f"Init alpha = {ia}")
        print(f"{'='*60}")

        condition_results = []

        for step in checkpoint_steps:
            seeker_path = ckpt_dir / f"seeker_{step:08d}.pt"
            hider_path = ckpt_dir / f"hider_{step:08d}.pt"

            # Use final policies if checkpoint doesn't exist at this step
            if not seeker_path.exists():
                seeker_path = run_dir / "policy_seeker_final.pt"
                hider_path = run_dir / "policy_hider_final.pt"
                if not seeker_path.exists():
                    print(f"  Step {step}: SKIP (no checkpoint)")
                    continue

            seeker = load_agent(str(seeker_path), obs_dim, act_dim)
            hider = load_agent(str(hider_path), obs_dim, act_dim)

            # Collect MC returns
            traj = collect_mc_returns(seeker, hider, layout,
                                      num_episodes=50, gamma=0.969)

            # Q-value correlation for each role
            step_result = {'step': step}
            for role, agent in [('seeker', seeker), ('hider', hider)]:
                corr, q_mean, mc_mean, q_std = compute_q_correlation(
                    agent,
                    traj[role]['obs'],
                    traj[role]['actions'],
                    traj[role]['returns'],
                )
                step_result[f'{role}_q_corr'] = corr
                step_result[f'{role}_q_mean'] = q_mean
                step_result[f'{role}_mc_mean'] = mc_mean
                step_result[f'{role}_q_std'] = q_std

            condition_results.append(step_result)

            s_corr = step_result['seeker_q_corr']
            h_corr = step_result['hider_q_corr']
            s_qm = step_result['seeker_q_mean']
            h_qm = step_result['hider_q_mean']
            print(f"  Step {step/1e6:.1f}M: S_corr={s_corr:+.3f} H_corr={h_corr:+.3f} "
                  f"S_Q={s_qm:+.1f} H_Q={h_qm:+.1f}")

        all_results[name] = condition_results

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: Q-value correlation at first checkpoint (500K)")
    print(f"{'='*60}")
    for ia in init_alphas:
        name = f"initalpha_{ia}"
        if name in all_results and all_results[name]:
            r = all_results[name][0]
            print(f"  init_alpha={ia:<6} seeker_corr={r['seeker_q_corr']:+.3f} "
                  f"hider_corr={r['hider_q_corr']:+.3f} "
                  f"seeker_Q_mean={r['seeker_q_mean']:+.2f} "
                  f"hider_Q_mean={r['hider_q_mean']:+.2f}")

    # Save
    out_path = output_dir / "qvalue_correlation.json"
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
