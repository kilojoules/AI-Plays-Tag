#!/usr/bin/env python3
"""
Critic transplant experiment for the init-alpha mechanism study.

Tests whether the strength difference between init_alpha conditions
is stored in the critic or the actor by swapping components between
trained agents and continuing training.

Conditions:
  - control_607: normal init_alpha=0.607 (baseline, should be strongest)
  - control_005: normal init_alpha=0.05 (baseline, should be weaker)
  - transplant_607critic: 0.607's critic + 0.05's actor, continue training
  - transplant_005critic: 0.05's critic + 0.607's actor, continue training

If transplanting 0.607's critic recovers the strength gap, the mechanism
is critic conditioning (H1/H5). If not, it's actor/optimizer based.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "trainer"))

from tag_env import VecTagEnv, TagEnvConfig
from sac import SACConfig, SACAgent, ReplayBuffer


def create_transplant_agent(critic_path: str, actor_path: str,
                            obs_dim: int, act_dim: int) -> SACAgent:
    """Create an agent with critic from one checkpoint and actor from another."""
    cfg = SACConfig(obs_dim=obs_dim, act_dim=act_dim)
    agent = SACAgent(cfg)

    # Load critic source
    critic_state = torch.load(critic_path, map_location="cpu")
    agent.critic.load_state_dict(critic_state['critic'])
    agent.critic_target.load_state_dict(critic_state['critic_target'])

    # Load actor source
    actor_state = torch.load(actor_path, map_location="cpu")
    agent.actor.load_state_dict(actor_state['actor'])

    # Use critic source's alpha (it reflects the critic's learned value scale)
    if 'log_alpha' in critic_state:
        agent.log_alpha.data.copy_(critic_state['log_alpha'])

    return agent


def train_from_agents(seeker: SACAgent, hider: SACAgent,
                      total_timesteps: int, num_envs: int,
                      output_dir: str, condition_name: str):
    """Train from pre-initialized agents."""
    env_config = TagEnvConfig(
        layout="four_corners", hider_speed_mult=1.15,
        seeker_time_penalty=0.0, distance_reward_scale=0.0,
        hider_dist_reward_scale=0.0, hider_abs_dist_reward_scale=0.0,
        runner_survival_bonus=0.0,
    )
    env = VecTagEnv(num_envs=num_envs, config=env_config)
    obs_dim = env.obs_dim
    act_dim = env.act_dim

    buffers = {
        'seeker': ReplayBuffer(obs_dim, act_dim, 100000),
        'hider': ReplayBuffer(obs_dim, act_dim, 100000),
    }
    agents = {'seeker': seeker, 'hider': hider}

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(output_dir, run_id)
    os.makedirs(os.path.join(out, "checkpoints"), exist_ok=True)
    metrics_path = os.path.join(out, "metrics.csv")

    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "timesteps", "episodes", "seeker_reward_mean", "hider_reward_mean",
            "episode_length_mean", "seeker_win_rate",
            "seeker_alpha", "hider_alpha", "fps",
        ])

    print(f"Training {condition_name}: {total_timesteps} steps")
    print(f"  Output: {out}")

    obs = env.reset()
    timesteps = 0
    start_time = time.time()
    total_episodes = 0
    window_s_rew, window_h_rew, window_lens = [], [], []
    window_s_wins, window_h_wins = 0, 0
    ep_rewards = {'seeker': np.zeros(num_envs), 'hider': np.zeros(num_envs)}
    ep_lengths = np.zeros(num_envs, dtype=np.int32)
    train_infos = {}

    batch_size = 256
    warmup = 5000
    updates_per_step = 4

    while timesteps < total_timesteps:
        is_warmup = timesteps < warmup
        acts = {}
        for role in ['seeker', 'hider']:
            if is_warmup:
                acts[role] = np.random.uniform(-1, 1, (num_envs, act_dim)).astype(np.float32)
            else:
                with torch.no_grad():
                    x = torch.as_tensor(obs[role], dtype=torch.float32)
                    a, _ = agents[role].actor.sample(x)
                    acts[role] = a.cpu().numpy()

        next_obs, rewards, dones, infos = env.step(acts)
        timesteps += num_envs

        for role in ['seeker', 'hider']:
            buffers[role].add_batch(obs[role], acts[role], rewards[role],
                                    next_obs[role], dones.astype(np.float32))
            ep_rewards[role] += rewards[role]
        ep_lengths += 1

        for eid in np.where(dones)[0]:
            for role in ['seeker', 'hider']:
                if role == 'seeker':
                    window_s_rew.append(ep_rewards[role][eid])
                else:
                    window_h_rew.append(ep_rewards[role][eid])
                ep_rewards[role][eid] = 0
            window_lens.append(ep_lengths[eid])
            ep_lengths[eid] = 0
            total_episodes += 1
            if infos['tagged'][eid]:
                window_s_wins += 1
            else:
                window_h_wins += 1

        obs = env.auto_reset()

        if not is_warmup:
            for _ in range(updates_per_step):
                for role in ['seeker', 'hider']:
                    if buffers[role].size >= batch_size:
                        batch = buffers[role].sample(batch_size)
                        train_infos[role] = agents[role].update(batch)

        if timesteps % 5000 < num_envs:
            elapsed = time.time() - start_time
            fps = timesteps / elapsed if elapsed > 0 else 0
            total = window_s_wins + window_h_wins
            swr = window_s_wins / max(total, 1)
            s_info = train_infos.get('seeker', {})
            h_info = train_infos.get('hider', {})

            row = [
                timesteps, total_episodes,
                np.mean(window_s_rew) if window_s_rew else 0,
                np.mean(window_h_rew) if window_h_rew else 0,
                np.mean(window_lens) if window_lens else 0,
                swr,
                s_info.get('alpha', 0), h_info.get('alpha', 0),
                fps,
            ]
            with open(metrics_path, "a", newline="") as f:
                csv.writer(f).writerow(row)

            print(f"  Steps: {timesteps:8d} | SWR: {swr:.0%} | "
                  f"α_s: {s_info.get('alpha', 0):.4f} | FPS: {fps:.0f}")

            window_s_rew, window_h_rew, window_lens = [], [], []
            window_s_wins, window_h_wins = 0, 0

    # Save final policies
    for role in ['seeker', 'hider']:
        agents[role].save_policy(os.path.join(out, f"policy_{role}_final.pt"))

    elapsed = time.time() - start_time
    print(f"  Done! {timesteps} steps in {elapsed:.1f}s ({timesteps/elapsed:.0f} FPS)\n")
    return out


def main():
    parser = argparse.ArgumentParser(description="Critic transplant experiment")
    parser.add_argument("--task-id", type=int, required=True,
                        help="0=control_607, 1=control_005, 2=transplant_607critic, 3=transplant_005critic")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str,
                        default="experiments/results/paper_final/critic_transplant")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    obs_dim = 87
    act_dim = 3
    base = Path("experiments/results/paper_final/init_alpha")

    # Find checkpoints — use the best seed from the gauntlet
    def get_final_path(ia, role):
        name = f"initalpha_{ia}"
        # Use seed_0 for simplicity (or best seed)
        runs = sorted((base / name / "seed_0").glob("2026*"))
        if not runs:
            raise FileNotFoundError(f"No runs for {name}/seed_0")
        return str(runs[-1] / f"policy_{role}_final.pt")

    conditions = {
        0: {
            "name": "control_607",
            "seeker_critic": get_final_path(0.607, "seeker"),
            "seeker_actor": get_final_path(0.607, "seeker"),
            "hider_critic": get_final_path(0.607, "hider"),
            "hider_actor": get_final_path(0.607, "hider"),
        },
        1: {
            "name": "control_005",
            "seeker_critic": get_final_path(0.05, "seeker"),
            "seeker_actor": get_final_path(0.05, "seeker"),
            "hider_critic": get_final_path(0.05, "hider"),
            "hider_actor": get_final_path(0.05, "hider"),
        },
        2: {
            "name": "transplant_607critic",
            "seeker_critic": get_final_path(0.607, "seeker"),
            "seeker_actor": get_final_path(0.05, "seeker"),
            "hider_critic": get_final_path(0.607, "hider"),
            "hider_actor": get_final_path(0.05, "hider"),
        },
        3: {
            "name": "transplant_005critic",
            "seeker_critic": get_final_path(0.05, "seeker"),
            "seeker_actor": get_final_path(0.607, "seeker"),
            "hider_critic": get_final_path(0.05, "hider"),
            "hider_actor": get_final_path(0.607, "hider"),
        },
    }

    cond = conditions[args.task_id]
    print(f"Condition: {cond['name']}")
    print(f"  Seeker critic: {cond['seeker_critic']}")
    print(f"  Seeker actor:  {cond['seeker_actor']}")
    print(f"  Hider critic:  {cond['hider_critic']}")
    print(f"  Hider actor:   {cond['hider_actor']}")

    seeker = create_transplant_agent(
        cond['seeker_critic'], cond['seeker_actor'], obs_dim, act_dim)
    hider = create_transplant_agent(
        cond['hider_critic'], cond['hider_actor'], obs_dim, act_dim)

    out_dir = os.path.join(args.output_dir, cond['name'], f"seed_{args.seed}")
    train_from_agents(seeker, hider, args.timesteps, args.num_envs,
                      out_dir, cond['name'])


if __name__ == "__main__":
    main()
