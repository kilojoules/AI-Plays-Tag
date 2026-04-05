#!/usr/bin/env python3
"""
Deep dive into the actor feature conditioning mechanism.

Exp 1: 5-seed transplant replication (task_ids 0-19)
  4 conditions x 5 seeds = 20 training runs at 1M steps
  Conditions: control_607, control_005, transplant_607critic, transplant_005critic

Exp 2: Effective rank analysis (task_id 20)
  Load 500K checkpoints for all init_alpha, compute actor hidden rep rank.
  No training — analysis only.

Exp 3: Random critic transplant (task_ids 21-25)
  0.607's actor + randomly initialized critic, 5 seeds at 1M steps

Exp 4: Layer-wise transplant (task_ids 26-34)
  3 conditions x 3 seeds = 9 runs at 1M steps
  - trunk_only: 0.607's trunk + 0.05's heads
  - heads_only: 0.607's heads + 0.05's trunk
  - layer0_only: 0.607's first layer + 0.05's rest

Exp 5: Freeze actor (task_ids 35-37)
  0.607's actor frozen + fresh critic, 3 seeds at 1M steps

Total: 38 tasks
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "trainer"))

from tag_env import VecTagEnv, TagEnvConfig
from sac import SACConfig, SACAgent, SACActorNet, SACCriticNet, ReplayBuffer


BASE = Path("experiments/results/paper_final")
IA_BASE = BASE / "init_alpha"
OUTPUT = BASE / "mechanism_deep_dive"

OBS_DIM = 87
ACT_DIM = 3


def get_checkpoint_path(init_alpha, role, seed=0):
    name = f"initalpha_{init_alpha}"
    runs = sorted((IA_BASE / name / f"seed_{seed}").glob("2026*"))
    if not runs:
        raise FileNotFoundError(f"No run for {name}/seed_{seed}")
    return str(runs[-1] / f"policy_{role}_final.pt")


def load_agent(path):
    cfg = SACConfig(obs_dim=OBS_DIM, act_dim=ACT_DIM)
    agent = SACAgent(cfg)
    agent.load_policy(path)
    return agent


def create_mixed_agent(actor_source_path, critic_source_path,
                       layer_mode="full", freeze_actor=False):
    """Create agent with mixed actor/critic from different sources.

    layer_mode:
      "full" — full actor from actor_source
      "trunk_only" — trunk from actor_source, heads from critic_source
      "heads_only" — heads from actor_source, trunk from critic_source
      "layer0_only" — first trunk layer from actor_source, rest from critic_source
      "random_critic" — actor from actor_source, critic randomly initialized
    """
    cfg = SACConfig(obs_dim=OBS_DIM, act_dim=ACT_DIM)
    agent = SACAgent(cfg)

    actor_state = torch.load(actor_source_path, map_location="cpu")

    if layer_mode == "random_critic":
        # Load full actor, keep random critic
        agent.actor.load_state_dict(actor_state['actor'])
        if 'log_alpha' in actor_state:
            agent.log_alpha.data.copy_(actor_state['log_alpha'])
    else:
        critic_state = torch.load(critic_source_path, map_location="cpu")

        if layer_mode == "full":
            agent.actor.load_state_dict(actor_state['actor'])
            agent.critic.load_state_dict(critic_state['critic'])
            agent.critic_target.load_state_dict(critic_state['critic_target'])
        elif layer_mode == "trunk_only":
            # Actor trunk from actor_source, heads from critic_source's actor
            agent.actor.load_state_dict(critic_state['actor'])  # start with critic_source actor
            agent.actor.trunk.load_state_dict(
                {k: v for k, v in actor_state['actor'].items() if k.startswith('trunk.')})
            agent.critic.load_state_dict(critic_state['critic'])
            agent.critic_target.load_state_dict(critic_state['critic_target'])
        elif layer_mode == "heads_only":
            agent.actor.load_state_dict(critic_state['actor'])  # start with critic_source actor
            for key in ['mean_head.weight', 'mean_head.bias',
                        'log_std_head.weight', 'log_std_head.bias']:
                agent.actor.state_dict()[key].copy_(actor_state['actor'][key])
            agent.critic.load_state_dict(critic_state['critic'])
            agent.critic_target.load_state_dict(critic_state['critic_target'])
        elif layer_mode == "layer0_only":
            agent.actor.load_state_dict(critic_state['actor'])  # start with critic_source actor
            for key in ['trunk.0.weight', 'trunk.0.bias']:
                agent.actor.state_dict()[key].copy_(actor_state['actor'][key])
            agent.critic.load_state_dict(critic_state['critic'])
            agent.critic_target.load_state_dict(critic_state['critic_target'])

        if 'log_alpha' in critic_state:
            agent.log_alpha.data.copy_(critic_state['log_alpha'])

    if freeze_actor:
        for p in agent.actor.parameters():
            p.requires_grad = False

    return agent


def train_agents(seeker, hider, total_timesteps, num_envs, output_dir, name,
                 freeze_actor=False):
    """Train from pre-initialized agents."""
    env_config = TagEnvConfig(
        layout="four_corners", hider_speed_mult=1.15,
        seeker_time_penalty=0.0, distance_reward_scale=0.0,
        hider_dist_reward_scale=0.0, hider_abs_dist_reward_scale=0.0,
        runner_survival_bonus=0.0,
    )
    env = VecTagEnv(num_envs=num_envs, config=env_config)

    buffers = {
        'seeker': ReplayBuffer(OBS_DIM, ACT_DIM, 100000),
        'hider': ReplayBuffer(OBS_DIM, ACT_DIM, 100000),
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

    print(f"  Training {name}: {total_timesteps} steps, freeze_actor={freeze_actor}")

    obs = env.reset()
    timesteps = 0
    start_time = time.time()
    total_episodes = 0
    window_s_rew, window_h_rew, window_lens = [], [], []
    window_s_wins, window_h_wins = 0, 0
    ep_rewards = {'seeker': np.zeros(num_envs), 'hider': np.zeros(num_envs)}
    ep_lengths = np.zeros(num_envs, dtype=np.int32)
    train_infos = {}
    warmup = 5000

    while timesteps < total_timesteps:
        is_warmup = timesteps < warmup
        acts = {}
        for role in ['seeker', 'hider']:
            if is_warmup:
                acts[role] = np.random.uniform(-1, 1, (num_envs, ACT_DIM)).astype(np.float32)
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
            window_s_rew.append(ep_rewards['seeker'][eid])
            window_h_rew.append(ep_rewards['hider'][eid])
            ep_rewards['seeker'][eid] = 0
            ep_rewards['hider'][eid] = 0
            window_lens.append(ep_lengths[eid])
            ep_lengths[eid] = 0
            total_episodes += 1
            if infos['tagged'][eid]:
                window_s_wins += 1
            else:
                window_h_wins += 1

        obs = env.auto_reset()

        if not is_warmup:
            for _ in range(4):
                for role in ['seeker', 'hider']:
                    if buffers[role].size >= 256:
                        batch = buffers[role].sample(256)
                        if freeze_actor and role in ['seeker', 'hider']:
                            # Update critic only
                            train_infos[role] = agents[role].update(batch)
                        else:
                            train_infos[role] = agents[role].update(batch)

        if timesteps % 5000 < num_envs:
            elapsed = time.time() - start_time
            fps = timesteps / elapsed if elapsed > 0 else 0
            total = window_s_wins + window_h_wins
            swr = window_s_wins / max(total, 1)
            s_info = train_infos.get('seeker', {})
            h_info = train_infos.get('hider', {})

            row = [timesteps, total_episodes,
                   np.mean(window_s_rew) if window_s_rew else 0,
                   np.mean(window_h_rew) if window_h_rew else 0,
                   np.mean(window_lens) if window_lens else 0, swr,
                   s_info.get('alpha', 0), h_info.get('alpha', 0), fps]
            with open(metrics_path, "a", newline="") as f:
                csv.writer(f).writerow(row)

            window_s_rew, window_h_rew, window_lens = [], [], []
            window_s_wins, window_h_wins = 0, 0

    for role in ['seeker', 'hider']:
        agents[role].save_policy(os.path.join(out, f"policy_{role}_final.pt"))

    elapsed = time.time() - start_time
    print(f"  Done: {timesteps} steps in {elapsed:.1f}s ({timesteps/elapsed:.0f} FPS)")


def run_effective_rank():
    """Exp 2: Compute effective rank of actor hidden representations."""
    print("=" * 60)
    print("Experiment 2: Effective Rank of Actor Representations")
    print("=" * 60)

    env_config = TagEnvConfig(layout="four_corners", hider_speed_mult=1.15)
    env = VecTagEnv(num_envs=200, config=env_config)
    obs = env.reset()

    # Collect diverse states by running random actions
    all_states = [obs['hider'].copy()]
    for _ in range(50):
        acts = {'seeker': np.random.uniform(-1, 1, (200, ACT_DIM)).astype(np.float32),
                'hider': np.random.uniform(-1, 1, (200, ACT_DIM)).astype(np.float32)}
        obs, _, dones, _ = env.step(acts)
        all_states.append(obs['hider'].copy())
        obs = env.auto_reset()
    states = torch.as_tensor(np.concatenate(all_states, axis=0), dtype=torch.float32)
    print(f"  Collected {states.shape[0]} diverse states")

    init_alphas = [0.05, 0.2, 0.607, 2.0]
    results = {}

    for ia in init_alphas:
        # Load 500K checkpoint
        name = f"initalpha_{ia}"
        runs = sorted((IA_BASE / name / "seed_0").glob("2026*"))
        if not runs:
            print(f"  SKIP {ia}: no run")
            continue

        ckpt_path = runs[-1] / "checkpoints" / "hider_00500032.pt"
        final_path = runs[-1] / "policy_hider_final.pt"

        for label, path in [("500K", ckpt_path), ("final", final_path)]:
            if not path.exists():
                continue

            agent = load_agent(str(path))
            with torch.no_grad():
                # Get hidden activations from trunk
                h1 = agent.actor.trunk[0](states)  # After first linear
                h1_act = torch.relu(h1)  # After ReLU
                h2 = agent.actor.trunk[2](h1_act)  # After second linear
                h2_act = torch.relu(h2)  # After second ReLU

            for layer_name, activations in [("layer0", h1_act), ("layer1", h2_act)]:
                # Compute effective rank via singular values
                # Effective rank = exp(entropy of normalized singular values)
                U, S, V = torch.svd(activations)
                S_norm = S / S.sum()
                S_norm = S_norm[S_norm > 1e-10]
                entropy = -(S_norm * torch.log(S_norm)).sum().item()
                eff_rank = np.exp(entropy)

                # Also compute fraction of variance in top-k singular values
                var_explained = (S ** 2).cumsum(0) / (S ** 2).sum()
                rank_90 = int((var_explained < 0.9).sum()) + 1
                rank_99 = int((var_explained < 0.99).sum()) + 1

                key = f"{ia}_{label}_{layer_name}"
                results[key] = {
                    'init_alpha': ia, 'checkpoint': label, 'layer': layer_name,
                    'effective_rank': eff_rank,
                    'rank_90pct': rank_90, 'rank_99pct': rank_99,
                    'max_singular': float(S[0]),
                    'num_nonzero': int((S > 1e-6).sum()),
                }
                print(f"  ia={ia:<6} {label:<6} {layer_name}: eff_rank={eff_rank:.1f} "
                      f"rank90={rank_90} rank99={rank_99}")

    out_path = OUTPUT / "effective_rank.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved {out_path}")


def build_task_table():
    tasks = []

    # Exp 1: 5-seed transplant replication (20 tasks)
    conditions_1 = [
        {"name": "control_607", "actor_ia": 0.607, "critic_ia": 0.607, "mode": "full"},
        {"name": "control_005", "actor_ia": 0.05, "critic_ia": 0.05, "mode": "full"},
        {"name": "transplant_607critic", "actor_ia": 0.05, "critic_ia": 0.607, "mode": "full"},
        {"name": "transplant_005critic", "actor_ia": 0.607, "critic_ia": 0.05, "mode": "full"},
    ]
    for cond in conditions_1:
        for seed in range(5):
            tasks.append({**cond, "experiment": "1_transplant_5seed",
                         "seed": seed, "freeze": False})

    # Exp 2: effective rank (1 task, no training)
    tasks.append({"experiment": "2_effective_rank", "name": "rank_analysis",
                  "seed": 0, "actor_ia": 0, "critic_ia": 0, "mode": "analysis",
                  "freeze": False})

    # Exp 3: random critic (5 tasks)
    for seed in range(5):
        tasks.append({"experiment": "3_random_critic", "name": "random_critic",
                     "actor_ia": 0.607, "critic_ia": None, "mode": "random_critic",
                     "seed": seed, "freeze": False})

    # Exp 4: layer-wise transplant (9 tasks)
    layer_conditions = [
        {"name": "trunk_only", "mode": "trunk_only"},
        {"name": "heads_only", "mode": "heads_only"},
        {"name": "layer0_only", "mode": "layer0_only"},
    ]
    for cond in layer_conditions:
        for seed in range(3):
            tasks.append({**cond, "experiment": "4_layerwise",
                         "actor_ia": 0.607, "critic_ia": 0.05,
                         "seed": seed, "freeze": False})

    # Exp 5: freeze actor (3 tasks)
    for seed in range(3):
        tasks.append({"experiment": "5_freeze_actor", "name": "freeze_607",
                     "actor_ia": 0.607, "critic_ia": None, "mode": "random_critic",
                     "seed": seed, "freeze": True})

    return tasks


def run_task(task_id):
    tasks = build_task_table()
    if task_id < 0 or task_id >= len(tasks):
        print(f"Invalid task_id {task_id}. Valid: 0-{len(tasks)-1}")
        sys.exit(1)

    task = tasks[task_id]
    exp = task["experiment"]
    name = task["name"]
    seed = task["seed"]

    print(f"Task {task_id}: {exp}/{name} seed={seed}")

    if exp == "2_effective_rank":
        run_effective_rank()
        return

    torch.manual_seed(seed)
    np.random.seed(seed)

    out_dir = os.path.join(str(OUTPUT), exp, name, f"seed_{seed}")
    os.makedirs(out_dir, exist_ok=True)

    actor_ia = task["actor_ia"]
    critic_ia = task["critic_ia"]
    mode = task["mode"]
    freeze = task["freeze"]

    # Build agents
    for role in ['seeker', 'hider']:
        actor_path = get_checkpoint_path(actor_ia, role)

        if mode == "random_critic":
            agent = create_mixed_agent(actor_path, None,
                                       layer_mode="random_critic",
                                       freeze_actor=freeze)
        else:
            critic_path = get_checkpoint_path(critic_ia, role)
            agent = create_mixed_agent(actor_path, critic_path,
                                       layer_mode=mode, freeze_actor=freeze)

        if role == 'seeker':
            seeker = agent
        else:
            hider = agent

    train_agents(seeker, hider, total_timesteps=1_000_000, num_envs=64,
                 output_dir=out_dir, name=f"{exp}/{name}/seed_{seed}",
                 freeze_actor=freeze)


def main():
    parser = argparse.ArgumentParser(description="Mechanism deep dive experiments")
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    tasks = build_task_table()

    if args.list:
        for i, t in enumerate(tasks):
            print(f"{i:3d}: {t['experiment']:<22s} {t['name']:<20s} "
                  f"mode={t['mode']:<14s} seed={t['seed']}")
        print(f"\nTotal: {len(tasks)} tasks")
        return

    if args.task_id is not None:
        run_task(args.task_id)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
