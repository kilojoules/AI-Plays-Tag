#!/usr/bin/env python3
"""
Optuna hyperparameter optimization for SAC self-play training.

Usage:
  # Single worker (interactive)
  pixi run python experiments/optuna_sac.py --n-trials 10

  # SLURM workers share the same study via SQLite
  pixi run python experiments/optuna_sac.py --n-trials 5 --study-name sac_hpo

Each trial trains SAC self-play for 1M steps and returns a balance score.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import optuna
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'trainer'))

from tag_env import VecTagEnv, TagEnvConfig
from sac import SACConfig, SACAgent, ReplayBuffer


# Fixed environment config (R4_sparse with HSM=1.15)
# Pure sparse reward: only +1 tag / -1 tagged, no shaping
ENV_CONFIG = TagEnvConfig(
    layout="four_corners",
    hider_speed_mult=1.15,
    distance_reward_scale=0.0,
    hider_dist_reward_scale=0.0,
    hider_abs_dist_reward_scale=0.0,
    seeker_time_penalty=0.0,
    runner_survival_bonus=0.0,
)

TOTAL_TIMESTEPS = 1_000_000
NUM_ENVS = 64
LOG_WINDOW = 100_000  # steps per evaluation window


def sac_objective(trial: optuna.Trial) -> float:
    """Train SAC self-play and return balance score."""
    # Sample hyperparameters
    lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.999)
    tau = trial.suggest_float("tau", 0.001, 0.05, log=True)
    init_alpha = trial.suggest_float("init_alpha", 0.05, 1.0, log=True)
    buffer_size = trial.suggest_categorical("buffer_size",
                                            [25_000, 50_000, 100_000, 250_000, 500_000])
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
    updates_per_step = trial.suggest_int("updates_per_step", 1, 4)
    warmup_steps = trial.suggest_categorical("warmup_steps", [3_000, 5_000, 10_000])

    seed = trial.number % 3
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = VecTagEnv(num_envs=NUM_ENVS, config=ENV_CONFIG)
    obs_dim = env.obs_dim
    act_dim = env.act_dim

    sac_cfg = SACConfig(
        obs_dim=obs_dim, act_dim=act_dim,
        actor_lr=lr, critic_lr=lr, alpha_lr=lr,
        gamma=gamma, tau=tau, init_alpha=init_alpha,
        buffer_size=buffer_size, batch_size=batch_size,
        warmup_steps=warmup_steps,
    )

    agents = {r: SACAgent(sac_cfg) for r in ['seeker', 'hider']}
    buffers = {r: ReplayBuffer(obs_dim, act_dim, buffer_size) for r in ['seeker', 'hider']}

    obs = env.reset()
    timesteps = 0
    window_sw = 0
    window_hw = 0
    window_swins = []
    last_window_step = 0

    while timesteps < TOTAL_TIMESTEPS:
        is_warmup = timesteps < warmup_steps

        acts = {}
        for r in ['seeker', 'hider']:
            if is_warmup:
                acts[r] = np.random.uniform(-1, 1, (NUM_ENVS, act_dim)).astype(np.float32)
            else:
                with torch.no_grad():
                    x = torch.as_tensor(obs[r], dtype=torch.float32)
                    a, _ = agents[r].actor.sample(x)
                    acts[r] = a.cpu().numpy()

        next_obs, rewards, dones, infos = env.step(acts)
        timesteps += NUM_ENVS

        for r in ['seeker', 'hider']:
            buffers[r].add_batch(obs[r], acts[r], rewards[r],
                                 next_obs[r], dones.astype(np.float32))

        for eid in np.where(dones)[0]:
            if infos['tagged'][eid]:
                window_sw += 1
            else:
                window_hw += 1

        obs = env.auto_reset()

        if not is_warmup:
            for _ in range(updates_per_step):
                for r in ['seeker', 'hider']:
                    if buffers[r].size >= batch_size:
                        batch = buffers[r].sample(batch_size)
                        agents[r].update(batch)

        # Window logging
        if timesteps - last_window_step >= LOG_WINDOW:
            total = window_sw + window_hw
            if total > 0:
                swr = window_sw / total
                window_swins.append(swr)

                # Report for pruning
                score = min(swr, 1 - swr)
                trial.report(score, len(window_swins))
                if trial.should_prune():
                    raise optuna.TrialPruned()

            window_sw = 0
            window_hw = 0
            last_window_step = timesteps

    # Final score: average balance over last 3 windows
    if len(window_swins) < 3:
        return 0.0
    last_swrs = window_swins[-3:]
    mean_swr = np.mean(last_swrs)
    score = min(mean_swr, 1 - mean_swr)
    return score


def main():
    parser = argparse.ArgumentParser(description="Optuna SAC hyperparameter optimization")
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--study-name", type=str, default="sac_hpo_v1")
    parser.add_argument("--storage-dir", type=str,
                        default="experiments/results/optuna")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Max seconds for this worker")
    args = parser.parse_args()

    # Stagger SLURM array workers to avoid filesystem contention
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    time.sleep(task_id * 1.5)

    os.makedirs(args.storage_dir, exist_ok=True)
    journal_path = os.path.join(args.storage_dir, f"{args.study_name}.journal")
    storage = optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage(os.path.abspath(journal_path)),
    )

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )

    print(f"Study: {args.study_name} | Journal: {journal_path}")
    print(f"Running {args.n_trials} trials...")

    study.optimize(sac_objective, n_trials=args.n_trials, timeout=args.timeout)

    print(f"\nBest trial: {study.best_trial.number}")
    print(f"  Score: {study.best_trial.value:.4f}")
    print(f"  Params: {study.best_trial.params}")


if __name__ == "__main__":
    main()
