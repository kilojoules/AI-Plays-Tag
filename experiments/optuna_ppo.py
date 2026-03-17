#!/usr/bin/env python3
"""
Optuna hyperparameter optimization for PPO self-play training.

Usage:
  pixi run python experiments/optuna_ppo.py --n-trials 10
  # SLURM workers share the same SQLite study
  pixi run python experiments/optuna_ppo.py --n-trials 5 --study-name ppo_hpo

Each trial trains PPO self-play for 1M steps and returns a balance score.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import types
from typing import Tuple

import numpy as np
import optuna
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'trainer'))

from tag_env import VecTagEnv, TagEnvConfig
from ppo import PPOConfig, PPOAgent


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


def patch_entropy_coef(agent: PPOAgent, entropy_coef: float):
    """Monkey-patch PPOAgent.update to use a custom entropy coefficient."""
    agent._ent_coef = entropy_coef

    def custom_update(self, obs, actions, logp_old, returns, advantages):
        device = torch.device("cpu")
        o = torch.as_tensor(obs, dtype=torch.float32, device=device)
        a = torch.as_tensor(actions, dtype=torch.float32, device=device)
        logp_old_t = torch.as_tensor(logp_old, dtype=torch.float32, device=device)
        ret = torch.as_tensor(returns, dtype=torch.float32, device=device)
        adv = torch.as_tensor(advantages, dtype=torch.float32, device=device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        info = {}
        for _ in range(self.cfg.train_iters):
            logp, entropy, value = self.evaluate_actions(o, a)
            log_ratio = torch.clamp(logp - logp_old_t, -20.0, 20.0)
            ratio = torch.exp(log_ratio)
            clip_adv = torch.clamp(ratio, 1.0 - self.cfg.clip_ratio,
                                   1.0 + self.cfg.clip_ratio) * adv
            loss_pi = -(torch.min(ratio * adv, clip_adv)).mean() \
                - self._ent_coef * entropy.mean()
            loss_v = 0.5 * ((ret - value) ** 2).mean()
            approx_kl = (logp_old_t - logp).mean().item()
            if not (torch.isfinite(loss_pi) and torch.isfinite(loss_v)):
                break
            self.pi_opt.zero_grad(); loss_pi.backward()
            torch.nn.utils.clip_grad_norm_(self.pi.parameters(), max_norm=0.5)
            self.pi_opt.step()
            self.vf_opt.zero_grad(); loss_v.backward()
            torch.nn.utils.clip_grad_norm_(self.vf.parameters(), max_norm=0.5)
            self.vf_opt.step()
            info = {"policy_loss": float(loss_pi.item()), "value_loss": float(loss_v.item()),
                    "entropy": float(entropy.mean().item()), "approx_kl": approx_kl}
            if approx_kl > 1.5 * self.cfg.target_kl:
                break
        return info

    agent.update = types.MethodType(custom_update, agent)


def batch_act(agent, obs_batch):
    """Batched action sampling from PPOAgent (avoids per-env loop)."""
    with torch.no_grad():
        x = torch.as_tensor(obs_batch, dtype=torch.float32)
        logits = agent.pi(x)
        mean, log_std_raw = torch.chunk(logits, 2, dim=-1)
        log_std = torch.clamp(log_std_raw, -2.0, 1.5)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        raw_actions = dist.sample()
        actions = torch.tanh(raw_actions)
        pre_tanh = torch.atanh(torch.clamp(actions, -0.999, 0.999))
        log_prob = (dist.log_prob(pre_tanh) - torch.log(1 - actions.pow(2) + 1e-6)).sum(-1)
        values = agent.vf(x).squeeze(-1)
        return actions.cpu().numpy(), log_prob.cpu().numpy(), values.cpu().numpy()


def collect_rollout(env, obs, agent, opponent, role, opp_role, batch_size, num_envs):
    """Collect a rollout for one role using batched action sampling.

    Takes current obs state (env persists across rollouts so timeouts can complete).
    Returns (obs_flat, act_flat, logp_flat, ret_flat, adv_flat, steps, wins, new_obs).
    """
    obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []
    win_counts = {'seeker': 0, 'hider': 0}

    collected = 0

    while collected < batch_size:
        actions, logps, values = batch_act(agent, obs[role])
        opp_actions, _, _ = batch_act(opponent, obs[opp_role])

        acts = {role: actions, opp_role: opp_actions}
        next_obs, rewards, dones, infos = env.step(acts)

        obs_buf.append(obs[role].copy())
        act_buf.append(actions)
        logp_buf.append(logps)
        rew_buf.append(rewards[role])
        val_buf.append(values)
        done_buf.append(dones.astype(np.float32))

        for eid in np.where(dones)[0]:
            if infos['tagged'][eid]:
                win_counts['seeker'] += 1
            else:
                win_counts['hider'] += 1

        obs = env.auto_reset()
        collected += num_envs

    # Compute last values for GAE (batched)
    with torch.no_grad():
        x = torch.as_tensor(obs[role], dtype=torch.float32)
        last_values = agent.vf(x).squeeze(-1).cpu().numpy()

    # Stack and compute GAE
    obs_arr = np.stack(obs_buf)       # [T, N, obs_dim]
    act_arr = np.stack(act_buf)       # [T, N, act_dim]
    rew_arr = np.stack(rew_buf)       # [T, N]
    done_arr = np.stack(done_buf)     # [T, N]
    logp_arr = np.stack(logp_buf)     # [T, N]
    val_arr = np.stack(val_buf)       # [T, N]

    T, N = rew_arr.shape
    advantages = np.zeros((T, N), dtype=np.float32)
    last_gae = 0.0
    gamma = agent.cfg.gamma
    lam = agent.cfg.lam
    for t in reversed(range(T)):
        if t == T - 1:
            next_val = last_values
            next_done = np.zeros(N)
        else:
            next_val = val_arr[t + 1]
            next_done = done_arr[t + 1]
        mask = 1.0 - next_done
        delta = rew_arr[t] + gamma * next_val * mask - val_arr[t]
        advantages[t] = last_gae = delta + gamma * lam * mask * last_gae
    returns = advantages + val_arr

    flat = lambda x, d: x.reshape(-1, d) if d > 1 else x.reshape(-1)
    return (flat(obs_arr, obs_arr.shape[-1]), flat(act_arr, act_arr.shape[-1]),
            flat(logp_arr, 1), flat(returns, 1), flat(advantages, 1),
            collected, win_counts, obs)


def ppo_objective(trial: optuna.Trial) -> float:
    """Train PPO self-play and return balance score."""
    lr = trial.suggest_float("lr", 3e-5, 3e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.95, 0.999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.99)
    clip_ratio = trial.suggest_float("clip_ratio", 0.1, 0.3)
    train_iters = trial.suggest_int("train_iters", 3, 20)
    batch_size = trial.suggest_categorical("batch_size", [1024, 2048, 4096])
    entropy_coef = trial.suggest_float("entropy_coef", 0.001, 0.1, log=True)
    target_kl = trial.suggest_float("target_kl", 0.005, 0.05)

    seed = trial.number % 3
    torch.manual_seed(seed); np.random.seed(seed)

    env = VecTagEnv(num_envs=NUM_ENVS, config=ENV_CONFIG)
    ppo_cfg = PPOConfig(obs_dim=env.obs_dim, act_dim=env.act_dim, gamma=gamma,
                        lam=gae_lambda, clip_ratio=clip_ratio, lr=lr,
                        train_iters=train_iters, target_kl=target_kl)

    agents = {r: PPOAgent(ppo_cfg) for r in ['seeker', 'hider']}
    for r in agents:
        patch_entropy_coef(agents[r], entropy_coef)

    timesteps = 0
    window_swins = []
    obs = env.reset()  # Initial reset; env state persists across rollouts

    while timesteps < TOTAL_TIMESTEPS:
        total_sw = 0
        total_hw = 0

        for learn_role in ['seeker', 'hider']:
            opp_role = 'hider' if learn_role == 'seeker' else 'seeker'
            obs_arr, act_arr, logp_arr, ret_arr, adv_arr, steps, wins, obs = \
                collect_rollout(env, obs, agents[learn_role], agents[opp_role],
                                learn_role, opp_role, batch_size, NUM_ENVS)
            agents[learn_role].update(obs_arr, act_arr, logp_arr, ret_arr, adv_arr)
            timesteps += steps
            total_sw += wins['seeker']
            total_hw += wins['hider']

        total = total_sw + total_hw
        if total > 0:
            swr = total_sw / total
            window_swins.append(swr)
            score = min(swr, 1 - swr)
            trial.report(score, len(window_swins))
            if trial.should_prune():
                raise optuna.TrialPruned()

    if len(window_swins) < 3:
        return 0.0
    mean_swr = np.mean(window_swins[-3:])
    score = min(mean_swr, 1 - mean_swr)
    print(f"  Trial {trial.number}: {len(window_swins)} windows, "
          f"last3_swr={window_swins[-3:]}, mean={mean_swr:.3f}, score={score:.3f}")
    return score


def main():
    parser = argparse.ArgumentParser(description="Optuna PPO HPO")
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--study-name", type=str, default="ppo_hpo_v1")
    parser.add_argument("--storage-dir", type=str, default="experiments/results/optuna")
    parser.add_argument("--timeout", type=int, default=None)
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
        study_name=args.study_name, storage=storage, direction="maximize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )

    print(f"Study: {args.study_name} | Journal: {journal_path} | Trials: {args.n_trials}")
    study.optimize(ppo_objective, n_trials=args.n_trials, timeout=args.timeout)
    print(f"\nBest trial #{study.best_trial.number}: score={study.best_trial.value:.4f}")
    print(f"  Params: {study.best_trial.params}")


if __name__ == "__main__":
    main()
