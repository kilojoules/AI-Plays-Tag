#!/usr/bin/env python3
"""
Simple self-play PPO trainer for tag (A=0 baseline).

Both agents train simultaneously from the same environment steps.
No zoo sampling, no phase switching - just clean on-policy self-play.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from tag_env import VecTagEnv, TagEnvConfig
from ppo import PPOConfig, PPOAgent


@dataclass
class SelfPlayConfig:
    num_envs: int = 64
    total_timesteps: int = 100_000
    batch_size: int = 2048
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    lr: float = 3e-4
    train_iters: int = 10
    target_kl: float = 0.02
    log_interval: int = 10
    save_interval: int = 100
    output_dir: str = "experiments/results/selfplay"


class RolloutBuffer:
    """Stores rollout data for a single role."""

    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.obs: list = []
        self.actions: list = []
        self.rewards: list = []
        self.dones: list = []
        self.log_probs: list = []
        self.values: list = []

    @property
    def size(self) -> int:
        return len(self.obs) * self.num_envs

    def add(self, obs, actions, rewards, dones, log_probs, values):
        self.obs.append(obs.copy())
        self.actions.append(actions.copy())
        self.rewards.append(rewards.copy())
        self.dones.append(dones.copy())
        self.log_probs.append(log_probs.copy())
        self.values.append(values.copy())

    def compute_gae(self, last_values: np.ndarray,
                    gamma: float, lam: float) -> Tuple[np.ndarray, ...]:
        obs = np.stack(self.obs)
        actions = np.stack(self.actions)
        rewards = np.stack(self.rewards)
        dones = np.stack(self.dones)
        log_probs = np.stack(self.log_probs)
        values = np.stack(self.values)

        T, N = rewards.shape
        advantages = np.zeros((T, N), dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(T)):
            if t == T - 1:
                next_val = last_values
                next_done = np.zeros(N)
            else:
                next_val = values[t + 1]
                next_done = dones[t + 1]

            mask = 1.0 - next_done.astype(np.float32)
            delta = rewards[t] + gamma * next_val * mask - values[t]
            advantages[t] = last_gae = delta + gamma * lam * mask * last_gae

        returns = advantages + values

        return (
            obs.reshape(-1, obs.shape[-1]),
            actions.reshape(-1, actions.shape[-1]),
            log_probs.reshape(-1),
            returns.reshape(-1),
            advantages.reshape(-1),
        )


class SelfPlayTrainer:
    """Clean self-play PPO trainer. Both agents train from the same env steps."""

    def __init__(self, config: SelfPlayConfig, env_config: Optional[TagEnvConfig] = None):
        self.cfg = config
        self.env = VecTagEnv(num_envs=config.num_envs, config=env_config)

        ppo_cfg = PPOConfig(
            obs_dim=self.env.obs_dim,
            act_dim=self.env.act_dim,
            gamma=config.gamma,
            lam=config.gae_lambda,
            clip_ratio=config.clip_ratio,
            lr=config.lr,
            train_iters=config.train_iters,
            target_kl=config.target_kl,
        )

        self.policies = {
            'seeker': PPOAgent(ppo_cfg),
            'hider': PPOAgent(ppo_cfg),
        }

        # Persistent episode tracking (survives across rollout calls)
        self._ep_rewards = {
            'seeker': np.zeros(config.num_envs, dtype=np.float32),
            'hider': np.zeros(config.num_envs, dtype=np.float32),
        }
        self._ep_lengths = np.zeros(config.num_envs, dtype=np.int32)

        # Windowed metrics (reset each log interval)
        self._window_ep_rewards = {'seeker': [], 'hider': []}
        self._window_ep_lengths = []
        self._window_seeker_wins = 0
        self._window_hider_wins = 0
        self._window_episodes = 0
        self._total_episodes = 0

        # Output
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(config.output_dir, self.run_id)
        os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)
        self.metrics_path = os.path.join(self.output_dir, "metrics.csv")
        self._init_csv()

    def _init_csv(self):
        with open(self.metrics_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "update", "timesteps", "episodes",
                "seeker_reward_mean", "seeker_reward_std",
                "hider_reward_mean", "hider_reward_std",
                "episode_length_mean", "seeker_win_rate", "hider_win_rate",
                "seeker_policy_loss", "seeker_value_loss",
                "hider_policy_loss", "hider_value_loss",
                "fps",
            ])

    def _act_batch(self, policy: PPOAgent, obs_batch: np.ndarray):
        """Batched forward pass - all envs at once."""
        with torch.no_grad():
            x = torch.as_tensor(obs_batch, dtype=torch.float32)
            logits = policy.pi(x)
            mean, log_std = torch.chunk(logits, 2, dim=-1)
            log_std = torch.clamp(log_std, -2.0, 1.5)
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mean, std)
            raw_action = dist.sample()
            action = torch.tanh(raw_action)
            # Log prob with tanh correction
            log_prob = dist.log_prob(raw_action) - torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1)
            value = policy.vf(x).squeeze(-1)
        return action.numpy(), log_prob.numpy(), value.numpy()

    def _bootstrap_values(self, obs):
        """Get value estimates for final observations."""
        vals = {}
        with torch.no_grad():
            for role in ['seeker', 'hider']:
                x = torch.as_tensor(obs[role], dtype=torch.float32)
                vals[role] = self.policies[role].vf(x).squeeze(-1).numpy()
        return vals

    def collect_rollout(self):
        """Collect on-policy data for both agents simultaneously."""
        obs = self.env.auto_reset()
        bufs = {
            'seeker': RolloutBuffer(self.cfg.num_envs),
            'hider': RolloutBuffer(self.cfg.num_envs),
        }

        while bufs['seeker'].size < self.cfg.batch_size:
            # Both agents act with their learning policies
            acts, lps, vals = {}, {}, {}
            for role in ['seeker', 'hider']:
                acts[role], lps[role], vals[role] = self._act_batch(
                    self.policies[role], obs[role])

            next_obs, rewards, dones, infos = self.env.step(acts)

            for role in ['seeker', 'hider']:
                bufs[role].add(obs[role], acts[role], rewards[role],
                               dones, lps[role], vals[role])

            # Persistent episode tracking
            for role in ['seeker', 'hider']:
                self._ep_rewards[role] += rewards[role]
            self._ep_lengths += 1

            for eid in np.where(dones)[0]:
                for role in ['seeker', 'hider']:
                    self._window_ep_rewards[role].append(
                        self._ep_rewards[role][eid])
                    self._ep_rewards[role][eid] = 0.0
                self._window_ep_lengths.append(self._ep_lengths[eid])
                self._ep_lengths[eid] = 0
                self._window_episodes += 1
                self._total_episodes += 1
                if infos['tagged'][eid]:
                    self._window_seeker_wins += 1
                else:
                    self._window_hider_wins += 1

            obs = self.env.auto_reset()

        last_vals = self._bootstrap_values(obs)
        return bufs, last_vals

    def train_step(self, role: str, buf: RolloutBuffer,
                   last_values: np.ndarray) -> Dict[str, float]:
        obs, actions, log_probs_old, returns, advantages = buf.compute_gae(
            last_values, self.cfg.gamma, self.cfg.gae_lambda)
        return self.policies[role].update(
            obs, actions, log_probs_old, returns, advantages)

    def _log_metrics(self, update, timesteps, train_info, start_time):
        elapsed = time.time() - start_time
        fps = timesteps / elapsed if elapsed > 0 else 0

        s_rews = self._window_ep_rewards['seeker'] or [0]
        h_rews = self._window_ep_rewards['hider'] or [0]
        ep_lens = self._window_ep_lengths or [0]
        total = self._window_seeker_wins + self._window_hider_wins
        s_wr = self._window_seeker_wins / max(total, 1)
        h_wr = self._window_hider_wins / max(total, 1)

        row = [
            update, timesteps, self._total_episodes,
            np.mean(s_rews), np.std(s_rews),
            np.mean(h_rews), np.std(h_rews),
            np.mean(ep_lens), s_wr, h_wr,
            train_info.get('seeker', {}).get('policy_loss', 0),
            train_info.get('seeker', {}).get('value_loss', 0),
            train_info.get('hider', {}).get('policy_loss', 0),
            train_info.get('hider', {}).get('value_loss', 0),
            fps,
        ]

        with open(self.metrics_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

        print(f"Update {update:4d} | Steps: {timesteps:8d} | "
              f"Eps: {self._total_episodes:5d} | "
              f"Seeker WR: {s_wr:.0%} | "
              f"EpLen: {np.mean(ep_lens):.0f} | "
              f"S_R: {np.mean(s_rews):+.2f} | H_R: {np.mean(h_rews):+.2f} | "
              f"FPS: {fps:.0f}")

        # Reset window
        self._window_ep_rewards = {'seeker': [], 'hider': []}
        self._window_ep_lengths = []
        self._window_seeker_wins = 0
        self._window_hider_wins = 0
        self._window_episodes = 0

    def save_checkpoint(self, update):
        for role in ['seeker', 'hider']:
            path = os.path.join(self.output_dir, "checkpoints",
                                f"{role}_{update:05d}.pt")
            self.policies[role].save_policy(path)

    def save_final(self):
        for role in ['seeker', 'hider']:
            path = os.path.join(self.output_dir, f"policy_{role}_final.pt")
            self.policies[role].save_policy(path)
        print(f"\nFinal policies saved to {self.output_dir}")

    def train(self):
        print(f"Self-play training: {self.cfg.total_timesteps} timesteps")
        print(f"  num_envs={self.cfg.num_envs}, batch_size={self.cfg.batch_size}, "
              f"lr={self.cfg.lr}")
        print(f"  Output: {self.output_dir}\n")

        timesteps = 0
        update = 0
        start_time = time.time()

        while timesteps < self.cfg.total_timesteps:
            bufs, last_vals = self.collect_rollout()
            timesteps += bufs['seeker'].size

            train_info = {}
            for role in ['seeker', 'hider']:
                train_info[role] = self.train_step(
                    role, bufs[role], last_vals[role])
            update += 1

            if update % self.cfg.log_interval == 0:
                self._log_metrics(update, timesteps, train_info, start_time)

            if update % self.cfg.save_interval == 0:
                self.save_checkpoint(update)

        self._log_metrics(update, timesteps, train_info, start_time)
        self.save_final()

        elapsed = time.time() - start_time
        print(f"\nDone! {timesteps} steps in {elapsed:.1f}s "
              f"({timesteps/elapsed:.0f} FPS)")
        print(f"Total episodes: {self._total_episodes}")


def main():
    parser = argparse.ArgumentParser(description="Self-play PPO for tag")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--layout", type=str, default="four_corners",
                        choices=["empty", "four_corners", "central_cross", "playground"])
    parser.add_argument("--enable-sprint", action="store_true",
                        help="Enable stamina/sprint system")
    parser.add_argument("--hider-speed-mult", type=float, default=1.0,
                        help="Hider base speed multiplier (e.g. 1.1 for 10%% advantage)")
    parser.add_argument("--sprint-speed-mult", type=float, default=1.5,
                        help="Max speed multiplier when sprinting")
    parser.add_argument("--seeker-time-penalty", type=float, default=-0.005,
                        help="Per-step time penalty for seeker (default: -0.005)")
    parser.add_argument("--hider-dist-reward", type=float, default=0.0,
                        help="Hider reward scale for increasing distance (default: 0.0)")
    parser.add_argument("--hider-abs-dist-reward", type=float, default=0.1,
                        help="Hider reward scale for absolute distance (default: 0.1)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--output-dir", type=str,
                        default="experiments/results/selfplay")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    config = SelfPlayConfig(
        num_envs=args.num_envs,
        total_timesteps=args.timesteps,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output_dir,
    )
    env_config = TagEnvConfig(
        layout=args.layout,
        enable_sprint=args.enable_sprint,
        hider_speed_mult=args.hider_speed_mult,
        sprint_speed_mult=args.sprint_speed_mult,
        seeker_time_penalty=args.seeker_time_penalty,
        hider_dist_reward_scale=args.hider_dist_reward,
        hider_abs_dist_reward_scale=args.hider_abs_dist_reward,
    )

    trainer = SelfPlayTrainer(config, env_config=env_config)
    trainer.train()


if __name__ == "__main__":
    main()
