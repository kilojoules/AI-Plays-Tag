#!/usr/bin/env python3
"""
Zoo-based training for tag agents.

Implements opponent sampling from a zoo of past checkpoints:
- A% of the time: play against latest self-play opponent
- (100-A)% of the time: uniformly sample from checkpoint zoo

Supports:
- Hider zoo only (seeker trains against zoo of hiders)
- Both zoos (both roles train against their respective zoos)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
except ImportError:
    print("PyTorch required. Install with: pip install torch", file=sys.stderr)
    sys.exit(1)

from tag_env import VecTagEnv, TagEnvConfig
from ppo import PPOConfig, PPOAgent


@dataclass
class ZooTrainConfig:
    """Training configuration with zoo sampling."""
    # Environment
    num_envs: int = 64
    total_timesteps: int = 1_000_000

    # PPO hyperparameters
    batch_size: int = 2048
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    lr: float = 1e-4  # Lower for stability
    train_iters: int = 10
    target_kl: float = 0.02

    # Zoo parameters
    latest_opponent_prob: float = 0.1  # A = probability of latest opponent
    use_seeker_zoo: bool = False       # Whether to use seeker zoo too
    zoo_update_interval: int = 50      # Add to zoo every N updates
    zoo_max_size: int = 50             # Max checkpoints in zoo

    # Logging
    log_interval: int = 10
    save_interval: int = 50

    # Output
    output_dir: str = "experiments/results/zoo_training"


class OpponentZoo:
    """Manages a zoo of opponent checkpoints."""

    def __init__(self, obs_dim: int, act_dim: int, max_size: int = 50):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.max_size = max_size
        self.checkpoints: List[Dict[str, Any]] = []  # List of state dicts
        self.ppo_cfg = PPOConfig(obs_dim=obs_dim, act_dim=act_dim)

    def add(self, policy: PPOAgent, update: int):
        """Add a checkpoint to the zoo."""
        state = {
            'pi': {k: v.clone() for k, v in policy.pi.state_dict().items()},
            'vf': {k: v.clone() for k, v in policy.vf.state_dict().items()},
            'update': update,
        }
        self.checkpoints.append(state)

        # Remove oldest if over capacity
        if len(self.checkpoints) > self.max_size:
            self.checkpoints.pop(0)

    def sample(self) -> PPOAgent:
        """Sample a random opponent from the zoo."""
        if not self.checkpoints:
            raise ValueError("Zoo is empty!")

        state = random.choice(self.checkpoints)
        agent = PPOAgent(self.ppo_cfg)
        agent.pi.load_state_dict(state['pi'])
        agent.vf.load_state_dict(state['vf'])
        return agent

    def __len__(self):
        return len(self.checkpoints)


class RolloutBuffer:
    """Buffer for storing rollout data."""

    def __init__(self, buffer_size: int, obs_dim: int, act_dim: int, num_envs: int):
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.obs = {'seeker': [], 'hider': []}
        self.actions = {'seeker': [], 'hider': []}
        self.rewards = {'seeker': [], 'hider': []}
        self.dones = {'seeker': [], 'hider': []}
        self.log_probs = {'seeker': [], 'hider': []}
        self.values = {'seeker': [], 'hider': []}

        self.ptr = 0

    def add(self, obs: Dict[str, np.ndarray], actions: Dict[str, np.ndarray],
            rewards: Dict[str, np.ndarray], dones: np.ndarray,
            log_probs: Dict[str, np.ndarray], values: Dict[str, np.ndarray]):
        for role in ['seeker', 'hider']:
            self.obs[role].append(obs[role].copy())
            self.actions[role].append(actions[role].copy())
            self.rewards[role].append(rewards[role].copy())
            self.dones[role].append(dones.copy())
            self.log_probs[role].append(log_probs[role].copy())
            self.values[role].append(values[role].copy())
        self.ptr += 1

    def get_size(self) -> int:
        return self.ptr * self.num_envs

    def clear(self):
        for role in ['seeker', 'hider']:
            self.obs[role].clear()
            self.actions[role].clear()
            self.rewards[role].clear()
            self.dones[role].clear()
            self.log_probs[role].clear()
            self.values[role].clear()
        self.ptr = 0

    def compute_returns_and_advantages(self, last_values: Dict[str, np.ndarray],
                                       gamma: float, gae_lambda: float) -> Dict[str, Tuple[np.ndarray, ...]]:
        result = {}
        for role in ['seeker', 'hider']:
            obs = np.stack(self.obs[role])
            actions = np.stack(self.actions[role])
            rewards = np.stack(self.rewards[role])
            dones = np.stack(self.dones[role])
            log_probs = np.stack(self.log_probs[role])
            values = np.stack(self.values[role])

            T, N = rewards.shape
            advantages = np.zeros((T, N), dtype=np.float32)
            last_gae = 0

            for t in reversed(range(T)):
                if t == T - 1:
                    next_value = last_values[role]
                    next_done = np.zeros(N)
                else:
                    next_value = values[t + 1]
                    next_done = dones[t + 1]

                mask = 1.0 - next_done.astype(np.float32)
                delta = rewards[t] + gamma * next_value * mask - values[t]
                advantages[t] = last_gae = delta + gamma * gae_lambda * mask * last_gae

            returns = advantages + values

            result[role] = (
                obs.reshape(-1, obs.shape[-1]),
                actions.reshape(-1, actions.shape[-1]),
                log_probs.reshape(-1),
                returns.reshape(-1),
                advantages.reshape(-1),
            )

        return result


class ZooTrainer:
    """PPO trainer with zoo-based opponent sampling."""

    def __init__(self, config: ZooTrainConfig, env_config: Optional[TagEnvConfig] = None):
        self.config = config
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

        # Learning policies (these get trained)
        self.policies = {
            'seeker': PPOAgent(ppo_cfg),
            'hider': PPOAgent(ppo_cfg),
        }

        # Opponent zoos
        self.hider_zoo = OpponentZoo(self.env.obs_dim, self.env.act_dim, config.zoo_max_size)
        self.seeker_zoo = OpponentZoo(self.env.obs_dim, self.env.act_dim, config.zoo_max_size) if config.use_seeker_zoo else None

        # Current opponents (may be from zoo or latest)
        self.current_opponents = {
            'seeker': self.policies['seeker'],  # Opponent for hider
            'hider': self.policies['hider'],    # Opponent for seeker
        }

        # Setup output
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(config.output_dir, self.run_id)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)

        self.metrics_path = os.path.join(self.output_dir, "metrics.csv")
        self._init_metrics_csv()

        # Stats
        self.ep_reward_buffer = {'seeker': [], 'hider': []}
        self.ep_len_buffer = []
        self.seeker_wins = 0
        self.hider_wins = 0
        self.total_episodes = 0

        # Zoo sampling stats
        self.zoo_samples = {'seeker': 0, 'hider': 0}
        self.latest_samples = {'seeker': 0, 'hider': 0}

    def _init_metrics_csv(self):
        columns = [
            "update", "timesteps", "episodes", "seeker_reward_mean", "seeker_reward_std",
            "hider_reward_mean", "hider_reward_std", "episode_length_mean",
            "seeker_win_rate", "hider_win_rate", "seeker_policy_loss", "seeker_value_loss",
            "hider_policy_loss", "hider_value_loss", "fps", "time_elapsed",
            "hider_zoo_size", "seeker_zoo_size", "zoo_sample_rate"
        ]
        with open(self.metrics_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(columns)

    def _sample_opponents(self):
        """Sample opponents for this rollout."""
        # Sample hider opponent (for seeker to play against)
        if len(self.hider_zoo) > 0 and random.random() > self.config.latest_opponent_prob:
            self.current_opponents['hider'] = self.hider_zoo.sample()
            self.zoo_samples['hider'] += 1
        else:
            self.current_opponents['hider'] = self.policies['hider']
            self.latest_samples['hider'] += 1

        # Sample seeker opponent (for hider to play against) if using seeker zoo
        if self.seeker_zoo is not None:
            if len(self.seeker_zoo) > 0 and random.random() > self.config.latest_opponent_prob:
                self.current_opponents['seeker'] = self.seeker_zoo.sample()
                self.zoo_samples['seeker'] += 1
            else:
                self.current_opponents['seeker'] = self.policies['seeker']
                self.latest_samples['seeker'] += 1
        else:
            self.current_opponents['seeker'] = self.policies['seeker']

    def _update_zoos(self, update: int):
        """Add current policies to zoos."""
        if update % self.config.zoo_update_interval == 0:
            self.hider_zoo.add(self.policies['hider'], update)
            if self.seeker_zoo is not None:
                self.seeker_zoo.add(self.policies['seeker'], update)

    def act(self, obs: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], ...]:
        """Get actions - learning policies for own role, opponents for other role."""
        actions = {}
        log_probs = {}
        values = {}

        for role in ['seeker', 'hider']:
            # Use learning policy for this role's decisions
            policy = self.policies[role]
            role_obs = obs[role]

            acts = np.zeros((self.config.num_envs, self.env.act_dim), dtype=np.float32)
            lps = np.zeros(self.config.num_envs, dtype=np.float32)
            vals = np.zeros(self.config.num_envs, dtype=np.float32)

            for i in range(self.config.num_envs):
                act, lp, val = policy.act(role_obs[i])
                acts[i] = act
                lps[i] = lp
                vals[i] = val

            actions[role] = acts
            log_probs[role] = lps
            values[role] = vals

        return actions, log_probs, values

    def act_with_opponents(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Get actions using current opponents (for environment stepping)."""
        actions = {}

        for role in ['seeker', 'hider']:
            # Seeker acts based on seeker obs, using seeker policy or opponent
            # But when collecting data, we want the LEARNING policy's perspective
            # The opponent is what the OTHER role sees

            # For stepping the env, each role uses its current opponent assignment
            if role == 'seeker':
                # Seeker uses learning seeker policy
                policy = self.policies['seeker']
            else:
                # Hider uses learning hider policy
                policy = self.policies['hider']

            role_obs = obs[role]
            acts = np.zeros((self.config.num_envs, self.env.act_dim), dtype=np.float32)

            for i in range(self.config.num_envs):
                act, _, _ = policy.act(role_obs[i])
                acts[i] = act

            actions[role] = acts

        return actions

    def get_values(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        values = {}
        for role in ['seeker', 'hider']:
            vals = np.zeros(self.config.num_envs, dtype=np.float32)
            for i in range(self.config.num_envs):
                vals[i] = self.policies[role].value(obs[role][i])
            values[role] = vals
        return values

    def collect_rollout(self, buffer: RolloutBuffer) -> Dict[str, Any]:
        """Collect rollout with opponent sampling."""
        # Sample new opponents at start of rollout
        self._sample_opponents()

        obs = self.env.auto_reset()
        ep_rewards = {'seeker': np.zeros(self.config.num_envs),
                      'hider': np.zeros(self.config.num_envs)}
        ep_lengths = np.zeros(self.config.num_envs, dtype=np.int32)

        while buffer.get_size() < self.config.batch_size:
            actions, log_probs, values = self.act(obs)
            next_obs, rewards, dones, infos = self.env.step(actions)

            buffer.add(obs, actions, rewards, dones, log_probs, values)

            for role in ['seeker', 'hider']:
                ep_rewards[role] += rewards[role]
            ep_lengths += 1

            done_ids = np.where(dones)[0]
            for eid in done_ids:
                self.total_episodes += 1
                self.ep_reward_buffer['seeker'].append(ep_rewards['seeker'][eid])
                self.ep_reward_buffer['hider'].append(ep_rewards['hider'][eid])
                self.ep_len_buffer.append(ep_lengths[eid])

                if infos['tagged'][eid]:
                    self.seeker_wins += 1
                else:
                    self.hider_wins += 1

                ep_rewards['seeker'][eid] = 0
                ep_rewards['hider'][eid] = 0
                ep_lengths[eid] = 0

            obs = self.env.auto_reset()

        last_values = self.get_values(obs)
        return {'last_values': last_values, 'last_obs': obs}

    def train_step(self, buffer: RolloutBuffer, last_values: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        data = buffer.compute_returns_and_advantages(
            last_values, self.config.gamma, self.config.gae_lambda
        )

        train_info = {}
        for role in ['seeker', 'hider']:
            obs, actions, log_probs_old, returns, advantages = data[role]
            adv_mean = np.mean(advantages)
            adv_std = np.std(advantages) + 1e-8
            advantages = (advantages - adv_mean) / adv_std

            info = self.policies[role].update(
                obs, actions, log_probs_old, returns, advantages
            )
            train_info[role] = info

        return train_info

    def _log_metrics(self, update: int, timesteps: int, train_info: Dict[str, Any],
                     fps: float, time_elapsed: float):
        seeker_rewards = self.ep_reward_buffer.get('seeker', [0])
        hider_rewards = self.ep_reward_buffer.get('hider', [0])
        ep_lens = self.ep_len_buffer if self.ep_len_buffer else [0]

        total_eps = self.seeker_wins + self.hider_wins
        seeker_wr = self.seeker_wins / max(total_eps, 1)
        hider_wr = self.hider_wins / max(total_eps, 1)

        total_samples = sum(self.zoo_samples.values()) + sum(self.latest_samples.values())
        zoo_rate = sum(self.zoo_samples.values()) / max(total_samples, 1)

        row = [
            update, timesteps, self.total_episodes,
            np.mean(seeker_rewards) if seeker_rewards else 0,
            np.std(seeker_rewards) if seeker_rewards else 0,
            np.mean(hider_rewards) if hider_rewards else 0,
            np.std(hider_rewards) if hider_rewards else 0,
            np.mean(ep_lens) if ep_lens else 0,
            seeker_wr, hider_wr,
            train_info.get('seeker', {}).get('policy_loss', 0),
            train_info.get('seeker', {}).get('value_loss', 0),
            train_info.get('hider', {}).get('policy_loss', 0),
            train_info.get('hider', {}).get('value_loss', 0),
            fps, time_elapsed,
            len(self.hider_zoo),
            len(self.seeker_zoo) if self.seeker_zoo else 0,
            zoo_rate,
        ]

        with open(self.metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        self.ep_reward_buffer = {'seeker': [], 'hider': []}
        self.ep_len_buffer = []

    def save_checkpoint(self, update: int):
        for role in ['seeker', 'hider']:
            path = os.path.join(self.output_dir, "checkpoints", f"{role}_{update:05d}.pt")
            self.policies[role].save_policy(path)

    def save_final(self):
        for role in ['seeker', 'hider']:
            path = os.path.join(self.output_dir, f"policy_{role}_final.pt")
            self.policies[role].save_policy(path)

        metadata = {
            'run_id': self.run_id,
            'total_episodes': self.total_episodes,
            'seeker_wins': self.seeker_wins,
            'hider_wins': self.hider_wins,
            'seeker_win_rate': self.seeker_wins / max(self.total_episodes, 1),
            'config': {
                'latest_opponent_prob': self.config.latest_opponent_prob,
                'use_seeker_zoo': self.config.use_seeker_zoo,
                'zoo_update_interval': self.config.zoo_update_interval,
                'zoo_max_size': self.config.zoo_max_size,
                'total_timesteps': self.config.total_timesteps,
                'lr': self.config.lr,
            },
            'zoo_stats': {
                'hider_zoo_final_size': len(self.hider_zoo),
                'seeker_zoo_final_size': len(self.seeker_zoo) if self.seeker_zoo else 0,
                'zoo_samples': self.zoo_samples,
                'latest_samples': self.latest_samples,
            }
        }
        with open(os.path.join(self.output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\nPolicies saved to {self.output_dir}")

    def train(self):
        print(f"Starting zoo training: {self.config.total_timesteps} timesteps")
        print(f"  Latest opponent prob (A): {self.config.latest_opponent_prob}")
        print(f"  Use seeker zoo: {self.config.use_seeker_zoo}")
        print(f"  Zoo update interval: {self.config.zoo_update_interval}")
        print(f"  Output: {self.output_dir}\n")

        buffer = RolloutBuffer(
            buffer_size=self.config.batch_size // self.config.num_envs + 1,
            obs_dim=self.env.obs_dim,
            act_dim=self.env.act_dim,
            num_envs=self.config.num_envs,
        )

        timesteps = 0
        update = 0
        start_time = time.time()

        while timesteps < self.config.total_timesteps:
            buffer.clear()
            rollout_info = self.collect_rollout(buffer)
            timesteps += buffer.get_size()
            update += 1

            train_info = self.train_step(buffer, rollout_info['last_values'])

            # Update zoos with current policies
            self._update_zoos(update)

            if update % self.config.log_interval == 0:
                elapsed = time.time() - start_time
                fps = timesteps / elapsed

                self._log_metrics(update, timesteps, train_info, fps, elapsed)

                total_eps = self.seeker_wins + self.hider_wins
                seeker_wr = self.seeker_wins / max(total_eps, 1)

                print(f"Update {update:4d} | Steps: {timesteps:8d} | "
                      f"Episodes: {self.total_episodes:5d} | "
                      f"Seeker WR: {seeker_wr:.2%} | "
                      f"Zoo: H={len(self.hider_zoo)},S={len(self.seeker_zoo) if self.seeker_zoo else 0} | "
                      f"FPS: {fps:.0f}")

            if update % self.config.save_interval == 0:
                self.save_checkpoint(update)

        elapsed = time.time() - start_time
        fps = timesteps / elapsed if elapsed > 0 else 0
        self._log_metrics(update, timesteps, train_info, fps, elapsed)

        self.save_final()

        print(f"\nTraining complete!")
        print(f"  Total time: {elapsed:.1f}s")
        print(f"  Total episodes: {self.total_episodes}")
        print(f"  Final seeker win rate: {self.seeker_wins / max(self.total_episodes, 1):.2%}")
        print(f"  Final zoo sizes: hider={len(self.hider_zoo)}, seeker={len(self.seeker_zoo) if self.seeker_zoo else 0}")


def main():
    parser = argparse.ArgumentParser(description="Zoo-based tag agent training")
    parser.add_argument("--timesteps", type=int, default=10_000_000,
                        help="Total training timesteps")
    parser.add_argument("--num-envs", type=int, default=64,
                        help="Number of parallel environments")
    parser.add_argument("--batch-size", type=int, default=2048,
                        help="Batch size for updates")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--latest-prob", "-A", type=float, default=0.1,
                        help="Probability of using latest opponent (A)")
    parser.add_argument("--use-seeker-zoo", action="store_true",
                        help="Also maintain a seeker zoo")
    parser.add_argument("--zoo-interval", type=int, default=50,
                        help="Add to zoo every N updates")
    parser.add_argument("--zoo-max-size", type=int, default=50,
                        help="Maximum zoo size")
    parser.add_argument("--output-dir", type=str,
                        default="experiments/results/zoo_training",
                        help="Output directory")
    parser.add_argument("--layout", type=str, default="four_corners",
                        choices=["empty", "four_corners", "central_cross"],
                        help="Arena layout")

    args = parser.parse_args()

    config = ZooTrainConfig(
        num_envs=args.num_envs,
        total_timesteps=args.timesteps,
        batch_size=args.batch_size,
        lr=args.lr,
        latest_opponent_prob=args.latest_prob,
        use_seeker_zoo=args.use_seeker_zoo,
        zoo_update_interval=args.zoo_interval,
        zoo_max_size=args.zoo_max_size,
        output_dir=args.output_dir,
    )

    env_config = TagEnvConfig(layout=args.layout)
    trainer = ZooTrainer(config, env_config=env_config)
    trainer.train()


if __name__ == "__main__":
    main()
