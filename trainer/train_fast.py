#!/usr/bin/env python3
"""Standalone training script for tag agents using the vectorized 2D environment."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    print("PyTorch required. Install with: pip install torch", file=sys.stderr)
    sys.exit(1)

from tag_env import VecTagEnv, TagEnvConfig
from ppo import PPOConfig, PPOAgent


@dataclass
class TrainConfig:
    """Training configuration."""
    # Environment
    num_envs: int = 64              # Number of parallel environments
    total_timesteps: int = 1_000_000  # Total training timesteps

    # PPO hyperparameters
    batch_size: int = 2048          # Samples per update
    gamma: float = 0.99             # Discount factor
    gae_lambda: float = 0.95        # GAE lambda
    clip_ratio: float = 0.2         # PPO clip ratio
    lr: float = 3e-4                # Learning rate
    train_iters: int = 10           # Epochs per update
    target_kl: float = 0.02         # KL divergence target

    # Logging
    log_interval: int = 10          # Log every N updates
    save_interval: int = 50         # Save checkpoint every N updates
    eval_episodes: int = 20         # Episodes for evaluation

    # Output
    output_dir: str = "trainer/logs/fast_train"


class RolloutBuffer:
    """Buffer for storing rollout data."""

    def __init__(self, buffer_size: int, obs_dim: int, act_dim: int, num_envs: int):
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Per-role buffers
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
        """Add a transition for all environments."""
        for role in ['seeker', 'hider']:
            self.obs[role].append(obs[role].copy())
            self.actions[role].append(actions[role].copy())
            self.rewards[role].append(rewards[role].copy())
            self.dones[role].append(dones.copy())
            self.log_probs[role].append(log_probs[role].copy())
            self.values[role].append(values[role].copy())
        self.ptr += 1

    def get_size(self) -> int:
        """Return total samples in buffer."""
        return self.ptr * self.num_envs

    def clear(self):
        """Clear the buffer."""
        for role in ['seeker', 'hider']:
            self.obs[role].clear()
            self.actions[role].clear()
            self.rewards[role].clear()
            self.dones[role].clear()
            self.log_probs[role].clear()
            self.values[role].clear()
        self.ptr = 0

    def compute_returns_and_advantages(self, last_values: Dict[str, np.ndarray],
                                       gamma: float, gae_lambda: float) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Compute returns and advantages using GAE."""
        result = {}

        for role in ['seeker', 'hider']:
            # Stack arrays: [timesteps, num_envs]
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

            # Flatten for training: [timesteps * num_envs]
            result[role] = (
                obs.reshape(-1, obs.shape[-1]),
                actions.reshape(-1, actions.shape[-1]),
                log_probs.reshape(-1),
                returns.reshape(-1),
                advantages.reshape(-1),
            )

        return result


class FastTrainer:
    """Efficient PPO trainer for tag game."""

    def __init__(self, config: TrainConfig, env_config: Optional[TagEnvConfig] = None):
        self.config = config
        self.env = VecTagEnv(num_envs=config.num_envs, config=env_config)

        # Initialize policies for both roles
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

        # Setup output directory
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(config.output_dir, self.run_id)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)

        # Metrics tracking
        self.metrics_path = os.path.join(self.output_dir, "metrics.csv")
        self._init_metrics_csv()

        # Episode statistics
        self.episode_rewards = {'seeker': [], 'hider': []}
        self.episode_lengths = []
        self.seeker_wins = 0
        self.hider_wins = 0
        self.total_episodes = 0

        # Running stats for normalization
        self.ep_reward_buffer = {'seeker': [], 'hider': []}
        self.ep_len_buffer = []

    def _init_metrics_csv(self):
        """Initialize metrics CSV file."""
        columns = [
            "update", "timesteps", "episodes", "seeker_reward_mean", "seeker_reward_std",
            "hider_reward_mean", "hider_reward_std", "episode_length_mean",
            "seeker_win_rate", "hider_win_rate", "seeker_policy_loss", "seeker_value_loss",
            "hider_policy_loss", "hider_value_loss", "fps", "time_elapsed"
        ]
        with open(self.metrics_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(columns)

    def _log_metrics(self, update: int, timesteps: int, train_info: Dict[str, Any],
                     fps: float, time_elapsed: float):
        """Log training metrics."""
        seeker_rewards = self.ep_reward_buffer.get('seeker', [0])
        hider_rewards = self.ep_reward_buffer.get('hider', [0])
        ep_lens = self.ep_len_buffer if self.ep_len_buffer else [0]

        total_eps = self.seeker_wins + self.hider_wins
        seeker_wr = self.seeker_wins / max(total_eps, 1)
        hider_wr = self.hider_wins / max(total_eps, 1)

        row = [
            update,
            timesteps,
            self.total_episodes,
            np.mean(seeker_rewards) if seeker_rewards else 0,
            np.std(seeker_rewards) if seeker_rewards else 0,
            np.mean(hider_rewards) if hider_rewards else 0,
            np.std(hider_rewards) if hider_rewards else 0,
            np.mean(ep_lens) if ep_lens else 0,
            seeker_wr,
            hider_wr,
            train_info.get('seeker', {}).get('policy_loss', 0),
            train_info.get('seeker', {}).get('value_loss', 0),
            train_info.get('hider', {}).get('policy_loss', 0),
            train_info.get('hider', {}).get('value_loss', 0),
            fps,
            time_elapsed,
        ]

        with open(self.metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        # Clear episode buffers
        self.ep_reward_buffer = {'seeker': [], 'hider': []}
        self.ep_len_buffer = []

    def act(self, obs: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray],
                                                        Dict[str, np.ndarray],
                                                        Dict[str, np.ndarray]]:
        """Get actions from policies for all environments."""
        actions = {}
        log_probs = {}
        values = {}

        for role in ['seeker', 'hider']:
            policy = self.policies[role]
            role_obs = obs[role]

            # Batch processing
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

    def get_values(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Get value estimates for observations."""
        values = {}
        for role in ['seeker', 'hider']:
            vals = np.zeros(self.config.num_envs, dtype=np.float32)
            for i in range(self.config.num_envs):
                vals[i] = self.policies[role].value(obs[role][i])
            values[role] = vals
        return values

    def collect_rollout(self, buffer: RolloutBuffer) -> Dict[str, Any]:
        """Collect rollout data until buffer is full."""
        obs = self.env.auto_reset()

        ep_rewards = {'seeker': np.zeros(self.config.num_envs),
                      'hider': np.zeros(self.config.num_envs)}
        ep_lengths = np.zeros(self.config.num_envs, dtype=np.int32)

        while buffer.get_size() < self.config.batch_size:
            actions, log_probs, values = self.act(obs)
            next_obs, rewards, dones, infos = self.env.step(actions)

            buffer.add(obs, actions, rewards, dones, log_probs, values)

            # Track episode statistics
            for role in ['seeker', 'hider']:
                ep_rewards[role] += rewards[role]
            ep_lengths += 1

            # Handle episode endings
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

        # Get final values for GAE computation
        last_values = self.get_values(obs)

        return {'last_values': last_values, 'last_obs': obs}

    def train_step(self, buffer: RolloutBuffer, last_values: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Perform one training update."""
        data = buffer.compute_returns_and_advantages(
            last_values, self.config.gamma, self.config.gae_lambda
        )

        train_info = {}
        for role in ['seeker', 'hider']:
            obs, actions, log_probs_old, returns, advantages = data[role]

            # Normalize advantages
            adv_mean = np.mean(advantages)
            adv_std = np.std(advantages) + 1e-8
            advantages = (advantages - adv_mean) / adv_std

            info = self.policies[role].update(
                obs, actions, log_probs_old, returns, advantages
            )
            train_info[role] = info

        return train_info

    def save_checkpoint(self, update: int):
        """Save policy checkpoints."""
        for role in ['seeker', 'hider']:
            path = os.path.join(self.output_dir, "checkpoints", f"{role}_{update:05d}.pt")
            self.policies[role].save_policy(path)

        # Also save to main trainer directory for easy access
        base_dir = os.path.dirname(self.output_dir)
        for role in ['seeker', 'hider']:
            path = os.path.join(os.path.dirname(__file__), f"policy_{role}.pt")
            self.policies[role].save_policy(path)

    def save_final(self):
        """Save final trained policies."""
        for role in ['seeker', 'hider']:
            # Save to run directory
            path = os.path.join(self.output_dir, f"policy_{role}_final.pt")
            self.policies[role].save_policy(path)

            # Save to trainer root for easy access
            root_path = os.path.join(os.path.dirname(__file__), f"policy_{role}.pt")
            self.policies[role].save_policy(root_path)

        # Save metadata
        metadata = {
            'run_id': self.run_id,
            'total_episodes': self.total_episodes,
            'seeker_wins': self.seeker_wins,
            'hider_wins': self.hider_wins,
            'seeker_win_rate': self.seeker_wins / max(self.total_episodes, 1),
            'config': {
                'num_envs': self.config.num_envs,
                'total_timesteps': self.config.total_timesteps,
                'batch_size': self.config.batch_size,
                'gamma': self.config.gamma,
                'gae_lambda': self.config.gae_lambda,
                'lr': self.config.lr,
            }
        }
        with open(os.path.join(self.output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\nPolicies saved to {self.output_dir}")
        print(f"Also copied to trainer/policy_seeker.pt and trainer/policy_hider.pt")

    def load_checkpoint(self, seeker_path: Optional[str] = None, hider_path: Optional[str] = None):
        """Load policy checkpoints."""
        if seeker_path and os.path.exists(seeker_path):
            self.policies['seeker'].load_policy(seeker_path)
            print(f"Loaded seeker policy from {seeker_path}")

        if hider_path and os.path.exists(hider_path):
            self.policies['hider'].load_policy(hider_path)
            print(f"Loaded hider policy from {hider_path}")

    def train(self):
        """Main training loop."""
        print(f"Starting training: {self.config.total_timesteps} timesteps")
        print(f"  Environments: {self.config.num_envs}")
        print(f"  Batch size: {self.config.batch_size}")
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

            # Logging
            if update % self.config.log_interval == 0:
                elapsed = time.time() - start_time
                fps = timesteps / elapsed

                self._log_metrics(update, timesteps, train_info, fps, elapsed)

                total_eps = self.seeker_wins + self.hider_wins
                seeker_wr = self.seeker_wins / max(total_eps, 1)

                print(f"Update {update:4d} | Steps: {timesteps:8d} | "
                      f"Episodes: {self.total_episodes:5d} | "
                      f"Seeker WR: {seeker_wr:.2%} | "
                      f"FPS: {fps:.0f}")

            # Checkpointing
            if update % self.config.save_interval == 0:
                self.save_checkpoint(update)

        # Always log final metrics
        elapsed = time.time() - start_time
        fps = timesteps / elapsed if elapsed > 0 else 0
        self._log_metrics(update, timesteps, train_info, fps, elapsed)

        self.save_final()

        print(f"\nTraining complete!")
        print(f"  Total time: {elapsed:.1f}s")
        print(f"  Total episodes: {self.total_episodes}")
        print(f"  Final seeker win rate: {self.seeker_wins / max(self.total_episodes, 1):.2%}")


def main():
    parser = argparse.ArgumentParser(description="Fast tag agent training")
    parser.add_argument("--timesteps", type=int, default=500_000,
                        help="Total training timesteps (default: 500000)")
    parser.add_argument("--num-envs", type=int, default=64,
                        help="Number of parallel environments (default: 64)")
    parser.add_argument("--batch-size", type=int, default=2048,
                        help="Batch size for updates (default: 2048)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate (default: 3e-4)")
    parser.add_argument("--load-seeker", type=str, default=None,
                        help="Path to seeker policy checkpoint")
    parser.add_argument("--load-hider", type=str, default=None,
                        help="Path to hider policy checkpoint")
    parser.add_argument("--output-dir", type=str, default="trainer/logs/fast_train",
                        help="Output directory")
    parser.add_argument("--layout", type=str, default="empty",
                        choices=["empty", "four_corners", "central_cross"],
                        help="Arena layout with obstacles (default: empty)")

    args = parser.parse_args()

    config = TrainConfig(
        num_envs=args.num_envs,
        total_timesteps=args.timesteps,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output_dir,
    )

    # Create environment config with layout
    env_config = TagEnvConfig(layout=args.layout)

    trainer = FastTrainer(config, env_config=env_config)

    if args.load_seeker or args.load_hider:
        trainer.load_checkpoint(args.load_seeker, args.load_hider)

    trainer.train()


if __name__ == "__main__":
    main()
