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
    latest_opponent_prob: float = 0.1  # 1-A: probability of latest opponent (A = zoo prob)
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
    """Buffer for storing single-role rollout data."""

    def __init__(self, obs_dim: int, act_dim: int, num_envs: int):
        self.num_envs = num_envs
        self.obs_list: List[np.ndarray] = []
        self.actions_list: List[np.ndarray] = []
        self.rewards_list: List[np.ndarray] = []
        self.dones_list: List[np.ndarray] = []
        self.log_probs_list: List[np.ndarray] = []
        self.values_list: List[np.ndarray] = []
        self.ptr = 0

    def add(self, obs: np.ndarray, actions: np.ndarray, rewards: np.ndarray,
            dones: np.ndarray, log_probs: np.ndarray, values: np.ndarray):
        self.obs_list.append(obs.copy())
        self.actions_list.append(actions.copy())
        self.rewards_list.append(rewards.copy())
        self.dones_list.append(dones.copy())
        self.log_probs_list.append(log_probs.copy())
        self.values_list.append(values.copy())
        self.ptr += 1

    def get_size(self) -> int:
        return self.ptr * self.num_envs

    def clear(self):
        self.obs_list.clear()
        self.actions_list.clear()
        self.rewards_list.clear()
        self.dones_list.clear()
        self.log_probs_list.clear()
        self.values_list.clear()
        self.ptr = 0

    def compute_returns_and_advantages(self, last_values: np.ndarray,
                                       gamma: float, gae_lambda: float) -> Tuple[np.ndarray, ...]:
        obs = np.stack(self.obs_list)
        actions = np.stack(self.actions_list)
        rewards = np.stack(self.rewards_list)
        dones = np.stack(self.dones_list)
        log_probs = np.stack(self.log_probs_list)
        values = np.stack(self.values_list)

        T, N = rewards.shape
        advantages = np.zeros((T, N), dtype=np.float32)
        last_gae = 0

        for t in reversed(range(T)):
            if t == T - 1:
                next_value = last_values
                next_done = np.zeros(N)
            else:
                next_value = values[t + 1]
                next_done = dones[t + 1]

            mask = 1.0 - next_done.astype(np.float32)
            delta = rewards[t] + gamma * next_value * mask - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * mask * last_gae

        returns = advantages + values

        return (
            obs.reshape(-1, obs.shape[-1]),
            actions.reshape(-1, actions.shape[-1]),
            log_probs.reshape(-1),
            returns.reshape(-1),
            advantages.reshape(-1),
        )


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

        # Resume state
        self._resume_update = 0
        self._resume_timesteps = 0

        # Stats
        self.ep_reward_buffer = {'seeker': [], 'hider': []}
        self.ep_len_buffer = []
        self.seeker_wins = 0
        self.hider_wins = 0
        self.total_episodes = 0

        # Zoo sampling stats
        self.zoo_samples = {'seeker': 0, 'hider': 0}
        self.latest_samples = {'seeker': 0, 'hider': 0}

    def _init_metrics_csv(self, append: bool = False):
        if append:
            return  # Keep existing file, will append
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

    def _act_batch(self, policy: PPOAgent, obs_batch: np.ndarray
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get actions, log_probs, values for a batch of observations."""
        n = obs_batch.shape[0]
        acts = np.zeros((n, self.env.act_dim), dtype=np.float32)
        lps = np.zeros(n, dtype=np.float32)
        vals = np.zeros(n, dtype=np.float32)
        for i in range(n):
            act, lp, val = policy.act(obs_batch[i])
            acts[i] = act
            lps[i] = lp
            vals[i] = val
        return acts, lps, vals

    def _act_batch_actions_only(self, policy: PPOAgent, obs_batch: np.ndarray
                                ) -> np.ndarray:
        """Get actions only (for opponent, no training data needed)."""
        n = obs_batch.shape[0]
        acts = np.zeros((n, self.env.act_dim), dtype=np.float32)
        for i in range(n):
            act, _, _ = policy.act(obs_batch[i])
            acts[i] = act
        return acts

    def collect_rollout(self, buffer: RolloutBuffer, training_role: str) -> np.ndarray:
        """Collect rollout for a specific role.

        The training_role uses its learning policy (data collected for PPO).
        The opponent uses a sampled policy from the zoo (or latest).

        Returns last_values for GAE computation.
        """
        opponent_role = 'hider' if training_role == 'seeker' else 'seeker'

        # Sample opponent for this rollout
        self._sample_opponents()
        opponent_policy = self.current_opponents[opponent_role]

        obs = self.env.auto_reset()
        ep_rewards = {'seeker': np.zeros(self.config.num_envs),
                      'hider': np.zeros(self.config.num_envs)}
        ep_lengths = np.zeros(self.config.num_envs, dtype=np.int32)

        while buffer.get_size() < self.config.batch_size:
            # Training role: learning policy (collect obs, action, logp, value)
            train_acts, train_lps, train_vals = self._act_batch(
                self.policies[training_role], obs[training_role])

            # Opponent: sampled policy (actions only for env stepping)
            opp_acts = self._act_batch_actions_only(opponent_policy, obs[opponent_role])

            # Step env with training role's own actions + opponent's actions
            env_actions = {training_role: train_acts, opponent_role: opp_acts}
            next_obs, rewards, dones, infos = self.env.step(env_actions)

            # Store only the training role's on-policy data
            buffer.add(obs[training_role], train_acts, rewards[training_role],
                       dones, train_lps, train_vals)

            # Track episode stats (from both roles for logging)
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

        # Bootstrap value for GAE
        last_vals = np.zeros(self.config.num_envs, dtype=np.float32)
        for i in range(self.config.num_envs):
            last_vals[i] = self.policies[training_role].value(obs[training_role][i])

        return last_vals

    def train_step(self, role: str, buffer: RolloutBuffer,
                   last_values: np.ndarray) -> Dict[str, float]:
        """Run PPO update for a single role."""
        obs, actions, log_probs_old, returns, advantages = \
            buffer.compute_returns_and_advantages(
                last_values, self.config.gamma, self.config.gae_lambda)

        info = self.policies[role].update(
            obs, actions, log_probs_old, returns, advantages
        )
        return info

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

    def resume_from(self, resume_dir: str):
        """Resume training from an existing run directory."""
        # Remove the empty directory created by __init__
        init_dir = self.output_dir
        if os.path.isdir(init_dir) and init_dir != os.path.abspath(resume_dir):
            import shutil
            shutil.rmtree(init_dir, ignore_errors=True)

        resume_dir = os.path.abspath(resume_dir)
        if not os.path.isdir(resume_dir):
            raise FileNotFoundError(f"Resume directory not found: {resume_dir}")

        # Find latest checkpoint update number
        ckpt_dir = os.path.join(resume_dir, "checkpoints")
        if not os.path.isdir(ckpt_dir):
            raise FileNotFoundError(f"No checkpoints directory in: {resume_dir}")

        updates = set()
        for fname in os.listdir(ckpt_dir):
            if fname.endswith(".pt"):
                # e.g. hider_00500.pt -> 500
                try:
                    num = int(fname.split("_")[1].split(".")[0])
                    updates.add(num)
                except (IndexError, ValueError):
                    continue

        if not updates:
            raise FileNotFoundError(f"No checkpoints found in: {ckpt_dir}")

        latest_update = max(updates)

        # Load policies
        for role in ['seeker', 'hider']:
            path = os.path.join(ckpt_dir, f"{role}_{latest_update:05d}.pt")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing checkpoint: {path}")
            self.policies[role].load_policy(path)
            print(f"  Loaded {role} from {path}")

        # Read metrics.csv to get last timesteps/episodes
        metrics_path = os.path.join(resume_dir, "metrics.csv")
        last_timesteps = 0
        last_episodes = 0
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                reader = csv.reader(f)
                header = next(reader)
                for row in reader:
                    if row:
                        last_timesteps = int(float(row[1]))
                        last_episodes = int(float(row[2]))

        self._resume_update = latest_update
        self._resume_timesteps = last_timesteps
        self.total_episodes = last_episodes

        # Switch output to the existing run directory
        self.output_dir = resume_dir
        self.metrics_path = os.path.join(resume_dir, "metrics.csv")
        self._init_metrics_csv(append=True)

        # Rebuild zoo from saved checkpoints
        for upd in sorted(updates):
            hider_path = os.path.join(ckpt_dir, f"hider_{upd:05d}.pt")
            if os.path.exists(hider_path):
                agent = PPOAgent(self.hider_zoo.ppo_cfg)
                agent.load_policy(hider_path)
                self.hider_zoo.add(agent, upd)
            if self.seeker_zoo is not None:
                seeker_path = os.path.join(ckpt_dir, f"seeker_{upd:05d}.pt")
                if os.path.exists(seeker_path):
                    agent = PPOAgent(self.seeker_zoo.ppo_cfg)
                    agent.load_policy(seeker_path)
                    self.seeker_zoo.add(agent, upd)

        print(f"\nResuming from update {latest_update}, timesteps {last_timesteps}")
        print(f"  Zoo sizes: hider={len(self.hider_zoo)}, seeker={len(self.seeker_zoo) if self.seeker_zoo else 0}")

    def train(self):
        print(f"Starting zoo training: {self.config.total_timesteps} timesteps")
        print(f"  A (zoo prob): {1 - self.config.latest_opponent_prob:.2f} (latest_prob={self.config.latest_opponent_prob})")
        print(f"  Use seeker zoo: {self.config.use_seeker_zoo}")
        print(f"  Zoo update interval: {self.config.zoo_update_interval}")
        print(f"  Output: {self.output_dir}\n")

        buffer = RolloutBuffer(
            obs_dim=self.env.obs_dim,
            act_dim=self.env.act_dim,
            num_envs=self.config.num_envs,
        )

        timesteps = self._resume_timesteps
        update = self._resume_update
        start_time = time.time()

        while timesteps < self.config.total_timesteps:
            train_info = {}

            # Two-phase update: each role gets its own rollout against
            # a (possibly zoo-sampled) opponent, ensuring on-policy data
            for role in ['seeker', 'hider']:
                buffer.clear()
                last_values = self.collect_rollout(buffer, training_role=role)
                timesteps += buffer.get_size()
                info = self.train_step(role, buffer, last_values)
                train_info[role] = info

            update += 1

            # Update zoos with current policies
            self._update_zoos(update)

            if update % self.config.log_interval == 0:
                elapsed = time.time() - start_time
                new_steps = timesteps - self._resume_timesteps
                fps = new_steps / elapsed if elapsed > 0 else 0

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
        new_steps = timesteps - self._resume_timesteps
        fps = new_steps / elapsed if elapsed > 0 else 0
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
                        help="Probability of using latest opponent (1-A; legacy naming)")
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
                        choices=["empty", "four_corners", "central_cross", "playground"],
                        help="Arena layout")
    parser.add_argument("--enable-sprint", action="store_true",
                        help="Enable stamina/sprint system")
    parser.add_argument("--hider-speed-mult", type=float, default=1.0,
                        help="Hider base speed multiplier (e.g. 1.1 for 10%% advantage)")
    parser.add_argument("--sprint-speed-mult", type=float, default=1.5,
                        help="Max speed multiplier when sprinting")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from an existing run directory (the timestamped subdir)")

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

    env_config = TagEnvConfig(
        layout=args.layout,
        enable_sprint=args.enable_sprint,
        hider_speed_mult=args.hider_speed_mult,
        sprint_speed_mult=args.sprint_speed_mult,
    )
    trainer = ZooTrainer(config, env_config=env_config)

    if args.resume:
        trainer.resume_from(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()
