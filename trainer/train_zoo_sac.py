#!/usr/bin/env python3
"""
SAC-based zoo training for tag agents.

Key differences from PPO zoo trainer:
- Off-policy: both roles collect transitions simultaneously (no two-phase rollout)
- Replay buffer retains all past transitions for sample-efficient learning
- Gradient updates from replay buffer samples (not on-policy rollouts)
- Warmup period with random actions to seed diverse experience

Supports same zoo opponent sampling as PPO trainer:
- A% of the time: play against latest self-play opponent
- (100-A)% of the time: uniformly sample from checkpoint zoo
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
from sac import SACConfig, SACAgent, ReplayBuffer
from ppo import PPOConfig, PPOAgent


@dataclass
class SACZooTrainConfig:
    """Training configuration for SAC with zoo sampling."""
    # Environment
    num_envs: int = 64
    total_timesteps: int = 1_000_000

    # SAC hyperparameters
    hidden_dim: int = 256
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    init_alpha: float = 0.2
    buffer_size: int = 500_000
    batch_size: int = 256
    warmup_steps: int = 10_000
    updates_per_step: int = 1  # Gradient steps per env step

    # Zoo parameters
    latest_opponent_prob: float = 0.1
    use_seeker_zoo: bool = False
    zoo_update_interval: int = 50  # Add to zoo every N updates
    zoo_max_size: int = 50
    zoo_resample_interval: int = 5  # Re-sample zoo opponents every N updates

    # Logging
    log_interval: int = 10
    save_interval: int = 50

    # Output
    output_dir: str = "experiments/results/sac_zoo_training"


class PolymorphicOpponentZoo:
    """Opponent zoo that handles both SAC and PPO checkpoints."""

    def __init__(self, obs_dim: int, act_dim: int, max_size: int = 50,
                 hidden_dim: int = 256):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.max_size = max_size
        self.hidden_dim = hidden_dim
        self.checkpoints: List[Dict[str, Any]] = []

    def add_from_sac(self, agent: SACAgent, update: int):
        """Add an SAC agent's actor weights to the zoo."""
        state = {
            'type': 'sac',
            'actor': {k: v.clone() for k, v in agent.actor.state_dict().items()},
            'hidden_dim': agent.cfg.hidden_dim,
            'update': update,
        }
        self.checkpoints.append(state)
        if len(self.checkpoints) > self.max_size:
            self.checkpoints.pop(0)

    def add_from_file(self, path: str, update: int):
        """Load a checkpoint file and add to zoo."""
        ckpt = torch.load(path, map_location='cpu')
        if isinstance(ckpt, dict) and ckpt.get('type') == 'sac':
            state = {
                'type': 'sac',
                'actor': ckpt['actor'],
                'hidden_dim': ckpt.get('config', {}).get('hidden_dim', self.hidden_dim),
                'update': update,
            }
        else:
            # PPO checkpoint
            state = {
                'type': 'ppo',
                'pi': ckpt.get('pi', ckpt),
                'vf': ckpt.get('vf', None),
                'update': update,
            }
        self.checkpoints.append(state)
        if len(self.checkpoints) > self.max_size:
            self.checkpoints.pop(0)

    def sample(self):
        """Sample a random opponent from the zoo. Returns an agent that has .act()."""
        if not self.checkpoints:
            raise ValueError("Zoo is empty!")

        state = random.choice(self.checkpoints)

        if state['type'] == 'sac':
            cfg = SACConfig(
                obs_dim=self.obs_dim,
                act_dim=self.act_dim,
                hidden_dim=state['hidden_dim'],
            )
            agent = SACAgent(cfg)
            agent.actor.load_state_dict(state['actor'])
            return agent
        else:
            # PPO
            cfg = PPOConfig(obs_dim=self.obs_dim, act_dim=self.act_dim)
            agent = PPOAgent(cfg)
            agent.pi.load_state_dict(state['pi'])
            if state.get('vf') is not None:
                agent.vf.load_state_dict(state['vf'])
            return agent

    def __len__(self):
        return len(self.checkpoints)


class SACZooTrainer:
    """SAC trainer with zoo-based opponent sampling."""

    def __init__(self, config: SACZooTrainConfig,
                 env_config: Optional[TagEnvConfig] = None):
        self.config = config
        self.env = VecTagEnv(num_envs=config.num_envs, config=env_config)

        sac_cfg = SACConfig(
            obs_dim=self.env.obs_dim,
            act_dim=self.env.act_dim,
            hidden_dim=config.hidden_dim,
            actor_lr=config.actor_lr,
            critic_lr=config.critic_lr,
            alpha_lr=config.alpha_lr,
            gamma=config.gamma,
            tau=config.tau,
            init_alpha=config.init_alpha,
            buffer_size=config.buffer_size,
            batch_size=config.batch_size,
            warmup_steps=config.warmup_steps,
        )

        # Learning policies
        self.policies = {
            'seeker': SACAgent(sac_cfg),
            'hider': SACAgent(sac_cfg),
        }

        # Per-role replay buffers (separate because reward signals are opposite)
        self.buffers = {
            'seeker': ReplayBuffer(self.env.obs_dim, self.env.act_dim,
                                   config.buffer_size),
            'hider': ReplayBuffer(self.env.obs_dim, self.env.act_dim,
                                  config.buffer_size),
        }

        # Opponent zoos
        self.hider_zoo = PolymorphicOpponentZoo(
            self.env.obs_dim, self.env.act_dim, config.zoo_max_size,
            config.hidden_dim)
        self.seeker_zoo = PolymorphicOpponentZoo(
            self.env.obs_dim, self.env.act_dim, config.zoo_max_size,
            config.hidden_dim) if config.use_seeker_zoo else None

        # Current opponents
        self.current_opponents = {
            'seeker': None,  # Opponent for hider (seeker policy)
            'hider': None,   # Opponent for seeker (hider policy)
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

        # Per-episode accumulators (per env)
        self.ep_rewards = {
            'seeker': np.zeros(config.num_envs),
            'hider': np.zeros(config.num_envs),
        }
        self.ep_lengths = np.zeros(config.num_envs, dtype=np.int32)

        # Zoo sampling stats
        self.zoo_samples = {'seeker': 0, 'hider': 0}
        self.latest_samples = {'seeker': 0, 'hider': 0}

        # Training info accumulator
        self.train_info_accum = {'seeker': [], 'hider': []}

    def _init_metrics_csv(self, append: bool = False):
        if append:
            return
        columns = [
            "update", "timesteps", "episodes",
            "seeker_reward_mean", "seeker_reward_std",
            "hider_reward_mean", "hider_reward_std",
            "episode_length_mean",
            "seeker_win_rate", "hider_win_rate",
            "seeker_critic_loss", "seeker_actor_loss",
            "hider_critic_loss", "hider_actor_loss",
            "seeker_alpha", "hider_alpha",
            "seeker_entropy", "hider_entropy",
            "fps", "time_elapsed",
            "hider_zoo_size", "seeker_zoo_size", "zoo_sample_rate",
            "seeker_buffer_size", "hider_buffer_size",
        ]
        with open(self.metrics_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(columns)

    def _sample_opponents(self):
        """Sample opponents for upcoming env steps."""
        # Sample hider opponent (for seeker to play against)
        if len(self.hider_zoo) > 0 and random.random() > self.config.latest_opponent_prob:
            self.current_opponents['hider'] = self.hider_zoo.sample()
            self.zoo_samples['hider'] += 1
        else:
            self.current_opponents['hider'] = None  # Use learning policy
            self.latest_samples['hider'] += 1

        # Sample seeker opponent (for hider to play against)
        if self.seeker_zoo is not None:
            if len(self.seeker_zoo) > 0 and random.random() > self.config.latest_opponent_prob:
                self.current_opponents['seeker'] = self.seeker_zoo.sample()
                self.zoo_samples['seeker'] += 1
            else:
                self.current_opponents['seeker'] = None
                self.latest_samples['seeker'] += 1
        else:
            self.current_opponents['seeker'] = None

    def _update_zoos(self, update: int):
        """Add current policies to zoos."""
        if update % self.config.zoo_update_interval == 0:
            self.hider_zoo.add_from_sac(self.policies['hider'], update)
            if self.seeker_zoo is not None:
                self.seeker_zoo.add_from_sac(self.policies['seeker'], update)

    def _get_policy_for_role(self, role: str):
        """Get the policy to use for a role (zoo opponent or learning policy)."""
        opponent = self.current_opponents[role]
        if opponent is not None:
            return opponent
        return self.policies[role]

    def _act_batch(self, policy, obs_batch: np.ndarray, use_random: bool = False
                   ) -> np.ndarray:
        """Get actions for a batch of observations."""
        n = obs_batch.shape[0]
        acts = np.zeros((n, self.env.act_dim), dtype=np.float32)
        if use_random:
            acts = np.random.uniform(-1.0, 1.0,
                                     (n, self.env.act_dim)).astype(np.float32)
        else:
            for i in range(n):
                act, _, _ = policy.act(obs_batch[i])
                acts[i] = act
        return acts

    def _collect_step(self, obs: Dict[str, np.ndarray], warmup: bool
                      ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray],
                                 np.ndarray, Dict[str, Any]]:
        """Collect one env step, storing transitions in replay buffers.

        Both roles act simultaneously. Off-policy allows this.
        """
        actions = {}
        for role in ['seeker', 'hider']:
            policy = self._get_policy_for_role(role)
            actions[role] = self._act_batch(policy, obs[role],
                                            use_random=warmup)

        next_obs, rewards, dones, infos = self.env.step(actions)

        # Store transitions for both roles
        for role in ['seeker', 'hider']:
            self.buffers[role].add_batch(
                obs[role], actions[role], rewards[role],
                next_obs[role], dones.astype(np.float32),
            )

        # Track episode stats
        for role in ['seeker', 'hider']:
            self.ep_rewards[role] += rewards[role]
        self.ep_lengths += 1

        done_ids = np.where(dones)[0]
        for eid in done_ids:
            self.total_episodes += 1
            self.ep_reward_buffer['seeker'].append(self.ep_rewards['seeker'][eid])
            self.ep_reward_buffer['hider'].append(self.ep_rewards['hider'][eid])
            self.ep_len_buffer.append(self.ep_lengths[eid])

            if infos['tagged'][eid]:
                self.seeker_wins += 1
            else:
                self.hider_wins += 1

            self.ep_rewards['seeker'][eid] = 0
            self.ep_rewards['hider'][eid] = 0
            self.ep_lengths[eid] = 0

        return next_obs, rewards, dones, infos

    def _log_metrics(self, update: int, timesteps: int, fps: float,
                     time_elapsed: float):
        seeker_rewards = self.ep_reward_buffer.get('seeker', [0])
        hider_rewards = self.ep_reward_buffer.get('hider', [0])
        ep_lens = self.ep_len_buffer if self.ep_len_buffer else [0]

        total_eps = self.seeker_wins + self.hider_wins
        seeker_wr = self.seeker_wins / max(total_eps, 1)
        hider_wr = self.hider_wins / max(total_eps, 1)

        total_samples = sum(self.zoo_samples.values()) + sum(self.latest_samples.values())
        zoo_rate = sum(self.zoo_samples.values()) / max(total_samples, 1)

        # Average training info
        def avg_info(infos, key):
            vals = [i.get(key, 0) for i in infos if i]
            return np.mean(vals) if vals else 0

        seeker_infos = self.train_info_accum['seeker']
        hider_infos = self.train_info_accum['hider']

        row = [
            update, timesteps, self.total_episodes,
            np.mean(seeker_rewards) if seeker_rewards else 0,
            np.std(seeker_rewards) if seeker_rewards else 0,
            np.mean(hider_rewards) if hider_rewards else 0,
            np.std(hider_rewards) if hider_rewards else 0,
            np.mean(ep_lens) if ep_lens else 0,
            seeker_wr, hider_wr,
            avg_info(seeker_infos, 'critic_loss'),
            avg_info(seeker_infos, 'actor_loss'),
            avg_info(hider_infos, 'critic_loss'),
            avg_info(hider_infos, 'actor_loss'),
            avg_info(seeker_infos, 'alpha'),
            avg_info(hider_infos, 'alpha'),
            avg_info(seeker_infos, 'entropy'),
            avg_info(hider_infos, 'entropy'),
            fps, time_elapsed,
            len(self.hider_zoo),
            len(self.seeker_zoo) if self.seeker_zoo else 0,
            zoo_rate,
            self.buffers['seeker'].size,
            self.buffers['hider'].size,
        ]

        with open(self.metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        self.ep_reward_buffer = {'seeker': [], 'hider': []}
        self.ep_len_buffer = []
        self.train_info_accum = {'seeker': [], 'hider': []}

    def save_checkpoint(self, update: int):
        for role in ['seeker', 'hider']:
            path = os.path.join(self.output_dir, "checkpoints",
                                f"{role}_{update:05d}.pt")
            self.policies[role].save_policy(path)

    def save_final(self):
        for role in ['seeker', 'hider']:
            path = os.path.join(self.output_dir, f"policy_{role}_final.pt")
            self.policies[role].save_policy(path)

        metadata = {
            'run_id': self.run_id,
            'algorithm': 'sac',
            'total_episodes': self.total_episodes,
            'seeker_wins': self.seeker_wins,
            'hider_wins': self.hider_wins,
            'seeker_win_rate': self.seeker_wins / max(self.total_episodes, 1),
            'config': {
                'hidden_dim': self.config.hidden_dim,
                'actor_lr': self.config.actor_lr,
                'critic_lr': self.config.critic_lr,
                'gamma': self.config.gamma,
                'tau': self.config.tau,
                'buffer_size': self.config.buffer_size,
                'batch_size': self.config.batch_size,
                'warmup_steps': self.config.warmup_steps,
                'updates_per_step': self.config.updates_per_step,
                'latest_opponent_prob': self.config.latest_opponent_prob,
                'use_seeker_zoo': self.config.use_seeker_zoo,
                'zoo_update_interval': self.config.zoo_update_interval,
                'zoo_max_size': self.config.zoo_max_size,
                'total_timesteps': self.config.total_timesteps,
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
        init_dir = self.output_dir
        if os.path.isdir(init_dir) and init_dir != os.path.abspath(resume_dir):
            import shutil
            shutil.rmtree(init_dir, ignore_errors=True)

        resume_dir = os.path.abspath(resume_dir)
        if not os.path.isdir(resume_dir):
            raise FileNotFoundError(f"Resume directory not found: {resume_dir}")

        ckpt_dir = os.path.join(resume_dir, "checkpoints")
        if not os.path.isdir(ckpt_dir):
            raise FileNotFoundError(f"No checkpoints directory in: {resume_dir}")

        updates = set()
        for fname in os.listdir(ckpt_dir):
            if fname.endswith(".pt"):
                try:
                    num = int(fname.split("_")[1].split(".")[0])
                    updates.add(num)
                except (IndexError, ValueError):
                    continue

        if not updates:
            raise FileNotFoundError(f"No checkpoints found in: {ckpt_dir}")

        latest_update = max(updates)

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

        self.output_dir = resume_dir
        self.metrics_path = os.path.join(resume_dir, "metrics.csv")
        self._init_metrics_csv(append=True)

        # Rebuild zoos from saved checkpoints
        for upd in sorted(updates):
            hider_path = os.path.join(ckpt_dir, f"hider_{upd:05d}.pt")
            if os.path.exists(hider_path):
                self.hider_zoo.add_from_file(hider_path, upd)
            if self.seeker_zoo is not None:
                seeker_path = os.path.join(ckpt_dir, f"seeker_{upd:05d}.pt")
                if os.path.exists(seeker_path):
                    self.seeker_zoo.add_from_file(seeker_path, upd)

        print(f"\nResuming from update {latest_update}, timesteps {last_timesteps}")
        print(f"  Zoo sizes: hider={len(self.hider_zoo)}, seeker={len(self.seeker_zoo) if self.seeker_zoo else 0}")

    def train(self):
        print(f"Starting SAC zoo training: {self.config.total_timesteps} timesteps")
        print(f"  A (zoo prob): {1 - self.config.latest_opponent_prob:.2f} (latest_prob={self.config.latest_opponent_prob})")
        print(f"  Use seeker zoo: {self.config.use_seeker_zoo}")
        print(f"  Hidden dim: {self.config.hidden_dim}")
        print(f"  Buffer size: {self.config.buffer_size}")
        print(f"  Warmup steps: {self.config.warmup_steps}")
        print(f"  Updates per step: {self.config.updates_per_step}")
        print(f"  Output: {self.output_dir}\n")

        timesteps = self._resume_timesteps
        update = self._resume_update
        start_time = time.time()

        obs = self.env.auto_reset()

        # Sample initial opponents
        self._sample_opponents()

        while timesteps < self.config.total_timesteps:
            warmup = timesteps < self.config.warmup_steps

            # Collect one env step (all envs simultaneously)
            next_obs, _, dones, _ = self._collect_step(obs, warmup=warmup)
            timesteps += self.config.num_envs

            # Auto-reset done envs
            obs = self.env.auto_reset()

            # Run gradient updates from replay buffer (post-warmup)
            if not warmup:
                for _ in range(self.config.updates_per_step):
                    for role in ['seeker', 'hider']:
                        if self.buffers[role].size >= self.config.batch_size:
                            batch = self.buffers[role].sample(
                                self.config.batch_size)
                            info = self.policies[role].update(batch)
                            self.train_info_accum[role].append(info)

            # Periodic updates (keyed on env steps, scaled to match PPO update cadence)
            steps_per_update = self.config.num_envs * 32  # ~2048 steps
            if timesteps % steps_per_update < self.config.num_envs:
                update += 1

                self._update_zoos(update)

                # Re-sample opponents periodically
                if update % self.config.zoo_resample_interval == 0:
                    self._sample_opponents()

                if update % self.config.log_interval == 0:
                    elapsed = time.time() - start_time
                    new_steps = timesteps - self._resume_timesteps
                    fps = new_steps / elapsed if elapsed > 0 else 0

                    self._log_metrics(update, timesteps, fps, elapsed)

                    total_eps = self.seeker_wins + self.hider_wins
                    seeker_wr = self.seeker_wins / max(total_eps, 1)

                    print(
                        f"Update {update:4d} | Steps: {timesteps:8d} | "
                        f"Episodes: {self.total_episodes:5d} | "
                        f"Seeker WR: {seeker_wr:.2%} | "
                        f"Zoo: H={len(self.hider_zoo)},"
                        f"S={len(self.seeker_zoo) if self.seeker_zoo else 0} | "
                        f"Alpha: S={self.policies['seeker'].alpha:.3f},"
                        f"H={self.policies['hider'].alpha:.3f} | "
                        f"Buf: {self.buffers['seeker'].size}/{self.config.buffer_size} | "
                        f"FPS: {fps:.0f}"
                    )

                if update % self.config.save_interval == 0:
                    self.save_checkpoint(update)

        elapsed = time.time() - start_time
        new_steps = timesteps - self._resume_timesteps
        fps = new_steps / elapsed if elapsed > 0 else 0
        self._log_metrics(update, timesteps, fps, elapsed)

        self.save_final()

        print(f"\nTraining complete!")
        print(f"  Total time: {elapsed:.1f}s")
        print(f"  Total episodes: {self.total_episodes}")
        print(f"  Final seeker win rate: "
              f"{self.seeker_wins / max(self.total_episodes, 1):.2%}")
        print(f"  Final zoo sizes: hider={len(self.hider_zoo)}, "
              f"seeker={len(self.seeker_zoo) if self.seeker_zoo else 0}")


def main():
    parser = argparse.ArgumentParser(
        description="SAC zoo-based tag agent training")
    parser.add_argument("--timesteps", type=int, default=10_000_000,
                        help="Total training timesteps")
    parser.add_argument("--num-envs", type=int, default=64,
                        help="Number of parallel environments")
    parser.add_argument("--hidden-dim", type=int, default=256,
                        help="Hidden layer size for actor/critic")
    parser.add_argument("--actor-lr", type=float, default=3e-4,
                        help="Actor learning rate")
    parser.add_argument("--critic-lr", type=float, default=3e-4,
                        help="Critic learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.005,
                        help="Soft target update rate")
    parser.add_argument("--buffer-size", type=int, default=500_000,
                        help="Replay buffer size")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size for gradient updates")
    parser.add_argument("--warmup-steps", type=int, default=10_000,
                        help="Random action steps before training")
    parser.add_argument("--updates-per-step", type=int, default=1,
                        help="Gradient steps per env step")
    parser.add_argument("--latest-prob", "-A", type=float, default=0.1,
                        help="Probability of using latest opponent (1-A; legacy naming)")
    parser.add_argument("--use-seeker-zoo", action="store_true",
                        help="Also maintain a seeker zoo")
    parser.add_argument("--zoo-interval", type=int, default=50,
                        help="Add to zoo every N updates")
    parser.add_argument("--zoo-max-size", type=int, default=50,
                        help="Maximum zoo size")
    parser.add_argument("--output-dir", type=str,
                        default="experiments/results/sac_zoo_training",
                        help="Output directory")
    parser.add_argument("--layout", type=str, default="four_corners",
                        choices=["empty", "four_corners", "central_cross",
                                 "playground"],
                        help="Arena layout")
    parser.add_argument("--enable-sprint", action="store_true",
                        help="Enable stamina/sprint system")
    parser.add_argument("--hider-speed-mult", type=float, default=1.0,
                        help="Hider base speed multiplier")
    parser.add_argument("--sprint-speed-mult", type=float, default=1.5,
                        help="Max speed multiplier when sprinting")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from an existing run directory")

    args = parser.parse_args()

    config = SACZooTrainConfig(
        num_envs=args.num_envs,
        total_timesteps=args.timesteps,
        hidden_dim=args.hidden_dim,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        gamma=args.gamma,
        tau=args.tau,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        updates_per_step=args.updates_per_step,
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

    trainer = SACZooTrainer(config, env_config=env_config)

    if args.resume:
        trainer.resume_from(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()
