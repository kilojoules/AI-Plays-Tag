#!/usr/bin/env python3
"""
Self-play SAC trainer for tag.

Off-policy alternative to train_selfplay.py (PPO).
Both agents train simultaneously with separate replay buffers.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from tag_env import VecTagEnv, TagEnvConfig
from sac import SACConfig, SACAgent, ReplayBuffer


class SelfPlaySACTrainer:
    """Self-play SAC trainer. Both agents share the same env steps."""

    def __init__(self, num_envs: int, total_timesteps: int,
                 env_config: Optional[TagEnvConfig] = None,
                 buffer_size: int = 500_000, batch_size: int = 256,
                 warmup_steps: int = 10_000, updates_per_step: int = 1,
                 lr: float = 3e-4, init_alpha: float = 0.2,
                 fixed_alpha: float = None,
                 log_interval: int = 1000,
                 save_interval: int = 50_000, output_dir: str = "experiments/results/selfplay_sac"):
        self.num_envs = num_envs
        self.total_timesteps = total_timesteps
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.updates_per_step = updates_per_step
        self.log_interval = log_interval
        self.save_interval = save_interval

        self.env = VecTagEnv(num_envs=num_envs, config=env_config)
        obs_dim = self.env.obs_dim
        act_dim = self.env.act_dim

        sac_cfg = SACConfig(obs_dim=obs_dim, act_dim=act_dim,
                            buffer_size=buffer_size, batch_size=batch_size,
                            warmup_steps=warmup_steps, actor_lr=lr,
                            critic_lr=lr, alpha_lr=lr,
                            init_alpha=init_alpha,
                            fixed_alpha=fixed_alpha)

        self.agents = {
            'seeker': SACAgent(sac_cfg),
            'hider': SACAgent(sac_cfg),
        }
        self.buffers = {
            'seeker': ReplayBuffer(obs_dim, act_dim, buffer_size),
            'hider': ReplayBuffer(obs_dim, act_dim, buffer_size),
        }

        # Episode tracking
        self._ep_rewards = {
            'seeker': np.zeros(num_envs, dtype=np.float32),
            'hider': np.zeros(num_envs, dtype=np.float32),
        }
        self._ep_lengths = np.zeros(num_envs, dtype=np.int32)

        # Window metrics (reset each log interval)
        self._window_ep_rewards = {'seeker': [], 'hider': []}
        self._window_ep_lengths = []
        self._window_seeker_wins = 0
        self._window_hider_wins = 0
        self._total_episodes = 0

        # Behavioral metrics window
        self._window_hider_wall_dist = []
        self._window_hider_near_wall = []
        self._window_hider_speed = []
        self._window_seeker_speed = []

        # Output
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(output_dir, self.run_id)
        os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)
        self.metrics_path = os.path.join(self.output_dir, "metrics.csv")
        self._init_csv()

    def _init_csv(self):
        with open(self.metrics_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "timesteps", "episodes",
                "seeker_reward_mean", "seeker_reward_std",
                "hider_reward_mean", "hider_reward_std",
                "episode_length_mean", "seeker_win_rate", "hider_win_rate",
                "seeker_critic_loss", "seeker_actor_loss", "seeker_alpha",
                "hider_critic_loss", "hider_actor_loss", "hider_alpha",
                "hider_wall_dist", "hider_near_wall_frac",
                "hider_speed", "seeker_speed",
                "fps",
            ])

    def _act_batch(self, agent: SACAgent, obs: np.ndarray, random: bool = False) -> np.ndarray:
        """Batch actions from SAC agent."""
        if random:
            return np.random.uniform(-1.0, 1.0, (obs.shape[0], self.env.act_dim)).astype(np.float32)
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32)
            actions, _ = agent.actor.sample(x)
            return actions.cpu().numpy()

    def _log_metrics(self, timesteps, train_infos, start_time):
        elapsed = time.time() - start_time
        fps = timesteps / elapsed if elapsed > 0 else 0

        s_rews = self._window_ep_rewards['seeker'] or [0]
        h_rews = self._window_ep_rewards['hider'] or [0]
        ep_lens = self._window_ep_lengths or [0]
        total = self._window_seeker_wins + self._window_hider_wins
        s_wr = self._window_seeker_wins / max(total, 1)
        h_wr = self._window_hider_wins / max(total, 1)

        s_info = train_infos.get('seeker', {})
        h_info = train_infos.get('hider', {})

        row = [
            timesteps, self._total_episodes,
            np.mean(s_rews), np.std(s_rews),
            np.mean(h_rews), np.std(h_rews),
            np.mean(ep_lens), s_wr, h_wr,
            s_info.get('critic_loss', 0), s_info.get('actor_loss', 0), s_info.get('alpha', 0),
            h_info.get('critic_loss', 0), h_info.get('actor_loss', 0), h_info.get('alpha', 0),
            np.mean(self._window_hider_wall_dist) if self._window_hider_wall_dist else 0,
            np.mean(self._window_hider_near_wall) if self._window_hider_near_wall else 0,
            np.mean(self._window_hider_speed) if self._window_hider_speed else 0,
            np.mean(self._window_seeker_speed) if self._window_seeker_speed else 0,
            fps,
        ]

        with open(self.metrics_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

        print(f"Steps: {timesteps:8d} | Eps: {self._total_episodes:5d} | "
              f"SWR: {s_wr:.0%} | EpLen: {np.mean(ep_lens):.0f} | "
              f"S_R: {np.mean(s_rews):+.2f} | H_R: {np.mean(h_rews):+.2f} | "
              f"α_s: {s_info.get('alpha', 0):.3f} α_h: {h_info.get('alpha', 0):.3f} | "
              f"FPS: {fps:.0f}")

        # Reset window
        self._window_ep_rewards = {'seeker': [], 'hider': []}
        self._window_ep_lengths = []
        self._window_seeker_wins = 0
        self._window_hider_wins = 0
        self._window_hider_wall_dist = []
        self._window_hider_near_wall = []
        self._window_hider_speed = []
        self._window_seeker_speed = []

    def save_checkpoint(self, timesteps):
        for role in ['seeker', 'hider']:
            path = os.path.join(self.output_dir, "checkpoints",
                                f"{role}_{timesteps:08d}.pt")
            self.agents[role].save_policy(path)

    def train(self):
        print(f"Self-play SAC training: {self.total_timesteps} timesteps")
        print(f"  num_envs={self.num_envs}, buffer={self.buffers['seeker'].max_size}, "
              f"batch={self.batch_size}, warmup={self.warmup_steps}")
        print(f"  Output: {self.output_dir}\n")

        obs = self.env.reset()
        timesteps = 0
        start_time = time.time()
        train_infos = {}
        last_log = 0
        last_save = 0

        while timesteps < self.total_timesteps:
            is_warmup = timesteps < self.warmup_steps

            # Act
            acts = {}
            for role in ['seeker', 'hider']:
                acts[role] = self._act_batch(self.agents[role], obs[role],
                                             random=is_warmup)

            next_obs, rewards, dones, infos = self.env.step(acts)
            timesteps += self.num_envs

            # Store transitions in replay buffers
            for role in ['seeker', 'hider']:
                self.buffers[role].add_batch(
                    obs[role], acts[role], rewards[role],
                    next_obs[role], dones.astype(np.float32))

            # Track episodes
            for role in ['seeker', 'hider']:
                self._ep_rewards[role] += rewards[role]
            self._ep_lengths += 1

            # Behavioral metrics
            self._window_hider_wall_dist.append(infos['hider_wall_dist_mean'])
            self._window_hider_near_wall.append(infos['hider_near_wall_frac'])
            self._window_hider_speed.append(infos['hider_speed_mean'])
            self._window_seeker_speed.append(infos['seeker_speed_mean'])

            for eid in np.where(dones)[0]:
                for role in ['seeker', 'hider']:
                    self._window_ep_rewards[role].append(
                        self._ep_rewards[role][eid])
                    self._ep_rewards[role][eid] = 0.0
                self._window_ep_lengths.append(self._ep_lengths[eid])
                self._ep_lengths[eid] = 0
                self._total_episodes += 1
                if infos['tagged'][eid]:
                    self._window_seeker_wins += 1
                else:
                    self._window_hider_wins += 1

            obs = self.env.auto_reset()

            # Update agents (after warmup, every step)
            if not is_warmup:
                for _ in range(self.updates_per_step):
                    for role in ['seeker', 'hider']:
                        if self.buffers[role].size >= self.batch_size:
                            batch = self.buffers[role].sample(self.batch_size)
                            train_infos[role] = self.agents[role].update(batch)

            # Logging
            if timesteps - last_log >= self.log_interval:
                self._log_metrics(timesteps, train_infos, start_time)
                last_log = timesteps

            # Checkpointing
            if timesteps - last_save >= self.save_interval:
                self.save_checkpoint(timesteps)
                last_save = timesteps

        # Final
        self._log_metrics(timesteps, train_infos, start_time)
        self.save_checkpoint(timesteps)
        for role in ['seeker', 'hider']:
            path = os.path.join(self.output_dir, f"policy_{role}_final.pt")
            self.agents[role].save_policy(path)

        elapsed = time.time() - start_time
        print(f"\nDone! {timesteps} steps in {elapsed:.1f}s "
              f"({timesteps/elapsed:.0f} FPS)")
        print(f"Total episodes: {self._total_episodes}")


def main():
    parser = argparse.ArgumentParser(description="Self-play SAC for tag")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=10_000)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--layout", type=str, default="four_corners",
                        choices=["empty", "four_corners", "central_cross", "playground"])
    parser.add_argument("--hider-speed-mult", type=float, default=1.0)
    parser.add_argument("--seeker-time-penalty", type=float, default=-0.005)
    parser.add_argument("--distance-reward-scale", type=float, default=0.14)
    parser.add_argument("--runner-survival-bonus", type=float, default=0.01)
    parser.add_argument("--hider-dist-reward", type=float, default=0.0)
    parser.add_argument("--hider-abs-dist-reward", type=float, default=0.1)
    parser.add_argument("--hider-wall-prox-penalty", type=float, default=0.0)
    parser.add_argument("--hider-min-speed-reward", type=float, default=0.0)
    parser.add_argument("--seeker-escalating-urgency", action="store_true")
    parser.add_argument("--area-coverage-bonus", type=float, default=0.0)
    parser.add_argument("--fixed-alpha", type=float, default=None,
                        help="Fix entropy coefficient (disable auto-tuning). Use 0.0 for no-entropy ablation.")
    parser.add_argument("--init-alpha", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=5000)
    parser.add_argument("--save-interval", type=int, default=100_000)
    parser.add_argument("--output-dir", type=str,
                        default="experiments/results/selfplay_sac")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    env_config = TagEnvConfig(
        layout=args.layout,
        hider_speed_mult=args.hider_speed_mult,
        seeker_time_penalty=args.seeker_time_penalty,
        distance_reward_scale=args.distance_reward_scale,
        runner_survival_bonus=args.runner_survival_bonus,
        hider_dist_reward_scale=args.hider_dist_reward,
        hider_abs_dist_reward_scale=args.hider_abs_dist_reward,
        hider_wall_prox_penalty=args.hider_wall_prox_penalty,
        hider_min_speed_reward=args.hider_min_speed_reward,
        seeker_escalating_urgency=args.seeker_escalating_urgency,
        area_coverage_bonus=args.area_coverage_bonus,
    )

    trainer = SelfPlaySACTrainer(
        num_envs=args.num_envs,
        total_timesteps=args.timesteps,
        env_config=env_config,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        updates_per_step=args.updates_per_step,
        lr=args.lr,
        init_alpha=args.init_alpha,
        fixed_alpha=args.fixed_alpha,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        output_dir=args.output_dir,
    )
    trainer.train()


if __name__ == "__main__":
    main()
