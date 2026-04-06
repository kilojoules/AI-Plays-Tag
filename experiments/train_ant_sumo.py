#!/usr/bin/env python3
"""
SAC self-play trainer for Ant Sumo.

Tests whether entropy bootstrapping and actor feature conditioning
transfer from 2D tag to 3D MuJoCo competitive wrestling.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sac import SACConfig, SACAgent, ReplayBuffer
from ant_sumo import VecAntSumo


class AntSumoTrainer:

    def __init__(self, num_envs, total_timesteps, init_alpha=0.2,
                 fixed_alpha=None, buffer_size=200_000, batch_size=256,
                 warmup_steps=10_000, updates_per_step=2, lr=3e-4,
                 log_interval=2000, save_interval=50_000, output_dir="experiments/results/ant_sumo"):
        self.num_envs = num_envs
        self.total_timesteps = total_timesteps
        self.log_interval = log_interval
        self.save_interval = save_interval

        self.env = VecAntSumo(num_envs=num_envs)
        obs_dim = self.env.obs_dim
        act_dim = self.env.act_dim

        sac_cfg = SACConfig(obs_dim=obs_dim, act_dim=act_dim,
                            buffer_size=buffer_size, batch_size=batch_size,
                            warmup_steps=warmup_steps, actor_lr=lr,
                            critic_lr=lr, alpha_lr=lr,
                            init_alpha=init_alpha, fixed_alpha=fixed_alpha)

        self.agents = {'seeker': SACAgent(sac_cfg), 'hider': SACAgent(sac_cfg)}
        self.buffers = {'seeker': ReplayBuffer(obs_dim, act_dim, buffer_size),
                        'hider': ReplayBuffer(obs_dim, act_dim, buffer_size)}
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.updates_per_step = updates_per_step

        self._ep_rewards = {'seeker': np.zeros(num_envs), 'hider': np.zeros(num_envs)}
        self._ep_lengths = np.zeros(num_envs, dtype=np.int32)
        self._window = {'s_rew': [], 'h_rew': [], 'lens': [], 's_wins': 0, 'h_wins': 0}
        self._total_episodes = 0

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(output_dir, run_id)
        os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)
        self.metrics_path = os.path.join(self.output_dir, "metrics.csv")
        with open(self.metrics_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "timesteps", "episodes", "seeker_reward_mean", "hider_reward_mean",
                "episode_length_mean", "seeker_win_rate",
                "seeker_alpha", "hider_alpha", "fps",
            ])

    def train(self):
        print(f"Ant Sumo SAC: {self.total_timesteps} steps, {self.num_envs} envs")
        print(f"  init_alpha={self.agents['seeker'].alpha:.3f}")
        print(f"  Output: {self.output_dir}\n")

        obs = self.env.reset()
        timesteps = 0
        start_time = time.time()
        train_infos = {}
        last_log = 0
        last_save = 0

        while timesteps < self.total_timesteps:
            is_warmup = timesteps < self.warmup_steps
            acts = {}
            for role in ['seeker', 'hider']:
                if is_warmup:
                    acts[role] = np.random.uniform(-1, 1, (self.num_envs, self.env.act_dim)).astype(np.float32)
                else:
                    with torch.no_grad():
                        x = torch.as_tensor(obs[role], dtype=torch.float32)
                        a, _ = self.agents[role].actor.sample(x)
                        acts[role] = a.cpu().numpy()

            next_obs, rewards, dones, infos = self.env.step(acts)
            timesteps += self.num_envs

            for role in ['seeker', 'hider']:
                self.buffers[role].add_batch(obs[role], acts[role], rewards[role],
                                            next_obs[role], dones.astype(np.float32))
                self._ep_rewards[role] += rewards[role]
            self._ep_lengths += 1

            for eid in np.where(dones)[0]:
                self._window['s_rew'].append(self._ep_rewards['seeker'][eid])
                self._window['h_rew'].append(self._ep_rewards['hider'][eid])
                self._ep_rewards['seeker'][eid] = 0
                self._ep_rewards['hider'][eid] = 0
                self._window['lens'].append(self._ep_lengths[eid])
                self._ep_lengths[eid] = 0
                self._total_episodes += 1
                if infos['tagged'][eid]:
                    self._window['s_wins'] += 1
                else:
                    self._window['h_wins'] += 1

            obs = self.env.auto_reset()

            if not is_warmup:
                for _ in range(self.updates_per_step):
                    for role in ['seeker', 'hider']:
                        if self.buffers[role].size >= self.batch_size:
                            batch = self.buffers[role].sample(self.batch_size)
                            train_infos[role] = self.agents[role].update(batch)

            if timesteps - last_log >= self.log_interval:
                self._log(timesteps, train_infos, start_time)
                last_log = timesteps

            if timesteps - last_save >= self.save_interval:
                for role in ['seeker', 'hider']:
                    self.agents[role].save_policy(
                        os.path.join(self.output_dir, "checkpoints",
                                     f"{role}_{timesteps:08d}.pt"))
                last_save = timesteps

        self._log(timesteps, train_infos, start_time)
        for role in ['seeker', 'hider']:
            self.agents[role].save_policy(
                os.path.join(self.output_dir, f"policy_{role}_final.pt"))
        elapsed = time.time() - start_time
        print(f"\nDone! {timesteps} steps in {elapsed:.1f}s ({timesteps/elapsed:.0f} FPS)")

    def _log(self, timesteps, train_infos, start_time):
        w = self._window
        fps = timesteps / (time.time() - start_time) if time.time() > start_time else 0
        total = w['s_wins'] + w['h_wins']
        swr = w['s_wins'] / max(total, 1)
        s_info = train_infos.get('seeker', {})
        h_info = train_infos.get('hider', {})

        row = [timesteps, self._total_episodes,
               np.mean(w['s_rew']) if w['s_rew'] else 0,
               np.mean(w['h_rew']) if w['h_rew'] else 0,
               np.mean(w['lens']) if w['lens'] else 0, swr,
               s_info.get('alpha', 0), h_info.get('alpha', 0), fps]

        with open(self.metrics_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

        print(f"Steps: {timesteps:8d} | Eps: {self._total_episodes:5d} | "
              f"SWR: {swr:.0%} | EpLen: {np.mean(w['lens']) if w['lens'] else 0:.0f} | "
              f"α: {s_info.get('alpha', 0):.3f}/{h_info.get('alpha', 0):.3f} | "
              f"FPS: {fps:.0f}")

        self._window = {'s_rew': [], 'h_rew': [], 'lens': [], 's_wins': 0, 'h_wins': 0}


def main():
    parser = argparse.ArgumentParser(description="Ant Sumo SAC self-play")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--init-alpha", type=float, default=0.2)
    parser.add_argument("--fixed-alpha", type=float, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="experiments/results/ant_sumo")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    trainer = AntSumoTrainer(
        num_envs=args.num_envs, total_timesteps=args.timesteps,
        init_alpha=args.init_alpha, fixed_alpha=args.fixed_alpha,
        lr=args.lr, output_dir=args.output_dir)
    trainer.train()


if __name__ == "__main__":
    main()
