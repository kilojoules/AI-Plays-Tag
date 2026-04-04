#!/usr/bin/env python3
"""
Train SAC agents on PettingZoo MPE simple_tag with optional
KL responsiveness intrinsic reward.

Uses the same SAC agents and intrinsic rewards as the tag env,
but with the MPE simple_tag environment via VecMPETag wrapper.
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
from intrinsic_rewards import ResponsivenessTracker
from mpe_wrapper import VecMPETag


class MPETagTrainer:
    """SAC self-play trainer for MPE simple_tag."""

    def __init__(self, num_envs: int, total_timesteps: int,
                 buffer_size: int = 200_000, batch_size: int = 256,
                 warmup_steps: int = 5_000, updates_per_step: int = 2,
                 lr: float = 3e-4,
                 responsiveness_method: str = 'none',
                 responsiveness_scale: float = 0.1,
                 log_interval: int = 2000,
                 save_interval: int = 50_000,
                 output_dir: str = "experiments/results/mpe_tag"):
        self.num_envs = num_envs
        self.total_timesteps = total_timesteps
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.updates_per_step = updates_per_step
        self.log_interval = log_interval
        self.save_interval = save_interval

        self.env = VecMPETag(num_envs=num_envs)
        obs_dim = self.env.obs_dim
        act_dim = self.env.act_dim

        sac_cfg = SACConfig(obs_dim=obs_dim, act_dim=act_dim,
                            buffer_size=buffer_size, batch_size=batch_size,
                            warmup_steps=warmup_steps, actor_lr=lr,
                            critic_lr=lr, alpha_lr=lr)

        self.agents = {
            'seeker': SACAgent(sac_cfg),
            'hider': SACAgent(sac_cfg),
        }
        self.buffers = {
            'seeker': ReplayBuffer(obs_dim, act_dim, buffer_size),
            'hider': ReplayBuffer(obs_dim, act_dim, buffer_size),
        }

        # Responsiveness tracker
        self._responsiveness_scale = responsiveness_scale
        if responsiveness_method != 'none':
            self._resp_tracker = ResponsivenessTracker(
                num_envs=num_envs, obs_dim=obs_dim,
                method=responsiveness_method,
                # MPE obs: hider sees opponent at indices 6+2*n_obstacles
                # With 2 obstacles: other_pos at indices [8:10]
                # But relative pos isn't directly available — use raw obs
                near_threshold=0.3,  # MPE arena is smaller
            )
            # Override the opponent position indices for MPE
            # Hider obs: [vel(2), pos(2), landmark1(2), landmark2(2), adversary_rel(2)]
            # The adversary relative position is at [8:10] for 2 obstacles
            self._resp_tracker._opp_rel_pos_idx = slice(8, 10)
        else:
            self._resp_tracker = None

        # Episode tracking
        self._ep_rewards = {
            'seeker': np.zeros(num_envs, dtype=np.float32),
            'hider': np.zeros(num_envs, dtype=np.float32),
        }
        self._ep_lengths = np.zeros(num_envs, dtype=np.int32)
        self._window_ep_rewards = {'seeker': [], 'hider': []}
        self._window_ep_lengths = []
        self._window_seeker_wins = 0
        self._window_hider_wins = 0
        self._total_episodes = 0
        self._window_corner = []
        self._window_speed = []
        self._window_te = []
        self._window_kl = []

        # Output
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(output_dir, self.run_id)
        os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)
        self.metrics_path = os.path.join(self.output_dir, "metrics.csv")
        with open(self.metrics_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "timesteps", "episodes",
                "seeker_reward_mean", "hider_reward_mean",
                "episode_length_mean", "seeker_win_rate",
                "seeker_alpha", "hider_alpha",
                "hider_corner_frac", "hider_speed",
                "fps", "hider_te", "hider_kl",
            ])

    def _act_batch(self, agent, obs, random=False):
        if random:
            return np.random.uniform(-1, 1, (obs.shape[0], self.env.act_dim)).astype(np.float32)
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32)
            actions, _ = agent.actor.sample(x)
            return actions.cpu().numpy()

    def train(self):
        print(f"MPE simple_tag SAC training: {self.total_timesteps} steps")
        print(f"  num_envs={self.num_envs}, obs_dim={self.env.obs_dim}, act_dim={self.env.act_dim}")
        print(f"  responsiveness={self._resp_tracker is not None}, scale={self._responsiveness_scale}")
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
                acts[role] = self._act_batch(self.agents[role], obs[role], random=is_warmup)

            next_obs, rewards, dones, infos = self.env.step(acts)
            timesteps += self.num_envs

            # Intrinsic responsiveness reward
            if self._resp_tracker is not None:
                intrinsic = self._resp_tracker.compute(
                    obs['hider'], acts['hider'], next_obs['hider'], dones)
                rewards['hider'] = rewards['hider'] + self._responsiveness_scale * intrinsic
                if timesteps % (self.log_interval * self.num_envs) < self.num_envs:
                    stats = self._resp_tracker.get_stats()
                    self._window_te.append(stats['te_mean'])
                    self._window_kl.append(stats['kl_mean'])

            # Store transitions
            for role in ['seeker', 'hider']:
                self.buffers[role].add_batch(
                    obs[role], acts[role], rewards[role],
                    next_obs[role], dones.astype(np.float32))

            # Track episodes
            for role in ['seeker', 'hider']:
                self._ep_rewards[role] += rewards[role]
            self._ep_lengths += 1

            self._window_corner.append(infos['hider_corner_frac'])
            self._window_speed.append(infos['hider_speed_mean'])

            for eid in np.where(dones)[0]:
                for role in ['seeker', 'hider']:
                    self._window_ep_rewards[role].append(self._ep_rewards[role][eid])
                    self._ep_rewards[role][eid] = 0.0
                self._window_ep_lengths.append(self._ep_lengths[eid])
                self._ep_lengths[eid] = 0
                self._total_episodes += 1
                if infos['tagged'][eid]:
                    self._window_seeker_wins += 1
                else:
                    self._window_hider_wins += 1

            obs = self.env.auto_reset()

            # Update
            if not is_warmup:
                for _ in range(self.updates_per_step):
                    for role in ['seeker', 'hider']:
                        if self.buffers[role].size >= self.batch_size:
                            batch = self.buffers[role].sample(self.batch_size)
                            train_infos[role] = self.agents[role].update(batch)

            # Log
            if timesteps - last_log >= self.log_interval:
                self._log(timesteps, train_infos, start_time)
                last_log = timesteps

            # Save
            if timesteps - last_save >= self.save_interval:
                for role in ['seeker', 'hider']:
                    path = os.path.join(self.output_dir, "checkpoints",
                                        f"{role}_{timesteps:08d}.pt")
                    self.agents[role].save_policy(path)
                last_save = timesteps

        # Final save
        self._log(timesteps, train_infos, start_time)
        for role in ['seeker', 'hider']:
            self.agents[role].save_policy(
                os.path.join(self.output_dir, f"policy_{role}_final.pt"))

        elapsed = time.time() - start_time
        print(f"\nDone! {timesteps} steps in {elapsed:.1f}s ({timesteps/elapsed:.0f} FPS)")

    def _log(self, timesteps, train_infos, start_time):
        elapsed = time.time() - start_time
        fps = timesteps / elapsed if elapsed > 0 else 0

        s_rews = self._window_ep_rewards['seeker'] or [0]
        h_rews = self._window_ep_rewards['hider'] or [0]
        ep_lens = self._window_ep_lengths or [0]
        total = self._window_seeker_wins + self._window_hider_wins
        s_wr = self._window_seeker_wins / max(total, 1)

        s_info = train_infos.get('seeker', {})
        h_info = train_infos.get('hider', {})

        row = [
            timesteps, self._total_episodes,
            np.mean(s_rews), np.mean(h_rews),
            np.mean(ep_lens), s_wr,
            s_info.get('alpha', 0), h_info.get('alpha', 0),
            np.mean(self._window_corner) if self._window_corner else 0,
            np.mean(self._window_speed) if self._window_speed else 0,
            fps,
            np.mean(self._window_te) if self._window_te else 0,
            np.mean(self._window_kl) if self._window_kl else 0,
        ]

        with open(self.metrics_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

        print(f"Steps: {timesteps:8d} | Eps: {self._total_episodes:5d} | "
              f"SWR: {s_wr:.0%} | EpLen: {np.mean(ep_lens):.0f} | "
              f"S_R: {np.mean(s_rews):+.2f} | H_R: {np.mean(h_rews):+.2f} | "
              f"Corner: {np.mean(self._window_corner) if self._window_corner else 0:.2f} | "
              f"FPS: {fps:.0f}")

        # Reset window
        self._window_ep_rewards = {'seeker': [], 'hider': []}
        self._window_ep_lengths = []
        self._window_seeker_wins = 0
        self._window_hider_wins = 0
        self._window_corner = []
        self._window_speed = []
        self._window_te = []
        self._window_kl = []


def main():
    parser = argparse.ArgumentParser(description="MPE simple_tag SAC training")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--buffer-size", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=5_000)
    parser.add_argument("--updates-per-step", type=int, default=2)
    parser.add_argument("--responsiveness", type=str, default="none",
                        choices=["none", "te", "kl", "both"])
    parser.add_argument("--responsiveness-scale", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=2000)
    parser.add_argument("--save-interval", type=int, default=50_000)
    parser.add_argument("--output-dir", type=str,
                        default="experiments/results/mpe_tag")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    trainer = MPETagTrainer(
        num_envs=args.num_envs,
        total_timesteps=args.timesteps,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        updates_per_step=args.updates_per_step,
        lr=args.lr,
        responsiveness_method=args.responsiveness,
        responsiveness_scale=args.responsiveness_scale,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        output_dir=args.output_dir,
    )
    trainer.train()


if __name__ == "__main__":
    main()
