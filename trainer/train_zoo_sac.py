#!/usr/bin/env python3
"""
Zoo-based SAC training for tag agents.

Combines SAC (off-policy) with zoo opponent sampling:
- A% of the time: play against a zoo checkpoint
- (100-A)% of the time: play against latest opponent

Unlike PPO zoo training, SAC collects transitions into replay buffers
and updates from minibatches — no need for separate on-policy rollouts.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
except ImportError:
    print("PyTorch required. Install with: pip install torch", file=sys.stderr)
    sys.exit(1)

from tag_env import VecTagEnv, TagEnvConfig
from sac import SACConfig, SACAgent, SACActorNet, ReplayBuffer


class SACOpponentZoo:
    """Zoo of SAC actor checkpoints for opponent sampling."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256,
                 max_size: int = 50):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.max_size = max_size
        self.checkpoints: List[Dict[str, Any]] = []

    def add(self, agent: SACAgent, update: int):
        """Snapshot the actor network into the zoo."""
        state = {
            'actor': {k: v.clone() for k, v in agent.actor.state_dict().items()},
            'update': update,
        }
        self.checkpoints.append(state)
        if len(self.checkpoints) > self.max_size:
            self.checkpoints.pop(0)

    def sample(self) -> Tuple[SACActorNet, int]:
        """Uniformly sample a past actor from the zoo."""
        if not self.checkpoints:
            raise ValueError("Zoo is empty!")
        idx = random.randrange(len(self.checkpoints))
        state = self.checkpoints[idx]
        actor = SACActorNet(self.obs_dim, self.act_dim, self.hidden_dim)
        actor.load_state_dict(state['actor'])
        return actor, idx

    def __len__(self):
        return len(self.checkpoints)


class ZooSACTrainer:
    """SAC trainer with zoo-based opponent sampling."""

    def __init__(self, num_envs: int, total_timesteps: int,
                 env_config: Optional[TagEnvConfig] = None,
                 buffer_size: int = 500_000, batch_size: int = 256,
                 warmup_steps: int = 10_000, updates_per_step: int = 1,
                 lr: float = 3e-4, gamma: float = 0.99,
                 tau: float = 0.005, init_alpha: float = 0.2,
                 zoo_prob: float = 0.5,
                 zoo_update_interval: int = 50, zoo_max_size: int = 50,
                 opponent_resample_steps: int = 1280,
                 log_interval: int = 5000, save_interval: int = 50_000,
                 output_dir: str = "experiments/results/zoo_sac"):
        self.num_envs = num_envs
        self.total_timesteps = total_timesteps
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.updates_per_step = updates_per_step
        self.zoo_prob = zoo_prob
        self.zoo_update_interval = zoo_update_interval
        self.opponent_resample_steps = opponent_resample_steps
        self.log_interval = log_interval
        self.save_interval = save_interval

        self.env = VecTagEnv(num_envs=num_envs, config=env_config)
        obs_dim = self.env.obs_dim
        act_dim = self.env.act_dim

        sac_cfg = SACConfig(obs_dim=obs_dim, act_dim=act_dim,
                            buffer_size=buffer_size, batch_size=batch_size,
                            warmup_steps=warmup_steps, actor_lr=lr,
                            critic_lr=lr, alpha_lr=lr,
                            gamma=gamma, tau=tau, init_alpha=init_alpha)

        self.agents = {
            'seeker': SACAgent(sac_cfg),
            'hider': SACAgent(sac_cfg),
        }
        self.buffers = {
            'seeker': ReplayBuffer(obs_dim, act_dim, buffer_size),
            'hider': ReplayBuffer(obs_dim, act_dim, buffer_size),
        }

        self.hider_zoo = SACOpponentZoo(obs_dim, act_dim, max_size=zoo_max_size)
        self.seeker_zoo = SACOpponentZoo(obs_dim, act_dim, max_size=zoo_max_size)

        self._current_opp_actor = {
            'seeker': self.agents['seeker'].actor,
            'hider': self.agents['hider'].actor,
        }

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

        self.zoo_samples = {'seeker': 0, 'hider': 0}
        self.latest_samples = {'seeker': 0, 'hider': 0}

        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(output_dir, self.run_id)
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
                "seeker_critic_loss", "seeker_actor_loss", "seeker_alpha",
                "hider_critic_loss", "hider_actor_loss", "hider_alpha",
                "hider_zoo_size", "seeker_zoo_size", "zoo_sample_rate",
                "fps",
            ])

    def _resample_opponents(self):
        """Resample opponents from zoo or use latest."""
        for opp_role in ['seeker', 'hider']:
            zoo = self.seeker_zoo if opp_role == 'seeker' else self.hider_zoo
            if len(zoo) > 0 and random.random() < self.zoo_prob:
                actor, _ = zoo.sample()
                self._current_opp_actor[opp_role] = actor
                self.zoo_samples[opp_role] += 1
            else:
                self._current_opp_actor[opp_role] = self.agents[opp_role].actor
                self.latest_samples[opp_role] += 1

    def _act_batch(self, actor, obs: np.ndarray,
                   random_act: bool = False) -> np.ndarray:
        """Get actions from an actor network."""
        if random_act:
            return np.random.uniform(-1.0, 1.0,
                                     (obs.shape[0], self.env.act_dim)).astype(np.float32)
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32)
            actions, _ = actor.sample(x)
            return actions.cpu().numpy()

    def _log_metrics(self, update, timesteps, train_infos, start_time):
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

        total_samples = sum(self.zoo_samples.values()) + sum(self.latest_samples.values())
        zoo_rate = sum(self.zoo_samples.values()) / max(total_samples, 1)

        row = [
            update, timesteps, self._total_episodes,
            np.mean(s_rews), np.std(s_rews),
            np.mean(h_rews), np.std(h_rews),
            np.mean(ep_lens), s_wr, h_wr,
            s_info.get('critic_loss', 0), s_info.get('actor_loss', 0),
            s_info.get('alpha', 0),
            h_info.get('critic_loss', 0), h_info.get('actor_loss', 0),
            h_info.get('alpha', 0),
            len(self.hider_zoo), len(self.seeker_zoo), zoo_rate,
            fps,
        ]

        with open(self.metrics_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

        print(f"U{update:4d} | Steps: {timesteps:8d} | Eps: {self._total_episodes:5d} | "
              f"SWR: {s_wr:.0%} | EpLen: {np.mean(ep_lens):.0f} | "
              f"Zoo: H={len(self.hider_zoo)},S={len(self.seeker_zoo)} | "
              f"ZooRate: {zoo_rate:.0%} | FPS: {fps:.0f}")

        self._window_ep_rewards = {'seeker': [], 'hider': []}
        self._window_ep_lengths = []
        self._window_seeker_wins = 0
        self._window_hider_wins = 0

    def save_checkpoint(self, update):
        for role in ['seeker', 'hider']:
            path = os.path.join(self.output_dir, "checkpoints",
                                f"{role}_{update:05d}.pt")
            self.agents[role].save_policy(path)

    def save_final(self):
        for role in ['seeker', 'hider']:
            path = os.path.join(self.output_dir, f"policy_{role}_final.pt")
            self.agents[role].save_policy(path)

        metadata = {
            'run_id': self.run_id,
            'algorithm': 'sac',
            'total_episodes': self._total_episodes,
            'config': {
                'zoo_prob': self.zoo_prob,
                'zoo_update_interval': self.zoo_update_interval,
                'total_timesteps': self.total_timesteps,
            },
            'zoo_stats': {
                'hider_zoo_final_size': len(self.hider_zoo),
                'seeker_zoo_final_size': len(self.seeker_zoo),
                'zoo_samples': self.zoo_samples,
                'latest_samples': self.latest_samples,
            }
        }
        with open(os.path.join(self.output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    def train(self):
        print(f"Zoo SAC training: {self.total_timesteps} timesteps")
        print(f"  A (zoo prob): {self.zoo_prob:.2f}")
        print(f"  Zoo update every {self.zoo_update_interval} updates")
        print(f"  Opponent resample every {self.opponent_resample_steps} env steps")
        print(f"  Output: {self.output_dir}\n")

        obs = self.env.reset()
        timesteps = 0
        update = 0
        start_time = time.time()
        train_infos = {}
        last_log = 0
        last_save = 0
        steps_since_resample = 0

        self._resample_opponents()

        while timesteps < self.total_timesteps:
            is_warmup = timesteps < self.warmup_steps

            steps_since_resample += self.num_envs
            if steps_since_resample >= self.opponent_resample_steps:
                self._resample_opponents()
                steps_since_resample = 0

            acts = {}
            for role in ['seeker', 'hider']:
                if is_warmup:
                    acts[role] = np.random.uniform(
                        -1.0, 1.0,
                        (self.num_envs, self.env.act_dim)).astype(np.float32)
                else:
                    acts[role] = self._act_batch(self.agents[role].actor, obs[role])

            next_obs, rewards, dones, infos = self.env.step(acts)
            timesteps += self.num_envs

            for role in ['seeker', 'hider']:
                self.buffers[role].add_batch(
                    obs[role], acts[role], rewards[role],
                    next_obs[role], dones.astype(np.float32))

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
                self._total_episodes += 1
                if infos['tagged'][eid]:
                    self._window_seeker_wins += 1
                else:
                    self._window_hider_wins += 1

            obs = self.env.auto_reset()

            if not is_warmup:
                for _ in range(self.updates_per_step):
                    for role in ['seeker', 'hider']:
                        if self.buffers[role].size >= self.batch_size:
                            batch = self.buffers[role].sample(self.batch_size)
                            train_infos[role] = self.agents[role].update(batch)

                update += 1

                if update % self.zoo_update_interval == 0:
                    self.hider_zoo.add(self.agents['hider'], update)
                    self.seeker_zoo.add(self.agents['seeker'], update)

            if timesteps - last_log >= self.log_interval:
                self._log_metrics(update, timesteps, train_infos, start_time)
                last_log = timesteps

            if timesteps - last_save >= self.save_interval:
                self.save_checkpoint(update)
                last_save = timesteps

        self._log_metrics(update, timesteps, train_infos, start_time)
        self.save_checkpoint(update)
        self.save_final()

        elapsed = time.time() - start_time
        print(f"\nDone! {timesteps} steps in {elapsed:.1f}s "
              f"({timesteps/elapsed:.0f} FPS)")
        print(f"Total episodes: {self._total_episodes}")
        print(f"Zoo sizes: hider={len(self.hider_zoo)}, seeker={len(self.seeker_zoo)}")


def main():
    parser = argparse.ArgumentParser(description="Zoo-based SAC tag training")
    parser.add_argument("--timesteps", type=int, default=5_000_000)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=10_000)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.005,
                        help="Soft target update rate")
    parser.add_argument("--init-alpha", type=float, default=0.2,
                        help="Initial entropy temperature")

    parser.add_argument("--zoo-prob", "-A", type=float, default=0.5,
                        help="Probability of sampling from zoo (A parameter)")
    parser.add_argument("--zoo-interval", type=int, default=50,
                        help="Add to zoo every N updates")
    parser.add_argument("--zoo-max-size", type=int, default=50)
    parser.add_argument("--opponent-resample-steps", type=int, default=1280,
                        help="Resample opponent every N env steps")

    parser.add_argument("--layout", type=str, default="four_corners",
                        choices=["empty", "four_corners", "central_cross", "playground"])
    parser.add_argument("--hider-speed-mult", type=float, default=1.0)
    parser.add_argument("--enable-sprint", action="store_true")
    parser.add_argument("--sprint-speed-mult", type=float, default=1.5)

    parser.add_argument("--seeker-time-penalty", type=float, default=-0.005)
    parser.add_argument("--distance-reward-scale", type=float, default=0.14)
    parser.add_argument("--runner-survival-bonus", type=float, default=0.01)
    parser.add_argument("--hider-dist-reward", type=float, default=0.0)
    parser.add_argument("--hider-abs-dist-reward", type=float, default=0.1)
    parser.add_argument("--hider-wall-prox-penalty", type=float, default=0.0)
    parser.add_argument("--hider-min-speed-reward", type=float, default=0.0)
    parser.add_argument("--seeker-escalating-urgency", action="store_true")
    parser.add_argument("--area-coverage-bonus", type=float, default=0.0)

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=5000)
    parser.add_argument("--save-interval", type=int, default=100_000)
    parser.add_argument("--output-dir", type=str,
                        default="experiments/results/zoo_sac")

    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    env_config = TagEnvConfig(
        layout=args.layout,
        enable_sprint=args.enable_sprint,
        hider_speed_mult=args.hider_speed_mult,
        sprint_speed_mult=args.sprint_speed_mult,
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

    trainer = ZooSACTrainer(
        num_envs=args.num_envs,
        total_timesteps=args.timesteps,
        env_config=env_config,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        updates_per_step=args.updates_per_step,
        lr=args.lr,
        gamma=args.gamma,
        tau=args.tau,
        init_alpha=args.init_alpha,
        zoo_prob=args.zoo_prob,
        zoo_update_interval=args.zoo_interval,
        zoo_max_size=args.zoo_max_size,
        opponent_resample_steps=args.opponent_resample_steps,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        output_dir=args.output_dir,
    )
    trainer.train()


if __name__ == "__main__":
    main()
