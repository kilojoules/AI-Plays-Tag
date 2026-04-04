"""
Wrapper that makes PettingZoo MPE simple_tag look like VecTagEnv.

Maps the PettingZoo parallel API to our {seeker, hider} dict interface
so the same SAC trainer and intrinsic rewards work on both environments.
"""
from __future__ import annotations

import numpy as np

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class VecMPETag:
    """Vectorized wrapper around multiple PettingZoo simple_tag instances.

    Presents the same interface as VecTagEnv:
        obs = env.reset()          -> {seeker: [E, obs_dim], hider: [E, obs_dim]}
        obs, rew, done, info = env.step(actions)
        obs = env.auto_reset()
    """

    def __init__(self, num_envs: int = 64, num_obstacles: int = 2,
                 max_cycles: int = 200):
        from pettingzoo.mpe import simple_tag_v3

        self.num_envs = num_envs
        self.max_cycles = max_cycles
        self._envs = []
        for _ in range(num_envs):
            env = simple_tag_v3.parallel_env(
                num_good=1, num_adversaries=1,
                num_obstacles=num_obstacles,
                max_cycles=max_cycles,
                continuous_actions=True,
            )
            self._envs.append(env)

        # Agent name mapping: adversary_0 = seeker, agent_0 = hider
        self._seeker_name = "adversary_0"
        self._hider_name = "agent_0"

        # Infer dimensions from first env
        test_obs, _ = self._envs[0].reset()
        self._seeker_obs_dim = test_obs[self._seeker_name].shape[0]
        self._hider_obs_dim = test_obs[self._hider_name].shape[0]
        # Pad to common obs dim (max of the two)
        self.obs_dim = max(self._seeker_obs_dim, self._hider_obs_dim)
        # MPE uses 5D actions
        self._mpe_act_dim = 5
        # We expose 2D actions (x, y forces) and map internally
        self.act_dim = 2

        self.dones = np.zeros(num_envs, dtype=bool)
        self._step_counts = np.zeros(num_envs, dtype=np.int32)

    def _pad_obs(self, obs: np.ndarray, target_dim: int) -> np.ndarray:
        """Pad observation to target dimension with zeros."""
        if obs.shape[-1] == target_dim:
            return obs
        pad_width = target_dim - obs.shape[-1]
        if obs.ndim == 1:
            return np.concatenate([obs, np.zeros(pad_width, dtype=np.float32)])
        return np.concatenate([obs, np.zeros((obs.shape[0], pad_width), dtype=np.float32)], axis=1)

    def _map_actions(self, actions_2d: np.ndarray) -> np.ndarray:
        """Map 2D continuous actions [-1,1] to MPE 5D format [0,1].

        MPE actions: [no_action, move_left, move_right, move_down, move_up]
        We map (ax, ay) -> proportional forces in each direction.
        """
        batch = actions_2d.shape[0]
        mpe_acts = np.zeros((batch, self._mpe_act_dim), dtype=np.float32)

        ax = actions_2d[:, 0]  # [-1, 1]
        ay = actions_2d[:, 1]  # [-1, 1]

        # Map x: negative = left (idx 1), positive = right (idx 2)
        mpe_acts[:, 1] = np.clip(-ax, 0, 1)  # left
        mpe_acts[:, 2] = np.clip(ax, 0, 1)   # right

        # Map y: negative = down (idx 3), positive = up (idx 4)
        mpe_acts[:, 3] = np.clip(-ay, 0, 1)  # down
        mpe_acts[:, 4] = np.clip(ay, 0, 1)   # up

        return mpe_acts

    def reset(self, env_ids: np.ndarray = None) -> dict:
        """Reset all or specified environments."""
        if env_ids is None:
            env_ids = np.arange(self.num_envs)

        all_seeker_obs = np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)
        all_hider_obs = np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)

        for i in env_ids:
            obs, _ = self._envs[i].reset()
            all_seeker_obs[i] = self._pad_obs(obs[self._seeker_name], self.obs_dim)
            all_hider_obs[i] = self._pad_obs(obs[self._hider_name], self.obs_dim)
            self.dones[i] = False
            self._step_counts[i] = 0

        return {'seeker': all_seeker_obs, 'hider': all_hider_obs}

    def step(self, actions: dict) -> tuple:
        """Step all environments."""
        seeker_acts_2d = actions['seeker']  # [E, 2]
        hider_acts_2d = actions['hider']    # [E, 2]

        seeker_mpe = self._map_actions(seeker_acts_2d)
        hider_mpe = self._map_actions(hider_acts_2d)

        all_seeker_obs = np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)
        all_hider_obs = np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)
        seeker_rewards = np.zeros(self.num_envs, dtype=np.float32)
        hider_rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        tagged = np.zeros(self.num_envs, dtype=bool)

        for i in range(self.num_envs):
            if self.dones[i]:
                continue

            mpe_actions = {
                self._seeker_name: seeker_mpe[i],
                self._hider_name: hider_mpe[i],
            }

            obs, rewards, terms, truncs, infos = self._envs[i].step(mpe_actions)

            self._step_counts[i] += 1

            # Check if any agent is done
            any_done = any(terms.values()) or any(truncs.values())

            if obs:  # obs can be empty if env terminates
                all_seeker_obs[i] = self._pad_obs(
                    obs.get(self._seeker_name, np.zeros(self._seeker_obs_dim, dtype=np.float32)),
                    self.obs_dim)
                all_hider_obs[i] = self._pad_obs(
                    obs.get(self._hider_name, np.zeros(self._hider_obs_dim, dtype=np.float32)),
                    self.obs_dim)

            seeker_rewards[i] = rewards.get(self._seeker_name, 0.0)
            hider_rewards[i] = -seeker_rewards[i]  # Zero-sum

            # Detect tag: seeker gets positive reward when catching
            if seeker_rewards[i] > 5.0:
                tagged[i] = True

            dones[i] = any_done

        self.dones = dones.copy()

        # Compute behavioral metrics (approximate — MPE doesn't expose wall dist)
        # Use hider obs to extract position info
        # In simple_tag 1v1: hider obs = [self_vel(2), self_pos(2), landmark_pos(2*n_obstacles), other_pos(2)]
        hider_pos = all_hider_obs[:, 2:4]  # self position
        hider_speed = np.linalg.norm(all_hider_obs[:, 0:2], axis=1)  # self velocity norm

        # MPE arena is roughly [-1, 1] but unbounded. Approximate "wall" as |pos| > 0.8
        near_wall = (np.abs(hider_pos) > 0.8).any(axis=1)
        in_corner = (np.abs(hider_pos[:, 0]) > 0.8) & (np.abs(hider_pos[:, 1]) > 0.8)

        infos = {
            'tagged': tagged,
            'timed_out': dones & ~tagged,
            'hider_wall_dist_mean': float((1.0 - np.abs(hider_pos).max(axis=1)).mean()),
            'hider_near_wall_frac': float(near_wall.mean()),
            'hider_corner_frac': float(in_corner.mean()),
            'hider_speed_mean': float(hider_speed.mean()),
            'hider_wall_speed_mean': float(hider_speed[near_wall].mean()) if near_wall.any() else 0.0,
            'seeker_speed_mean': float(np.linalg.norm(all_seeker_obs[:, 0:2], axis=1).mean()),
        }

        return (
            {'seeker': all_seeker_obs, 'hider': all_hider_obs},
            {'seeker': seeker_rewards, 'hider': hider_rewards},
            dones,
            infos,
        )

    def auto_reset(self) -> dict:
        """Reset finished environments and return current observations."""
        done_ids = np.where(self.dones)[0]
        if len(done_ids) > 0:
            obs = self.reset(done_ids)
            # Build full obs from all envs
            all_obs = {'seeker': np.zeros((self.num_envs, self.obs_dim), dtype=np.float32),
                       'hider': np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)}
            for i in range(self.num_envs):
                if i in done_ids:
                    all_obs['seeker'][i] = obs['seeker'][i]
                    all_obs['hider'][i] = obs['hider'][i]
                else:
                    # Get current obs from last step (stored in env state)
                    try:
                        env_obs = {a: self._envs[i].observe(a) for a in self._envs[i].agents}
                        all_obs['seeker'][i] = self._pad_obs(env_obs[self._seeker_name], self.obs_dim)
                        all_obs['hider'][i] = self._pad_obs(env_obs[self._hider_name], self.obs_dim)
                    except Exception:
                        pass  # Keep zeros for terminated envs
            return all_obs
        else:
            # No resets needed — get obs from envs
            all_obs = {'seeker': np.zeros((self.num_envs, self.obs_dim), dtype=np.float32),
                       'hider': np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)}
            for i in range(self.num_envs):
                try:
                    env_obs = {a: self._envs[i].observe(a) for a in self._envs[i].agents}
                    all_obs['seeker'][i] = self._pad_obs(env_obs[self._seeker_name], self.obs_dim)
                    all_obs['hider'][i] = self._pad_obs(env_obs[self._hider_name], self.obs_dim)
                except Exception:
                    pass
            return all_obs
