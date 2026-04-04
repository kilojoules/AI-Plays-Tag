"""
Intrinsic reward signals for detecting and penalizing degenerate behavior
in competitive self-play.

Two approaches:
  1. Transfer Entropy (TE): measures causal influence of opponent state on
     agent actions. TE ≈ 0 means the agent ignores the opponent (degenerate).
  2. Conditional KL Divergence: compares the agent's action distribution when
     the opponent is near vs far. KL ≈ 0 means same behavior regardless of
     opponent proximity (degenerate).

Both replace hand-crafted anti-degenerate penalties (wall proximity, min speed,
coverage bonus) with a single domain-agnostic "responsiveness" signal.

Usage:
    from intrinsic_rewards import ResponsivenessTracker

    tracker = ResponsivenessTracker(num_envs=64, method='te', obs_dim=87)

    # In training loop:
    intrinsic = tracker.compute(obs_hider, acts_hider, next_obs_hider)
    rewards['hider'] += te_scale * intrinsic
"""
from __future__ import annotations

import numpy as np
from typing import Literal


class ResponsivenessTracker:
    """Tracks opponent-responsiveness and computes intrinsic reward signals.

    Maintains a rolling window of (opponent_state, agent_action) pairs per
    environment, and computes either Transfer Entropy or Conditional KL
    divergence as the intrinsic reward.
    """

    def __init__(
        self,
        num_envs: int,
        obs_dim: int = 87,
        method: Literal['te', 'kl', 'both'] = 'both',
        window_size: int = 200,
        num_bins: int = 5,
        near_threshold: float = 0.4,
    ):
        """
        Args:
            num_envs: Number of parallel environments.
            obs_dim: Observation dimension (for extracting opponent state).
            method: 'te' for transfer entropy, 'kl' for conditional KL, 'both'.
            window_size: Rolling window of transitions to estimate from.
            num_bins: Bins per dimension for histogram estimation.
            near_threshold: Normalized distance threshold for near/far split
                           in KL method (relative to arena_half).
        """
        self.num_envs = num_envs
        self.method = method
        self.window_size = window_size
        self.num_bins = num_bins
        self.near_threshold = near_threshold

        # Rolling buffers: store opponent relative pos (2D) and agent actions (2D)
        # We use the movement dimensions of action (first 2), ignoring sprint
        self._opp_pos_buf = np.zeros((num_envs, window_size, 2), dtype=np.float32)
        self._action_buf = np.zeros((num_envs, window_size, 2), dtype=np.float32)
        self._next_action_buf = np.zeros((num_envs, window_size, 2), dtype=np.float32)
        self._ptr = np.zeros(num_envs, dtype=np.int32)
        self._count = np.zeros(num_envs, dtype=np.int32)

        # Observation indices for extracting opponent relative position
        # In the hider's obs: [4:6] = (seeker_pos - hider_pos) / arena_half
        self._opp_rel_pos_idx = slice(4, 6)

    def reset(self, env_ids: np.ndarray = None):
        """Reset buffers for specified environments (e.g., on episode end)."""
        if env_ids is None:
            self._ptr[:] = 0
            self._count[:] = 0
        else:
            self._ptr[env_ids] = 0
            self._count[env_ids] = 0

    def _store(self, obs: np.ndarray, actions: np.ndarray,
               next_obs: np.ndarray):
        """Store a transition in the rolling buffer."""
        opp_pos = obs[:, self._opp_rel_pos_idx]      # [E, 2]
        act_2d = actions[:, :2]                        # [E, 2] (ignore sprint)

        for i in range(self.num_envs):
            idx = self._ptr[i]
            self._opp_pos_buf[i, idx] = opp_pos[i]
            self._action_buf[i, idx] = act_2d[i]
            self._ptr[i] = (idx + 1) % self.window_size
            self._count[i] = min(self._count[i] + 1, self.window_size)

    def compute(self, obs: np.ndarray, actions: np.ndarray,
                next_obs: np.ndarray, dones: np.ndarray = None) -> np.ndarray:
        """Compute intrinsic responsiveness reward for the current step.

        Args:
            obs: Hider observations [num_envs, obs_dim]
            actions: Hider actions [num_envs, act_dim]
            next_obs: Next hider observations [num_envs, obs_dim]
            dones: Episode done flags [num_envs] (resets buffer on done)

        Returns:
            intrinsic_reward: [num_envs] float array. Higher = more responsive.
        """
        # Store transition
        self._store(obs, actions, next_obs)

        # Reset buffers for finished episodes
        if dones is not None:
            done_ids = np.where(dones)[0]
            if len(done_ids) > 0:
                self.reset(done_ids)

        # Compute rewards per env
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        min_samples = max(50, self.num_bins ** 2)  # Need enough data for histograms

        for i in range(self.num_envs):
            n = self._count[i]
            if n < min_samples:
                continue  # Not enough data yet

            opp = self._opp_pos_buf[i, :n]    # [n, 2]
            act = self._action_buf[i, :n]      # [n, 2]

            if self.method == 'te' or self.method == 'both':
                te = self._transfer_entropy(opp, act)
                rewards[i] += te

            if self.method == 'kl' or self.method == 'both':
                kl = self._conditional_kl(opp, act)
                rewards[i] += kl

        return rewards

    def _transfer_entropy(self, opp_states: np.ndarray,
                          actions: np.ndarray) -> float:
        """Estimate transfer entropy from opponent state to agent actions.

        TE(X -> Y) = H(Y_t | Y_{t-1}) - H(Y_t | Y_{t-1}, X_{t-1})

        where X = opponent state, Y = agent actions.

        High TE = opponent state adds information about the agent's next action
        beyond what the agent's previous action already predicts.
        TE ≈ 0 = agent ignores the opponent.
        """
        n = len(actions)
        if n < 3:
            return 0.0

        # Use consecutive action pairs and corresponding opponent state
        y_prev = actions[:-1]   # [n-1, 2] - previous action
        y_curr = actions[1:]    # [n-1, 2] - current action
        x_prev = opp_states[:-1]  # [n-1, 2] - previous opponent state

        nb = self.num_bins

        # Bin all variables
        y_prev_b = self._bin_2d(y_prev, nb)   # [n-1] integer bin ids
        y_curr_b = self._bin_2d(y_curr, nb)
        x_prev_b = self._bin_2d(x_prev, nb)

        # H(Y_t | Y_{t-1}) - conditional entropy without opponent
        h_y_given_yprev = self._cond_entropy(y_curr_b, y_prev_b, nb * nb)

        # H(Y_t | Y_{t-1}, X_{t-1}) - conditional entropy with opponent
        # Combine Y_{t-1} and X_{t-1} into a joint condition
        joint_cond = y_prev_b * (nb * nb) + x_prev_b
        n_joint_bins = (nb * nb) ** 2
        h_y_given_yprev_x = self._cond_entropy(y_curr_b, joint_cond, n_joint_bins)

        te = max(0.0, h_y_given_yprev - h_y_given_yprev_x)
        return te

    def _conditional_kl(self, opp_states: np.ndarray,
                        actions: np.ndarray) -> float:
        """Compute KL divergence between action distributions when opponent
        is near vs far.

        KL(P_near || P_far) where P is the histogram of agent actions.

        High KL = agent behaves differently when opponent is close vs far.
        KL ≈ 0 = same behavior regardless of opponent proximity (degenerate).
        """
        # Opponent distance (norm of relative position)
        opp_dist = np.linalg.norm(opp_states, axis=1)  # [n]

        near_mask = opp_dist < self.near_threshold
        far_mask = ~near_mask

        n_near = near_mask.sum()
        n_far = far_mask.sum()

        if n_near < 10 or n_far < 10:
            return 0.0  # Not enough data in both groups

        nb = self.num_bins

        # Bin actions
        act_bins = self._bin_2d(actions, nb)  # [n]

        # Compute histograms
        n_bins_total = nb * nb
        p_near = np.bincount(act_bins[near_mask], minlength=n_bins_total).astype(np.float64)
        p_far = np.bincount(act_bins[far_mask], minlength=n_bins_total).astype(np.float64)

        # Normalize to probability distributions with Laplace smoothing
        p_near = (p_near + 1.0) / (n_near + n_bins_total)
        p_far = (p_far + 1.0) / (n_far + n_bins_total)

        # Symmetric KL (Jensen-Shannon-like): 0.5 * KL(P||Q) + 0.5 * KL(Q||P)
        kl_pq = np.sum(p_near * np.log(p_near / p_far))
        kl_qp = np.sum(p_far * np.log(p_far / p_near))
        jsd = 0.5 * (kl_pq + kl_qp)

        return float(jsd)

    @staticmethod
    def _bin_2d(data: np.ndarray, num_bins: int) -> np.ndarray:
        """Bin 2D data into num_bins x num_bins grid, return flat bin indices.

        Data is clipped to [-1, 1] then mapped to [0, num_bins-1].
        """
        clipped = np.clip(data, -1.0, 1.0)
        # Map [-1, 1] -> [0, num_bins-1]
        binned = ((clipped + 1.0) * 0.5 * (num_bins - 1e-6)).astype(np.int32)
        binned = np.clip(binned, 0, num_bins - 1)
        # Flatten 2D bin to 1D index
        return binned[:, 0] * num_bins + binned[:, 1]

    @staticmethod
    def _cond_entropy(target: np.ndarray, condition: np.ndarray,
                      n_cond_bins: int) -> float:
        """Compute H(target | condition) using binned histograms.

        H(Y|X) = sum_x P(x) * H(Y|X=x)
        """
        n = len(target)
        n_target_bins = int(target.max()) + 1 if len(target) > 0 else 1

        h_total = 0.0
        for x in np.unique(condition):
            mask = condition == x
            n_x = mask.sum()
            if n_x < 2:
                continue
            p_x = n_x / n

            # H(Y | X=x)
            counts = np.bincount(target[mask], minlength=n_target_bins).astype(np.float64)
            probs = counts / n_x
            probs = probs[probs > 0]
            h_yx = -np.sum(probs * np.log(probs))

            h_total += p_x * h_yx

        return h_total

    def get_stats(self) -> dict:
        """Return current responsiveness statistics for logging."""
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        te_vals = np.zeros(self.num_envs, dtype=np.float32)
        kl_vals = np.zeros(self.num_envs, dtype=np.float32)
        min_samples = max(50, self.num_bins ** 2)

        for i in range(self.num_envs):
            n = self._count[i]
            if n < min_samples:
                continue
            opp = self._opp_pos_buf[i, :n]
            act = self._action_buf[i, :n]
            te_vals[i] = self._transfer_entropy(opp, act)
            kl_vals[i] = self._conditional_kl(opp, act)

        return {
            'te_mean': float(te_vals.mean()),
            'kl_mean': float(kl_vals.mean()),
            'te_max': float(te_vals.max()),
            'kl_max': float(kl_vals.max()),
        }
