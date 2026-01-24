#!/usr/bin/env python3
"""
Dual evaluation environment wrapper for the tag game.

Adapts the vectorized tag environment to work with both vanilla self-play
and the Sandwich CRO algorithm from sandwich-reef.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from trainer.tag_env import SingleTagEnv, TagEnvConfig


class TagDualEnv:
    """
    Dual evaluation environment for tag game compatible with sandwich-reef.

    Maps:
    - protagonist -> seeker (the chaser)
    - antagonist -> hider (the evader)

    Supports two evaluation modes:
    - 'adversarial': Both agents actively controlled
    - 'clean': Antagonist uses zero/random actions (protagonist evaluated alone)
    """

    def __init__(self, config: Optional[TagEnvConfig] = None):
        self.env = SingleTagEnv(config=config)
        self._mode = 'adversarial'

        # Define observation and action spaces
        self._obs_dim = self.env.obs_dim
        self._act_dim = self.env.act_dim

        # Gymnasium-compatible spaces
        self._observation_spaces = {
            'protagonist': spaces.Box(low=-np.inf, high=np.inf,
                                     shape=(self._obs_dim,), dtype=np.float32),
            'antagonist': spaces.Box(low=-np.inf, high=np.inf,
                                    shape=(self._obs_dim,), dtype=np.float32),
        }

        self._action_spaces = {
            'protagonist': spaces.Box(low=-1.0, high=1.0,
                                     shape=(self._act_dim,), dtype=np.float32),
            'antagonist': spaces.Box(low=-1.0, high=1.0,
                                    shape=(self._act_dim,), dtype=np.float32),
        }

        self._last_obs: Optional[Dict[str, np.ndarray]] = None

    def observation_space(self, agent_id: str) -> spaces.Space:
        """Get observation space for an agent."""
        return self._observation_spaces[agent_id]

    def action_space(self, agent_id: str) -> spaces.Space:
        """Get action space for an agent."""
        return self._action_spaces[agent_id]

    def set_evaluation_mode(self, mode: str):
        """
        Set evaluation mode.

        Args:
            mode: 'adversarial' or 'clean'
        """
        assert mode in ('adversarial', 'clean'), f"Invalid mode: {mode}"
        self._mode = mode

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment."""
        if seed is not None:
            np.random.seed(seed)

        obs = self.env.reset()

        # Map seeker/hider to protagonist/antagonist
        self._last_obs = {
            'protagonist': obs['seeker'].astype(np.float32),
            'antagonist': obs['hider'].astype(np.float32),
        }

        return self._last_obs, {}

    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Any]
    ]:
        """
        Take a step in the environment.

        Args:
            actions: Dict with 'protagonist' and 'antagonist' actions

        Returns:
            observations, rewards, terminations, truncations, infos
        """
        # Map protagonist/antagonist to seeker/hider
        seeker_action = actions['protagonist']

        if self._mode == 'adversarial':
            hider_action = actions['antagonist']
        else:
            # Clean mode: hider takes zero actions
            hider_action = np.zeros(self._act_dim, dtype=np.float32)

        env_actions = {
            'seeker': seeker_action,
            'hider': hider_action,
        }

        obs, rewards, done, info = self.env.step(env_actions)

        # Map back to protagonist/antagonist
        observations = {
            'protagonist': obs['seeker'].astype(np.float32),
            'antagonist': obs['hider'].astype(np.float32),
        }

        reward_dict = {
            'protagonist': rewards['seeker'],
            'antagonist': rewards['hider'],
        }

        terminations = {
            'protagonist': done,
            'antagonist': done,
        }

        truncations = {
            'protagonist': False,
            'antagonist': False,
        }

        infos = {
            'protagonist': {'tagged': info.get('tagged', False)},
            'antagonist': {'tagged': info.get('tagged', False)},
        }

        self._last_obs = observations
        return observations, reward_dict, terminations, truncations, infos

    def get_state(self) -> Dict[str, Any]:
        """Get full state for debugging/visualization."""
        return self.env.get_state()


class TagGymEnv(gym.Env):
    """
    Single-agent Gym wrapper for tag (seeker perspective).

    Used for vanilla self-play where we train seeker and hider separately.
    """

    def __init__(self, role: str = 'seeker', config: Optional[TagEnvConfig] = None):
        super().__init__()
        assert role in ('seeker', 'hider')
        self.role = role
        self.opponent_role = 'hider' if role == 'seeker' else 'seeker'

        self.env = SingleTagEnv(config=config)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.env.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.env.act_dim,), dtype=np.float32
        )

        self._opponent_policy = None
        self._last_obs = None

    def set_opponent(self, policy):
        """Set the opponent policy (callable: obs -> action)."""
        self._opponent_policy = policy

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        obs = self.env.reset()
        self._last_obs = obs
        return obs[self.role].astype(np.float32), {}

    def step(self, action):
        # Get opponent action
        if self._opponent_policy is not None:
            opp_obs = self._last_obs[self.opponent_role]
            opp_action = self._opponent_policy(opp_obs)
        else:
            opp_action = np.zeros(self.env.act_dim, dtype=np.float32)

        actions = {
            self.role: action,
            self.opponent_role: opp_action,
        }

        obs, rewards, done, info = self.env.step(actions)
        self._last_obs = obs

        return (
            obs[self.role].astype(np.float32),
            rewards[self.role],
            done,
            False,  # truncated
            info
        )


if __name__ == "__main__":
    # Test the dual environment
    env = TagDualEnv()

    print("Testing TagDualEnv...")
    print(f"Protagonist obs space: {env.observation_space('protagonist')}")
    print(f"Antagonist action space: {env.action_space('antagonist')}")

    obs, _ = env.reset()
    print(f"\nInitial obs shapes: protagonist={obs['protagonist'].shape}, antagonist={obs['antagonist'].shape}")

    # Test adversarial mode
    env.set_evaluation_mode('adversarial')
    for i in range(5):
        actions = {
            'protagonist': np.random.uniform(-1, 1, 3).astype(np.float32),
            'antagonist': np.random.uniform(-1, 1, 3).astype(np.float32),
        }
        obs, rewards, terms, truncs, infos = env.step(actions)
        print(f"Step {i}: rewards={rewards}, done={terms['protagonist']}")
        if terms['protagonist']:
            obs, _ = env.reset()

    print("\nTagDualEnv test passed!")
