#!/usr/bin/env python3
"""Vectorized 2D tag environment for efficient RL training."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, Optional, List


@dataclass
class Obstacle:
    """A rectangular obstacle that blocks movement and vision."""
    x: float           # Center x position
    y: float           # Center y position
    half_width: float  # Half-width (extends +/- from center)
    half_height: float # Half-height (extends +/- from center)


@dataclass
class SafeZone:
    """A circular safe zone where the hider cannot be tagged."""
    x: float = 0.0              # Center x position
    y: float = 0.0              # Center y position
    radius: float = 2.5         # Zone radius
    protection_duration: float = 2.0   # Seconds of protection before exhaustion
    cooldown_duration: float = 5.0     # Seconds of exhaustion (no protection)


# Predefined arena layouts
LAYOUTS: Dict[str, Dict[str, Any]] = {
    'empty': {
        'obstacles': [],
        'safe_zone': None,
    },
    'four_corners': {
        'obstacles': [
            Obstacle(x=-8.0, y=8.0, half_width=1.5, half_height=1.5),   # Top-left
            Obstacle(x=8.0, y=8.0, half_width=1.5, half_height=1.5),    # Top-right
            Obstacle(x=-3.0, y=-8.0, half_width=1.5, half_height=1.5),  # Bottom-left (asymmetric)
            Obstacle(x=3.0, y=-8.0, half_width=1.5, half_height=1.5),   # Bottom-right (asymmetric)
        ],
        'safe_zone': SafeZone(x=0.0, y=0.0, radius=2.5),
    },
    'central_cross': {
        'obstacles': [
            Obstacle(x=0.0, y=6.0, half_width=1.5, half_height=1.5),    # Top
            Obstacle(x=0.0, y=-6.0, half_width=1.5, half_height=1.5),   # Bottom
            Obstacle(x=6.0, y=0.0, half_width=1.5, half_height=1.5),    # Right
            Obstacle(x=-6.0, y=0.0, half_width=1.5, half_height=1.5),   # Left
        ],
        'safe_zone': SafeZone(x=0.0, y=0.0, radius=2.5),
    },
    # --- Geometry study layouts: vary obstacle-to-corner proximity ---
    'one_corner': {
        'obstacles': [
            Obstacle(x=-8.0, y=8.0, half_width=1.5, half_height=1.5),   # Top-left only
        ],
        'safe_zone': SafeZone(x=0.0, y=0.0, radius=2.5),
    },
    'two_corners': {
        'obstacles': [
            Obstacle(x=-8.0, y=8.0, half_width=1.5, half_height=1.5),   # Top-left
            Obstacle(x=8.0, y=-8.0, half_width=1.5, half_height=1.5),   # Bottom-right (diagonal)
        ],
        'safe_zone': SafeZone(x=0.0, y=0.0, radius=2.5),
    },
    'wall_midpoints': {
        'obstacles': [
            Obstacle(x=0.0, y=10.0, half_width=1.5, half_height=1.5),   # Top wall midpoint
            Obstacle(x=0.0, y=-10.0, half_width=1.5, half_height=1.5),  # Bottom wall midpoint
            Obstacle(x=10.0, y=0.0, half_width=1.5, half_height=1.5),   # Right wall midpoint
            Obstacle(x=-10.0, y=0.0, half_width=1.5, half_height=1.5),  # Left wall midpoint
        ],
        'safe_zone': SafeZone(x=0.0, y=0.0, radius=2.5),
    },
    'corner_tight': {
        'obstacles': [
            Obstacle(x=-12.0, y=12.0, half_width=1.5, half_height=1.5), # Very close to TL corner
            Obstacle(x=12.0, y=12.0, half_width=1.5, half_height=1.5),  # Very close to TR corner
            Obstacle(x=-12.0, y=-12.0, half_width=1.5, half_height=1.5),# Very close to BL corner
            Obstacle(x=12.0, y=-12.0, half_width=1.5, half_height=1.5), # Very close to BR corner
        ],
        'safe_zone': SafeZone(x=0.0, y=0.0, radius=2.5),
    },
    'center_cluster': {
        'obstacles': [
            Obstacle(x=-3.0, y=3.0, half_width=1.5, half_height=1.5),
            Obstacle(x=3.0, y=3.0, half_width=1.5, half_height=1.5),
            Obstacle(x=-3.0, y=-3.0, half_width=1.5, half_height=1.5),
            Obstacle(x=3.0, y=-3.0, half_width=1.5, half_height=1.5),
        ],
        'safe_zone': SafeZone(x=0.0, y=0.0, radius=2.5),
    },
    'playground': {
        'obstacles': [
            # Central chokepoint: two pillars with gap between
            Obstacle(x=-2.0, y=0.0, half_width=1.0, half_height=2.5),   # Left pillar
            Obstacle(x=2.0, y=0.0, half_width=1.0, half_height=2.5),    # Right pillar
            # NW L-shaped corridor
            Obstacle(x=-8.0, y=8.0, half_width=3.0, half_height=0.75),  # Horizontal arm
            Obstacle(x=-10.25, y=5.5, half_width=0.75, half_height=2.5), # Vertical arm
            # SE L-shaped corridor
            Obstacle(x=8.0, y=-8.0, half_width=3.0, half_height=0.75),  # Horizontal arm
            Obstacle(x=10.25, y=-5.5, half_width=0.75, half_height=2.5), # Vertical arm
            # NE dead-end pocket (risky shelter)
            Obstacle(x=8.0, y=10.0, half_width=4.0, half_height=0.75),  # Top wall
            Obstacle(x=5.0, y=7.5, half_width=0.75, half_height=2.5),   # Side wall (gap at bottom)
            # SW cover block
            Obstacle(x=-7.0, y=-5.0, half_width=1.5, half_height=1.5),
            # East wall segment
            Obstacle(x=10.0, y=2.0, half_width=0.75, half_height=3.0),
        ],
        'safe_zone': SafeZone(x=-5.0, y=5.0, radius=2.5),  # Off-center
    },
}


@dataclass
class TagEnvConfig:
    """Configuration for the tag environment."""
    arena_half: float = 15.0       # Half-size of arena (30x30 total)
    agent_speed: float = 8.0       # Max agent movement speed
    agent_accel: float = 20.0      # Agent acceleration
    tag_distance: float = 1.5      # Distance threshold for tagging
    time_limit: float = 10.0       # Episode time limit in seconds
    dt: float = 1.0 / 60.0         # Physics timestep (60 Hz)
    steps_per_action: int = 3      # Physics steps per action

    # Observation config
    num_rays: int = 36             # Number of vision rays
    ray_fov: float = 120.0         # Field of view in degrees
    ray_max_dist: float = 15.0     # Max ray distance

    # Reward shaping
    distance_reward_scale: float = 0.14
    seeker_time_penalty: float = -0.005
    runner_survival_bonus: float = 0.01
    win_bonus: float = 10.0
    timeout_hider_bonus: float = 6.0
    timeout_seeker_penalty: float = 6.0

    # Hider distance shaping
    hider_dist_reward_scale: float = 0.0   # Reward for increasing distance (0 = off)
    hider_abs_dist_reward_scale: float = 0.1  # Reward proportional to absolute distance

    # Wall proximity penalty (hider): applied when within wall_prox_dist of any wall
    hider_wall_prox_penalty: float = 0.0   # Per-step penalty (negative value to penalize)
    wall_prox_dist: float = 2.0            # Distance threshold for wall penalty

    # Hider minimum speed reward: bonus when hider is moving
    hider_min_speed_reward: float = 0.0    # Per-step reward when hider speed > 1.0

    # Seeker escalating urgency: time penalty scales from 1x to 2x over episode
    seeker_escalating_urgency: bool = False

    # Area coverage bonus: reward for visiting new grid cells (6x6 grid)
    area_coverage_bonus: float = 0.0       # Per new cell visited

    # Arena layout
    layout: str = 'empty'          # Layout name: 'empty', 'four_corners', 'central_cross', 'playground'

    # Sprint/stamina system
    enable_sprint: bool = False
    sprint_speed_mult: float = 1.5       # Max speed multiplier when sprinting
    max_stamina: float = 3.0             # Seconds of full sprint
    stamina_regen_rate: float = 1.0      # Stamina per second when not sprinting

    # Hider speed advantage
    hider_speed_mult: float = 1.0        # Base speed multiplier for hider (1.0 = equal)


class VecTagEnv:
    """
    Vectorized tag environment for fast parallel training.

    Simulates multiple tag games simultaneously using numpy operations.
    Both seeker and hider are controlled by policies (self-play).
    """

    def __init__(self, num_envs: int, config: Optional[TagEnvConfig] = None):
        self.num_envs = num_envs
        self.cfg = config or TagEnvConfig()

        # Load layout
        layout_data = LAYOUTS.get(self.cfg.layout, LAYOUTS['empty'])
        self.obstacles: List[Obstacle] = layout_data['obstacles']
        self.safe_zone: Optional[SafeZone] = layout_data['safe_zone']

        # Pre-computed obstacle arrays
        self._num_obstacles = len(self.obstacles)
        if self._num_obstacles > 0:
            self._obs_centers = np.array(
                [[o.x, o.y] for o in self.obstacles], dtype=np.float32)  # [O, 2]
            self._obs_half_sizes = np.array(
                [[o.half_width, o.half_height] for o in self.obstacles], dtype=np.float32)  # [O, 2]
            self._obs_min = self._obs_centers - self._obs_half_sizes  # [O, 2]
            self._obs_max = self._obs_centers + self._obs_half_sizes  # [O, 2]

        # Pre-computed ray angle offsets
        half_fov = np.radians(self.cfg.ray_fov / 2)
        self._ray_offsets = np.linspace(
            -half_fov, half_fov, self.cfg.num_rays).astype(np.float32)  # [R]

        # Env index array (reused frequently)
        self._eids = np.arange(num_envs)

        # State arrays for all environments
        # Positions: [num_envs, 2, 2] - (env, agent, xy)
        self.positions = np.zeros((num_envs, 2, 2), dtype=np.float32)
        # Velocities: [num_envs, 2, 2] - (env, agent, xy)
        self.velocities = np.zeros((num_envs, 2, 2), dtype=np.float32)
        # Facing directions (angle in radians)
        self.facing = np.zeros((num_envs, 2), dtype=np.float32)
        # Time elapsed per environment
        self.time_elapsed = np.zeros(num_envs, dtype=np.float32)
        # Episode done flags
        self.dones = np.zeros(num_envs, dtype=bool)
        # Who is "it" (seeker): 0 = agent0, 1 = agent1
        self.seeker_idx = np.zeros(num_envs, dtype=np.int32)

        # Track previous distances for reward shaping
        self.prev_distances = np.zeros(num_envs, dtype=np.float32)

        # Safe zone state (for hider only)
        # Time spent in safe zone (resets when leaving)
        self.safe_zone_time = np.zeros(num_envs, dtype=np.float32)
        # Whether protection is exhausted
        self.safe_zone_exhausted = np.zeros(num_envs, dtype=bool)
        # Cooldown remaining (counts down when exhausted)
        self.safe_zone_cooldown = np.zeros(num_envs, dtype=np.float32)

        # Stamina state: [num_envs, 2] - one per agent
        self.stamina = np.full((num_envs, 2), self.cfg.max_stamina, dtype=np.float32)

        # Area coverage grid: 6x6 cells, tracked per agent per env
        self._coverage_grid_size = 6
        self.coverage_grid = np.zeros(
            (num_envs, 2, self._coverage_grid_size, self._coverage_grid_size), dtype=bool)

        # Observation dimension: position(2) + velocity(2) + relative(4) + roles(2) + forward(2) + rays(72) + safe_zone(3) + stamina(2 if sprint)
        self.obs_dim = 12 + self.cfg.num_rays * 2 + 3 + (2 if self.cfg.enable_sprint else 0)
        self.act_dim = 3  # move_x, move_z, sprint_intensity (3rd dim used for sprint when enabled)

        self.reset()

    def reset(self, env_ids: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Reset specified environments (or all if env_ids is None)."""
        if env_ids is None:
            env_ids = np.arange(self.num_envs)

        n = len(env_ids)

        # Random positions with minimum separation
        min_sep = 6.0
        half = self.cfg.arena_half - 2.0  # Keep away from walls
        agent_radius = 0.5

        for i, eid in enumerate(env_ids):
            # Place agents with guaranteed separation, avoiding obstacles
            max_attempts = 50
            for _ in range(max_attempts):
                pos1 = np.random.uniform(-half, half, size=2)
                if not self._point_in_obstacle(pos1, agent_radius):
                    break

            angle = np.random.uniform(0, 2 * np.pi)
            for _ in range(max_attempts):
                pos2 = pos1 + np.array([np.cos(angle), np.sin(angle)]) * min_sep
                pos2 = np.clip(pos2, -half, half)
                if not self._point_in_obstacle(pos2, agent_radius):
                    break
                angle = np.random.uniform(0, 2 * np.pi)

            self.positions[eid, 0] = pos1
            self.positions[eid, 1] = pos2

        # Reset velocities
        self.velocities[env_ids] = 0.0

        # Random facing directions
        self.facing[env_ids] = np.random.uniform(0, 2 * np.pi, size=(n, 2))

        # Reset time
        self.time_elapsed[env_ids] = 0.0
        self.dones[env_ids] = False

        # Randomly assign seeker role
        self.seeker_idx[env_ids] = np.random.randint(0, 2, size=n)

        # Reset safe zone state
        self.safe_zone_time[env_ids] = 0.0
        self.safe_zone_exhausted[env_ids] = False
        self.safe_zone_cooldown[env_ids] = 0.0

        # Reset stamina
        self.stamina[env_ids] = self.cfg.max_stamina

        # Reset coverage grid
        self.coverage_grid[env_ids] = False

        # Compute initial distances
        self.prev_distances[env_ids] = self._compute_distances(env_ids)

        return self._get_obs()

    def _point_in_obstacle(self, pos: np.ndarray, radius: float = 0.0) -> bool:
        """Check if a point (with optional radius) overlaps any obstacle."""
        if self._num_obstacles == 0:
            return False
        # Vectorized check against all obstacles at once
        dx = np.abs(pos[0] - self._obs_centers[:, 0])
        dy = np.abs(pos[1] - self._obs_centers[:, 1])
        expanded_hw = self._obs_half_sizes[:, 0] + radius
        expanded_hh = self._obs_half_sizes[:, 1] + radius
        return bool(np.any((dx < expanded_hw) & (dy < expanded_hh)))

    def _compute_distances(self, env_ids: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute distances between agents."""
        if env_ids is None:
            env_ids = np.arange(self.num_envs)

        diff = self.positions[env_ids, 0] - self.positions[env_ids, 1]
        return np.linalg.norm(diff, axis=1)

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """
        Get observations for both agents in all environments.
        Returns dict with 'seeker' and 'hider' observations.
        All operations are batched across environments.
        """
        E = self.num_envs
        eids = self._eids
        obs = {}

        for role in ['seeker', 'hider']:
            role_obs = np.zeros((E, self.obs_dim), dtype=np.float32)

            if role == 'seeker':
                agent_idx_arr = self.seeker_idx           # [E]
                other_idx_arr = 1 - self.seeker_idx       # [E]
            else:
                agent_idx_arr = 1 - self.seeker_idx       # [E]
                other_idx_arr = self.seeker_idx            # [E]

            # Gather per-role data using fancy indexing: [E, 2]
            pos = self.positions[eids, agent_idx_arr]
            vel = self.velocities[eids, agent_idx_arr]
            other_pos = self.positions[eids, other_idx_arr]
            other_vel = self.velocities[eids, other_idx_arr]
            facing = self.facing[eids, agent_idx_arr]     # [E]

            idx = 0
            # Normalized position
            role_obs[:, idx:idx+2] = pos / self.cfg.arena_half
            idx += 2
            # Normalized velocity
            role_obs[:, idx:idx+2] = vel / 10.0
            idx += 2
            # Relative opponent position
            role_obs[:, idx:idx+2] = (other_pos - pos) / self.cfg.arena_half
            idx += 2
            # Relative opponent velocity
            role_obs[:, idx:idx+2] = (other_vel - vel) / 10.0
            idx += 2
            # Role flags
            if role == 'seeker':
                role_obs[:, idx] = 1.0
                role_obs[:, idx+1] = 0.0
            else:
                role_obs[:, idx] = 0.0
                role_obs[:, idx+1] = 1.0
            idx += 2
            # Forward direction
            role_obs[:, idx] = np.cos(facing)
            role_obs[:, idx+1] = np.sin(facing)
            idx += 2

            # Batched ray casting: one call per role instead of per env
            rays = self._cast_rays_batched(agent_idx_arr, other_idx_arr)  # [E, R*2]
            role_obs[:, idx:idx + self.cfg.num_rays * 2] = rays
            idx += self.cfg.num_rays * 2

            # Safe zone state (3 values)
            if self.safe_zone is not None:
                hider_idx_arr = 1 - self.seeker_idx
                hider_pos = self.positions[eids, hider_idx_arr]  # [E, 2]
                in_zone = self._points_in_safe_zone(hider_pos)   # [E] bool
                role_obs[:, idx] = in_zone.astype(np.float32)
                role_obs[:, idx+1] = self.safe_zone_exhausted.astype(np.float32)
                role_obs[:, idx+2] = self.safe_zone_cooldown / self.safe_zone.cooldown_duration
            idx += 3

            # Stamina observations
            if self.cfg.enable_sprint:
                role_obs[:, idx] = self.stamina[eids, agent_idx_arr] / self.cfg.max_stamina
                role_obs[:, idx+1] = self.stamina[eids, other_idx_arr] / self.cfg.max_stamina
                idx += 2

            obs[role] = role_obs

        return obs

    def _points_in_safe_zone(self, positions: np.ndarray) -> np.ndarray:
        """Check if points [E, 2] are inside the safe zone. Returns [E] bool."""
        if self.safe_zone is None:
            return np.zeros(len(positions), dtype=bool)
        dx = positions[:, 0] - self.safe_zone.x
        dy = positions[:, 1] - self.safe_zone.y
        return (dx * dx + dy * dy) <= (self.safe_zone.radius ** 2)

    def _point_in_safe_zone(self, pos: np.ndarray) -> bool:
        """Check if a point is inside the safe zone."""
        if self.safe_zone is None:
            return False
        dx = pos[0] - self.safe_zone.x
        dy = pos[1] - self.safe_zone.y
        return (dx * dx + dy * dy) <= (self.safe_zone.radius * self.safe_zone.radius)

    def _cast_rays_batched(self, agent_idx_arr: np.ndarray,
                           other_idx_arr: np.ndarray) -> np.ndarray:
        """Batched ray casting for all envs at once.

        Args:
            agent_idx_arr: [E] array of agent indices (0 or 1)
            other_idx_arr: [E] array of other agent indices (0 or 1)

        Returns:
            [E, R*2] array of interleaved (distance, hit_type) pairs.
            hit_type: 0.0 = wall, 0.5 = obstacle, 1.0 = agent
        """
        E = self.num_envs
        R = self.cfg.num_rays
        max_dist = self.cfg.ray_max_dist
        half = self.cfg.arena_half
        eids = self._eids

        # Gather positions and facing for the role
        pos = self.positions[eids, agent_idx_arr]          # [E, 2]
        facing = self.facing[eids, agent_idx_arr]          # [E]
        other_pos = self.positions[eids, other_idx_arr]    # [E, 2]

        # Ray angles: facing + offsets -> [E, R]
        angles = facing[:, None] + self._ray_offsets[None, :]  # [E, R]
        dir_x = np.cos(angles)  # [E, R]
        dir_y = np.sin(angles)  # [E, R]

        pos_x = pos[:, 0:1]  # [E, 1]
        pos_y = pos[:, 1:2]  # [E, 1]

        # ---- Wall distances (4 walls) ----
        wall_dist = np.full((E, R), max_dist, dtype=np.float32)

        with np.errstate(divide='ignore', invalid='ignore'):
            # Right wall: x = +half
            t = (half - pos_x) / dir_x       # [E, R]
            hit_y = pos_y + dir_y * t
            valid = (t > 0) & (t < wall_dist) & (np.abs(hit_y) <= half)
            wall_dist = np.where(valid, t, wall_dist)

            # Left wall: x = -half
            t = (-half - pos_x) / dir_x
            hit_y = pos_y + dir_y * t
            valid = (t > 0) & (t < wall_dist) & (np.abs(hit_y) <= half)
            wall_dist = np.where(valid, t, wall_dist)

            # Top wall: y = +half
            t = (half - pos_y) / dir_y
            hit_x = pos_x + dir_x * t
            valid = (t > 0) & (t < wall_dist) & (np.abs(hit_x) <= half)
            wall_dist = np.where(valid, t, wall_dist)

            # Bottom wall: y = -half
            t = (-half - pos_y) / dir_y
            hit_x = pos_x + dir_x * t
            valid = (t > 0) & (t < wall_dist) & (np.abs(hit_x) <= half)
            wall_dist = np.where(valid, t, wall_dist)

        # ---- Agent (circle) distance ----
        to_circle = other_pos - pos  # [E, 2]
        # Project onto each ray direction: dot(to_circle, dir)
        proj = to_circle[:, 0:1] * dir_x + to_circle[:, 1:2] * dir_y  # [E, R]

        # Squared distance from circle center to closest point on ray
        tc_sq = to_circle[:, 0:1]**2 + to_circle[:, 1:2]**2  # [E, 1]
        dist_sq = tc_sq - proj**2  # [E, R]

        radius = 0.5
        radius_sq = radius * radius

        half_chord = np.sqrt(np.maximum(radius_sq - dist_sq, 0.0))
        hit_dist = proj - half_chord

        agent_dist = np.where(
            (proj > 0) & (dist_sq <= radius_sq) & (hit_dist < max_dist),
            np.maximum(hit_dist, 0.0),
            max_dist
        )

        # ---- Obstacle AABB distance (slab method) ----
        if self._num_obstacles > 0:
            # Broadcast: pos [E, 1, 1, 2] x dir [E, R, 1, 2] vs obs [1, 1, O, 2]
            obs_min = self._obs_min[None, None, :, :]  # [1, 1, O, 2]
            obs_max = self._obs_max[None, None, :, :]  # [1, 1, O, 2]

            dirs = np.stack([dir_x, dir_y], axis=-1)[:, :, None, :]  # [E, R, 1, 2]
            pos_exp = pos[:, None, None, :]                           # [E, 1, 1, 2]

            inv_dir = np.where(
                np.abs(dirs) > 1e-8, 1.0 / dirs, np.sign(dirs) * 1e8)

            t1 = (obs_min - pos_exp) * inv_dir  # [E, R, O, 2]
            t2 = (obs_max - pos_exp) * inv_dir  # [E, R, O, 2]

            t_near = np.minimum(t1, t2)  # [E, R, O, 2]
            t_far = np.maximum(t1, t2)   # [E, R, O, 2]

            t_enter = np.max(t_near, axis=-1)  # [E, R, O]
            t_exit = np.min(t_far, axis=-1)    # [E, R, O]

            valid_hit = (t_enter < t_exit) & (t_exit > 0) & (t_enter < max_dist)
            obs_dist_per = np.where(valid_hit, np.maximum(t_enter, 0.0), max_dist)

            obstacle_dist = np.min(obs_dist_per, axis=-1)  # [E, R]
        else:
            obstacle_dist = np.full((E, R), max_dist, dtype=np.float32)

        # ---- Combine: pick closest hit ----
        min_dist = wall_dist
        hit_type = np.zeros((E, R), dtype=np.float32)  # 0.0 = wall

        closer_obs = obstacle_dist < min_dist
        min_dist = np.where(closer_obs, obstacle_dist, min_dist)
        hit_type = np.where(closer_obs, 0.5, hit_type)

        closer_agent = agent_dist < min_dist
        min_dist = np.where(closer_agent, agent_dist, min_dist)
        hit_type = np.where(closer_agent, 1.0, hit_type)

        # Normalize and interleave
        norm_dist = np.minimum(min_dist / max_dist, 1.0)
        result = np.empty((E, R * 2), dtype=np.float32)
        result[:, 0::2] = norm_dist
        result[:, 1::2] = hit_type
        return result

    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray],
                                                            Dict[str, np.ndarray],
                                                            np.ndarray,
                                                            Dict[str, Any]]:
        """
        Execute actions for all environments.

        Args:
            actions: Dict with 'seeker' and 'hider' action arrays [num_envs, 3]

        Returns:
            obs: New observations
            rewards: Dict with 'seeker' and 'hider' rewards
            dones: Episode done flags
            infos: Additional info dict
        """
        seeker_actions = actions['seeker']
        hider_actions = actions['hider']
        eids = self._eids

        # Map role actions to agent indices (vectorized)
        agent_actions = np.zeros((self.num_envs, 2, 3), dtype=np.float32)
        agent_actions[eids, self.seeker_idx] = seeker_actions
        agent_actions[eids, 1 - self.seeker_idx] = hider_actions

        # Compute effective max speed for each agent [num_envs, 2]
        effective_speed = np.full((self.num_envs, 2), self.cfg.agent_speed, dtype=np.float32)

        # Hider speed advantage (vectorized)
        if self.cfg.hider_speed_mult != 1.0:
            effective_speed[eids, 1 - self.seeker_idx] *= self.cfg.hider_speed_mult

        # Sprint system (vectorized per agent)
        if self.cfg.enable_sprint:
            action_dt = self.cfg.dt * self.cfg.steps_per_action
            for agent_idx in range(2):
                # Map agent_idx -> role to get correct sprint action
                is_seeker = (self.seeker_idx == agent_idx)  # [E] bool
                sprint_raw = np.where(is_seeker, seeker_actions[:, 2], hider_actions[:, 2])
                sprint_intensity = np.clip((sprint_raw + 1.0) * 0.5, 0.0, 1.0)

                sprinting = (self.stamina[:, agent_idx] > 0) & (sprint_intensity > 0.1)

                # Speed boost for sprinters
                speed_mult = 1.0 + (self.cfg.sprint_speed_mult - 1.0) * sprint_intensity
                effective_speed[:, agent_idx] = np.where(
                    sprinting,
                    effective_speed[:, agent_idx] * speed_mult,
                    effective_speed[:, agent_idx])

                # Drain stamina for sprinters, regen for non-sprinters
                self.stamina[:, agent_idx] = np.where(
                    sprinting,
                    np.maximum(0.0, self.stamina[:, agent_idx] - sprint_intensity * action_dt),
                    np.minimum(self.cfg.max_stamina,
                               self.stamina[:, agent_idx] + self.cfg.stamina_regen_rate * action_dt))

        # Simulate physics for multiple substeps
        dt = self.cfg.dt
        for _ in range(self.cfg.steps_per_action):
            for agent_idx in range(2):
                # Get move inputs (clamped to [-1, 1])
                move_input = np.clip(agent_actions[:, agent_idx, :2], -1, 1)

                # Per-env max speed for this agent
                max_speed = effective_speed[:, agent_idx]

                # Apply acceleration
                target_vel = move_input * max_speed[:, np.newaxis]
                accel = (target_vel - self.velocities[:, agent_idx]) * self.cfg.agent_accel * dt
                self.velocities[:, agent_idx] += accel

                # Clamp velocity to per-env max speed
                speed = np.linalg.norm(self.velocities[:, agent_idx], axis=1, keepdims=True)
                speed = np.maximum(speed, 1e-6)
                self.velocities[:, agent_idx] = np.where(
                    speed > max_speed[:, np.newaxis],
                    self.velocities[:, agent_idx] / speed * max_speed[:, np.newaxis],
                    self.velocities[:, agent_idx]
                )

                # Update positions
                self.positions[:, agent_idx] += self.velocities[:, agent_idx] * dt

                # Wall collision
                self.positions[:, agent_idx] = np.clip(
                    self.positions[:, agent_idx],
                    -self.cfg.arena_half + 0.5,
                    self.cfg.arena_half - 0.5
                )

                # Obstacle collision (vectorized inner loop)
                self._resolve_obstacle_collisions(agent_idx)

                # Update facing direction based on velocity
                speed = np.linalg.norm(self.velocities[:, agent_idx], axis=1)
                moving = speed > 0.5
                new_facing = np.arctan2(
                    self.velocities[moving, agent_idx, 1],
                    self.velocities[moving, agent_idx, 0]
                )
                self.facing[moving, agent_idx] = new_facing

            self.time_elapsed += dt

        # Update safe zone state
        self._update_safe_zone_state(dt * self.cfg.steps_per_action)

        # Compute distances
        distances = self._compute_distances()

        # Check for tags (considering safe zone protection)
        tagged = distances < self.cfg.tag_distance

        # Check safe zone protection for hider (vectorized)
        if self.safe_zone is not None:
            hider_idx_arr = 1 - self.seeker_idx
            hider_pos = self.positions[eids, hider_idx_arr]  # [E, 2]
            in_zone = self._points_in_safe_zone(hider_pos)
            protected = tagged & in_zone & ~self.safe_zone_exhausted
            tagged = tagged & ~protected

        # Check for timeouts
        timed_out = self.time_elapsed >= self.cfg.time_limit

        # Compute rewards
        progress = self.prev_distances - distances  # Positive when closing gap

        seeker_rewards = np.zeros(self.num_envs, dtype=np.float32)
        hider_rewards = np.zeros(self.num_envs, dtype=np.float32)

        # Distance-based shaping
        seeker_rewards += progress * self.cfg.distance_reward_scale
        # Hider: reward for increasing distance from seeker
        hider_rewards -= progress * self.cfg.hider_dist_reward_scale
        # Hider: reward proportional to absolute distance
        hider_rewards += (distances / self.cfg.arena_half) * self.cfg.hider_abs_dist_reward_scale

        # Wall proximity penalty (hider)
        if self.cfg.hider_wall_prox_penalty != 0.0:
            eids = np.arange(self.num_envs)
            hider_idx_arr = 1 - self.seeker_idx
            hider_pos = self.positions[eids, hider_idx_arr]  # [E, 2]
            wall_dist = self.cfg.arena_half - np.abs(hider_pos)  # [E, 2]
            min_wall_dist = np.min(wall_dist, axis=1)  # [E]
            near_wall = min_wall_dist < self.cfg.wall_prox_dist
            # Linear penalty: full at wall, zero at threshold
            penalty_scale = 1.0 - min_wall_dist / self.cfg.wall_prox_dist
            penalty_scale = np.clip(penalty_scale, 0.0, 1.0)
            hider_rewards += self.cfg.hider_wall_prox_penalty * penalty_scale

        # Hider minimum speed reward
        if self.cfg.hider_min_speed_reward != 0.0:
            eids = np.arange(self.num_envs)
            hider_idx_arr = 1 - self.seeker_idx
            hider_speed = np.linalg.norm(self.velocities[eids, hider_idx_arr], axis=1)
            hider_rewards += self.cfg.hider_min_speed_reward * (hider_speed > 1.0)

        # Area coverage bonus
        if self.cfg.area_coverage_bonus != 0.0:
            eids = np.arange(self.num_envs)
            gs = self._coverage_grid_size
            half = self.cfg.arena_half
            for agent_idx in range(2):
                pos = self.positions[eids, agent_idx]  # [E, 2]
                # Map position to grid cell
                gx = np.clip(((pos[:, 0] + half) / (2 * half) * gs).astype(int), 0, gs - 1)
                gy = np.clip(((pos[:, 1] + half) / (2 * half) * gs).astype(int), 0, gs - 1)
                # Check which are newly visited
                already = self.coverage_grid[eids, agent_idx, gx, gy]
                new_cells = ~already
                # Mark visited
                self.coverage_grid[eids, agent_idx, gx, gy] = True
                # Assign reward based on role
                is_seeker = (self.seeker_idx == agent_idx)
                is_hider = ~is_seeker
                seeker_rewards += self.cfg.area_coverage_bonus * new_cells * is_seeker
                hider_rewards += self.cfg.area_coverage_bonus * new_cells * is_hider

        # Time-based rewards (with optional escalating urgency)
        if self.cfg.seeker_escalating_urgency:
            urgency_scale = 1.0 + self.time_elapsed / self.cfg.time_limit
            seeker_rewards += self.cfg.seeker_time_penalty * urgency_scale
        else:
            seeker_rewards += self.cfg.seeker_time_penalty
        hider_rewards += self.cfg.runner_survival_bonus

        # Terminal rewards
        seeker_rewards[tagged] += self.cfg.win_bonus
        hider_rewards[tagged] -= self.cfg.win_bonus

        seeker_rewards[timed_out & ~tagged] -= self.cfg.timeout_seeker_penalty
        hider_rewards[timed_out & ~tagged] += self.cfg.timeout_hider_bonus

        # Update done flags
        self.dones = tagged | timed_out
        self.prev_distances = distances.copy()

        # Compute behavioral metrics
        eids = np.arange(self.num_envs)
        hider_idx_arr = 1 - self.seeker_idx
        hider_pos = self.positions[eids, hider_idx_arr]
        hider_wall_dist = self.cfg.arena_half - np.abs(hider_pos)
        hider_min_wall = np.min(hider_wall_dist, axis=1)
        hider_speed = np.linalg.norm(self.velocities[eids, hider_idx_arr], axis=1)
        seeker_speed = np.linalg.norm(self.velocities[eids, self.seeker_idx], axis=1)

        # Corner proximity: near two walls simultaneously (both x and y within threshold)
        corner_thresh = 3.0
        in_corner = (hider_wall_dist[:, 0] < corner_thresh) & (hider_wall_dist[:, 1] < corner_thresh)
        # Speed while near walls
        near_wall_mask = hider_min_wall < 2.0
        hider_wall_speed = float(hider_speed[near_wall_mask].mean()) if near_wall_mask.any() else 0.0

        # Build info dict
        infos = {
            'tagged': tagged.copy(),
            'timed_out': timed_out.copy(),
            'distances': distances.copy(),
            'time_elapsed': self.time_elapsed.copy(),
            'seeker_wins': tagged.sum(),
            'hider_wins': (timed_out & ~tagged).sum(),
            'hider_wall_dist_mean': float(hider_min_wall.mean()),
            'hider_near_wall_frac': float((hider_min_wall < 2.0).mean()),
            'hider_corner_frac': float(in_corner.mean()),
            'hider_speed_mean': float(hider_speed.mean()),
            'hider_wall_speed_mean': hider_wall_speed,
            'seeker_speed_mean': float(seeker_speed.mean()),
        }

        # Get new observations
        obs = self._get_obs()

        rewards = {
            'seeker': seeker_rewards,
            'hider': hider_rewards,
        }

        return obs, rewards, self.dones.copy(), infos

    def _resolve_obstacle_collisions(self, agent_idx: int):
        """Push agents out of obstacles using AABB + Minkowski sum.

        Outer loop over obstacles preserved (sequential push semantics),
        inner per-env loop fully vectorized.
        """
        if self._num_obstacles == 0:
            return

        agent_radius = 0.5

        for o in range(self._num_obstacles):
            cx = self._obs_centers[o, 0]
            cy = self._obs_centers[o, 1]
            expanded_hw = self._obs_half_sizes[o, 0] + agent_radius
            expanded_hh = self._obs_half_sizes[o, 1] + agent_radius

            # Read current positions (reflects pushes from previous obstacles)
            px = self.positions[:, agent_idx, 0]  # [E]
            py = self.positions[:, agent_idx, 1]  # [E]
            dx = px - cx
            dy = py - cy
            abs_dx = np.abs(dx)
            abs_dy = np.abs(dy)

            inside = (abs_dx < expanded_hw) & (abs_dy < expanded_hh)
            if not np.any(inside):
                continue

            overlap_x = expanded_hw - abs_dx
            overlap_y = expanded_hh - abs_dy

            push_x_axis = overlap_x < overlap_y

            # Push along x axis
            mask_x = inside & push_x_axis
            if np.any(mask_x):
                sign_dx = np.sign(dx)
                sign_dx = np.where(sign_dx == 0, 1.0, sign_dx)
                self.positions[mask_x, agent_idx, 0] += (overlap_x * sign_dx)[mask_x]
                self.velocities[mask_x, agent_idx, 0] = 0

            # Push along y axis
            mask_y = inside & ~push_x_axis
            if np.any(mask_y):
                sign_dy = np.sign(dy)
                sign_dy = np.where(sign_dy == 0, 1.0, sign_dy)
                self.positions[mask_y, agent_idx, 1] += (overlap_y * sign_dy)[mask_y]
                self.velocities[mask_y, agent_idx, 1] = 0

    def _update_safe_zone_state(self, dt: float):
        """Update safe zone state for all environments (vectorized)."""
        if self.safe_zone is None:
            return

        eids = self._eids
        hider_idx_arr = 1 - self.seeker_idx
        hider_pos = self.positions[eids, hider_idx_arr]
        in_zone = self._points_in_safe_zone(hider_pos)

        # Snapshot exhausted state before modifications (important for correct semantics:
        # an env that was exhausted at tick start only processes the cooldown branch)
        was_exhausted = self.safe_zone_exhausted.copy()

        # --- Exhausted envs: countdown cooldown ---
        self.safe_zone_cooldown[was_exhausted] -= dt

        # Those whose cooldown expired -> reset
        reset_mask = was_exhausted & (self.safe_zone_cooldown <= 0)
        self.safe_zone_exhausted[reset_mask] = False
        self.safe_zone_cooldown[reset_mask] = 0.0
        self.safe_zone_time[reset_mask] = 0.0

        # --- Non-exhausted envs (those that were NOT exhausted at start of tick) ---
        not_exhausted = ~was_exhausted

        # In zone and not exhausted: accumulate time
        accum_mask = not_exhausted & in_zone
        self.safe_zone_time[accum_mask] += dt

        # Check for newly exhausted
        newly_exhausted = accum_mask & (self.safe_zone_time >= self.safe_zone.protection_duration)
        self.safe_zone_exhausted[newly_exhausted] = True
        self.safe_zone_cooldown[newly_exhausted] = self.safe_zone.cooldown_duration

        # Left zone and not exhausted: reset time
        left_zone = not_exhausted & ~in_zone
        self.safe_zone_time[left_zone] = 0.0

    def auto_reset(self) -> Dict[str, np.ndarray]:
        """Reset done environments and return new observations."""
        done_ids = np.where(self.dones)[0]
        if len(done_ids) > 0:
            self.reset(done_ids)
        return self._get_obs()


class SingleTagEnv:
    """
    Single-environment wrapper for evaluation/debugging.
    """

    def __init__(self, config: Optional[TagEnvConfig] = None):
        self.vec_env = VecTagEnv(num_envs=1, config=config)
        self.obs_dim = self.vec_env.obs_dim
        self.act_dim = self.vec_env.act_dim

    def reset(self) -> Dict[str, np.ndarray]:
        obs = self.vec_env.reset()
        return {k: v[0] for k, v in obs.items()}

    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray],
                                                             Dict[str, float],
                                                             bool,
                                                             Dict[str, Any]]:
        vec_actions = {k: v[np.newaxis] for k, v in actions.items()}
        obs, rewards, dones, infos = self.vec_env.step(vec_actions)
        return (
            {k: v[0] for k, v in obs.items()},
            {k: float(v[0]) for k, v in rewards.items()},
            bool(dones[0]),
            {k: (v[0] if isinstance(v, np.ndarray) and v.ndim > 0 else v) for k, v in infos.items()}
        )

    def get_state(self) -> Dict[str, Any]:
        """Get full state for visualization."""
        state = {
            'positions': self.vec_env.positions[0].copy(),
            'velocities': self.vec_env.velocities[0].copy(),
            'facing': self.vec_env.facing[0].copy(),
            'seeker_idx': int(self.vec_env.seeker_idx[0]),
            'time_elapsed': float(self.vec_env.time_elapsed[0]),
            'obstacles': self.vec_env.obstacles,
            'safe_zone': self.vec_env.safe_zone,
            'safe_zone_time': float(self.vec_env.safe_zone_time[0]),
            'safe_zone_exhausted': bool(self.vec_env.safe_zone_exhausted[0]),
            'safe_zone_cooldown': float(self.vec_env.safe_zone_cooldown[0]),
            'stamina': self.vec_env.stamina[0].copy(),
        }
        return state


if __name__ == "__main__":
    # Quick test
    import time

    num_envs = 64
    env = VecTagEnv(num_envs=num_envs)

    obs = env.reset()
    print(f"Observation shape: seeker={obs['seeker'].shape}, hider={obs['hider'].shape}")
    print(f"Obs dim: {env.obs_dim}, Act dim: {env.act_dim}")

    # Benchmark
    n_steps = 10000
    start = time.time()

    for _ in range(n_steps):
        actions = {
            'seeker': np.random.uniform(-1, 1, (num_envs, 3)).astype(np.float32),
            'hider': np.random.uniform(-1, 1, (num_envs, 3)).astype(np.float32),
        }
        obs, rewards, dones, infos = env.step(actions)
        obs = env.auto_reset()

    elapsed = time.time() - start
    total_steps = n_steps * num_envs
    print(f"\nBenchmark: {total_steps} steps in {elapsed:.2f}s")
    print(f"Throughput: {total_steps / elapsed:.0f} steps/sec")
    print(f"Seeker wins: {infos.get('seeker_wins', 0)}, Hider wins: {infos.get('hider_wins', 0)}")
