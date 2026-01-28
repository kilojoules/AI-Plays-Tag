#!/usr/bin/env python3
"""
Fast vectorized Python-only tag environment for efficient training.

This bypasses the Godot WebSocket bridge to enable rapid policy iteration.
The environment simulates simplified 2D tag game physics matching the Godot version.
"""
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
    steps_per_action: int = 3      # Physics steps per action (matches Godot)

    # Observation config
    num_rays: int = 36             # Number of vision rays
    ray_fov: float = 120.0         # Field of view in degrees
    ray_max_dist: float = 15.0     # Max ray distance

    # Reward shaping (matches Godot rl_env.gd)
    distance_reward_scale: float = 0.14
    seeker_time_penalty: float = -0.005
    runner_survival_bonus: float = 0.01
    win_bonus: float = 10.0
    timeout_hider_bonus: float = 6.0
    timeout_seeker_penalty: float = 6.0

    # Arena layout
    layout: str = 'empty'          # Layout name: 'empty', 'four_corners', 'central_cross'


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

        # Observation dimension: position(2) + velocity(2) + relative(4) + roles(2) + forward(2) + rays(72) + safe_zone(3)
        self.obs_dim = 12 + self.cfg.num_rays * 2 + 3
        self.act_dim = 3  # move_x, move_z, jump (jump ignored in 2D)

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

        # Compute initial distances
        self.prev_distances[env_ids] = self._compute_distances(env_ids)

        return self._get_obs()

    def _point_in_obstacle(self, pos: np.ndarray, radius: float = 0.0) -> bool:
        """Check if a point (with optional radius) overlaps any obstacle."""
        for obs in self.obstacles:
            # Expand obstacle by agent radius (Minkowski sum)
            expanded_hw = obs.half_width + radius
            expanded_hh = obs.half_height + radius
            dx = abs(pos[0] - obs.x)
            dy = abs(pos[1] - obs.y)
            if dx < expanded_hw and dy < expanded_hh:
                return True
        return False

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
        """
        obs = {}

        for role in ['seeker', 'hider']:
            role_obs = np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)

            for eid in range(self.num_envs):
                if role == 'seeker':
                    agent_idx = self.seeker_idx[eid]
                    other_idx = 1 - agent_idx
                    is_seeker = True
                else:
                    agent_idx = 1 - self.seeker_idx[eid]
                    other_idx = self.seeker_idx[eid]
                    is_seeker = False

                pos = self.positions[eid, agent_idx]
                vel = self.velocities[eid, agent_idx]
                other_pos = self.positions[eid, other_idx]
                other_vel = self.velocities[eid, other_idx]
                facing = self.facing[eid, agent_idx]

                # Normalized position
                obs_idx = 0
                role_obs[eid, obs_idx:obs_idx+2] = pos / self.cfg.arena_half
                obs_idx += 2

                # Normalized velocity
                role_obs[eid, obs_idx:obs_idx+2] = vel / 10.0
                obs_idx += 2

                # Relative opponent position and velocity
                role_obs[eid, obs_idx:obs_idx+2] = (other_pos - pos) / self.cfg.arena_half
                obs_idx += 2
                role_obs[eid, obs_idx:obs_idx+2] = (other_vel - vel) / 10.0
                obs_idx += 2

                # Role flags
                role_obs[eid, obs_idx] = 1.0 if is_seeker else 0.0
                role_obs[eid, obs_idx + 1] = 0.0 if is_seeker else 1.0
                obs_idx += 2

                # Forward direction
                role_obs[eid, obs_idx] = np.cos(facing)
                role_obs[eid, obs_idx + 1] = np.sin(facing)
                obs_idx += 2

                # Ray-based vision
                rays = self._cast_rays(eid, agent_idx)
                role_obs[eid, obs_idx:obs_idx + self.cfg.num_rays * 2] = rays
                obs_idx += self.cfg.num_rays * 2

                # Safe zone state (3 values)
                # Both agents observe the hider's safe zone state
                if self.safe_zone is not None:
                    hider_idx = 1 - self.seeker_idx[eid]
                    hider_pos = self.positions[eid, hider_idx]
                    in_zone = self._point_in_safe_zone(hider_pos)
                    role_obs[eid, obs_idx] = 1.0 if in_zone else 0.0
                    role_obs[eid, obs_idx + 1] = 1.0 if self.safe_zone_exhausted[eid] else 0.0
                    # Normalize cooldown to [0, 1]
                    role_obs[eid, obs_idx + 2] = self.safe_zone_cooldown[eid] / self.safe_zone.cooldown_duration
                else:
                    role_obs[eid, obs_idx:obs_idx + 3] = 0.0

            obs[role] = role_obs

        return obs

    def _point_in_safe_zone(self, pos: np.ndarray) -> bool:
        """Check if a point is inside the safe zone."""
        if self.safe_zone is None:
            return False
        dx = pos[0] - self.safe_zone.x
        dy = pos[1] - self.safe_zone.y
        return (dx * dx + dy * dy) <= (self.safe_zone.radius * self.safe_zone.radius)

    def _cast_rays(self, eid: int, agent_idx: int) -> np.ndarray:
        """Cast vision rays for an agent, returns [dist, hit_type] pairs.

        hit_type: 0.0 = wall, 0.5 = obstacle, 1.0 = agent
        """
        result = np.zeros(self.cfg.num_rays * 2, dtype=np.float32)

        pos = self.positions[eid, agent_idx]
        facing = self.facing[eid, agent_idx]
        other_idx = 1 - agent_idx
        other_pos = self.positions[eid, other_idx]

        half_fov = np.radians(self.cfg.ray_fov / 2)
        angles = np.linspace(-half_fov, half_fov, self.cfg.num_rays) + facing

        for i, angle in enumerate(angles):
            direction = np.array([np.cos(angle), np.sin(angle)])

            # Check wall intersections
            wall_dist = self._ray_wall_distance(pos, direction)

            # Check agent intersection
            agent_dist = self._ray_circle_distance(pos, direction, other_pos, radius=0.5)

            # Check obstacle intersections
            obstacle_dist = self.cfg.ray_max_dist
            for obs in self.obstacles:
                obs_dist = self._ray_aabb_distance(pos, direction, obs)
                obstacle_dist = min(obstacle_dist, obs_dist)

            # Determine closest hit and type
            min_dist = wall_dist
            hit_type = 0.0  # wall

            if obstacle_dist < min_dist:
                min_dist = obstacle_dist
                hit_type = 0.5  # obstacle

            if agent_dist < min_dist:
                min_dist = agent_dist
                hit_type = 1.0  # agent

            result[i * 2] = min(min_dist / self.cfg.ray_max_dist, 1.0)
            result[i * 2 + 1] = hit_type

        return result

    def _ray_aabb_distance(self, pos: np.ndarray, direction: np.ndarray, obs: Obstacle) -> float:
        """Compute distance to axis-aligned bounding box using slab method."""
        # Box bounds
        box_min = np.array([obs.x - obs.half_width, obs.y - obs.half_height])
        box_max = np.array([obs.x + obs.half_width, obs.y + obs.half_height])

        # Avoid division by zero
        inv_dir = np.where(np.abs(direction) > 1e-8, 1.0 / direction, np.sign(direction) * 1e8)

        # Compute intersection distances for each slab
        t1 = (box_min - pos) * inv_dir
        t2 = (box_max - pos) * inv_dir

        # Find the near and far intersection for each axis
        t_min = np.minimum(t1, t2)
        t_max = np.maximum(t1, t2)

        # The ray enters the box when it has entered all slabs
        t_enter = np.max(t_min)
        # The ray exits the box when it exits any slab
        t_exit = np.min(t_max)

        # Check for valid intersection
        if t_enter < t_exit and t_exit > 0:
            # Return the entry point (or 0 if we start inside)
            return max(t_enter, 0.0) if t_enter < self.cfg.ray_max_dist else self.cfg.ray_max_dist

        return self.cfg.ray_max_dist

    def _ray_wall_distance(self, pos: np.ndarray, direction: np.ndarray) -> float:
        """Compute distance to arena wall along ray direction."""
        half = self.cfg.arena_half
        min_dist = self.cfg.ray_max_dist

        # Check all 4 walls
        for wall_pos, wall_normal in [
            (half, np.array([1, 0])),   # Right wall
            (-half, np.array([-1, 0])), # Left wall
            (half, np.array([0, 1])),   # Top wall
            (-half, np.array([0, -1])), # Bottom wall
        ]:
            # Wall is at position wall_pos along the normal axis
            axis = 0 if wall_normal[0] != 0 else 1
            if abs(direction[axis]) > 1e-6:
                t = (wall_pos - pos[axis]) / direction[axis]
                if 0 < t < min_dist:
                    # Check if intersection is within wall bounds
                    hit_pos = pos + direction * t
                    other_axis = 1 - axis
                    if abs(hit_pos[other_axis]) <= half:
                        min_dist = t

        return min_dist

    def _ray_circle_distance(self, pos: np.ndarray, direction: np.ndarray,
                             circle_pos: np.ndarray, radius: float) -> float:
        """Compute distance to circle along ray direction."""
        to_circle = circle_pos - pos
        proj = np.dot(to_circle, direction)

        if proj < 0:
            return self.cfg.ray_max_dist

        closest = pos + direction * proj
        dist_to_center = np.linalg.norm(circle_pos - closest)

        if dist_to_center > radius:
            return self.cfg.ray_max_dist

        # Ray hits the circle
        half_chord = np.sqrt(radius**2 - dist_to_center**2)
        hit_dist = proj - half_chord

        return max(0, hit_dist) if hit_dist < self.cfg.ray_max_dist else self.cfg.ray_max_dist

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

        # Map role actions to agent indices
        agent_actions = np.zeros((self.num_envs, 2, 3), dtype=np.float32)
        for eid in range(self.num_envs):
            seeker_idx = self.seeker_idx[eid]
            hider_idx = 1 - seeker_idx
            agent_actions[eid, seeker_idx] = seeker_actions[eid]
            agent_actions[eid, hider_idx] = hider_actions[eid]

        # Simulate physics for multiple substeps
        dt = self.cfg.dt
        for _ in range(self.cfg.steps_per_action):
            for agent_idx in range(2):
                # Get move inputs (clamped to [-1, 1])
                move_input = np.clip(agent_actions[:, agent_idx, :2], -1, 1)

                # Apply acceleration
                target_vel = move_input * self.cfg.agent_speed
                accel = (target_vel - self.velocities[:, agent_idx]) * self.cfg.agent_accel * dt
                self.velocities[:, agent_idx] += accel

                # Clamp velocity
                speed = np.linalg.norm(self.velocities[:, agent_idx], axis=1, keepdims=True)
                speed = np.maximum(speed, 1e-6)
                self.velocities[:, agent_idx] = np.where(
                    speed > self.cfg.agent_speed,
                    self.velocities[:, agent_idx] / speed * self.cfg.agent_speed,
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

                # Obstacle collision
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

        # Check safe zone protection for hider
        if self.safe_zone is not None:
            for eid in range(self.num_envs):
                if tagged[eid]:
                    hider_idx = 1 - self.seeker_idx[eid]
                    hider_pos = self.positions[eid, hider_idx]
                    in_zone = self._point_in_safe_zone(hider_pos)
                    # Hider is protected if in zone AND not exhausted
                    if in_zone and not self.safe_zone_exhausted[eid]:
                        tagged[eid] = False

        # Check for timeouts
        timed_out = self.time_elapsed >= self.cfg.time_limit

        # Compute rewards
        progress = self.prev_distances - distances  # Positive when closing gap

        seeker_rewards = np.zeros(self.num_envs, dtype=np.float32)
        hider_rewards = np.zeros(self.num_envs, dtype=np.float32)

        # Distance-based shaping
        seeker_rewards += progress * self.cfg.distance_reward_scale
        hider_rewards -= progress * self.cfg.distance_reward_scale

        # Time-based rewards
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

        # Build info dict
        infos = {
            'tagged': tagged.copy(),
            'timed_out': timed_out.copy(),
            'distances': distances.copy(),
            'time_elapsed': self.time_elapsed.copy(),
            'seeker_wins': tagged.sum(),
            'hider_wins': (timed_out & ~tagged).sum(),
        }

        # Get new observations
        obs = self._get_obs()

        rewards = {
            'seeker': seeker_rewards,
            'hider': hider_rewards,
        }

        return obs, rewards, self.dones.copy(), infos

    def _resolve_obstacle_collisions(self, agent_idx: int):
        """Push agents out of obstacles using AABB + Minkowski sum."""
        agent_radius = 0.5

        for obs in self.obstacles:
            # Expand obstacle by agent radius
            expanded_hw = obs.half_width + agent_radius
            expanded_hh = obs.half_height + agent_radius

            for eid in range(self.num_envs):
                pos = self.positions[eid, agent_idx]
                dx = pos[0] - obs.x
                dy = pos[1] - obs.y

                # Check if agent center is inside expanded box
                if abs(dx) < expanded_hw and abs(dy) < expanded_hh:
                    # Determine which axis has the smallest overlap
                    overlap_x = expanded_hw - abs(dx)
                    overlap_y = expanded_hh - abs(dy)

                    if overlap_x < overlap_y:
                        # Push out along x axis
                        push = overlap_x * np.sign(dx) if dx != 0 else overlap_x
                        self.positions[eid, agent_idx, 0] += push
                        self.velocities[eid, agent_idx, 0] = 0  # Stop velocity in collision direction
                    else:
                        # Push out along y axis
                        push = overlap_y * np.sign(dy) if dy != 0 else overlap_y
                        self.positions[eid, agent_idx, 1] += push
                        self.velocities[eid, agent_idx, 1] = 0

    def _update_safe_zone_state(self, dt: float):
        """Update safe zone state for all environments."""
        if self.safe_zone is None:
            return

        for eid in range(self.num_envs):
            hider_idx = 1 - self.seeker_idx[eid]
            hider_pos = self.positions[eid, hider_idx]
            in_zone = self._point_in_safe_zone(hider_pos)

            if self.safe_zone_exhausted[eid]:
                # In cooldown period - count down
                self.safe_zone_cooldown[eid] -= dt
                if self.safe_zone_cooldown[eid] <= 0:
                    # Cooldown complete - reset
                    self.safe_zone_exhausted[eid] = False
                    self.safe_zone_cooldown[eid] = 0.0
                    self.safe_zone_time[eid] = 0.0
            else:
                # Not exhausted
                if in_zone:
                    # Accumulate time in zone
                    self.safe_zone_time[eid] += dt
                    if self.safe_zone_time[eid] >= self.safe_zone.protection_duration:
                        # Exhausted - start cooldown
                        self.safe_zone_exhausted[eid] = True
                        self.safe_zone_cooldown[eid] = self.safe_zone.cooldown_duration
                else:
                    # Left zone - reset time (but NOT exhausted state or cooldown)
                    self.safe_zone_time[eid] = 0.0

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
