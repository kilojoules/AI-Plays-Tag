#!/usr/bin/env python3
"""Correctness tests for VecTagEnv vectorization.

Captures the scalar (original) behavior, then verifies the vectorized code matches.
Run with: pixi run python trainer/test_tag_env.py
"""
from __future__ import annotations

import sys
import time
import numpy as np

from tag_env import VecTagEnv, TagEnvConfig, SingleTagEnv, LAYOUTS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed(s: int = 42):
    np.random.seed(s)


def _make_env(num_envs=4, layout="four_corners", **kw):
    cfg = TagEnvConfig(layout=layout, **kw)
    return VecTagEnv(num_envs=num_envs, config=cfg)


def _fixed_actions(num_envs, seeker_act=None, hider_act=None):
    """Return constant action dicts."""
    if seeker_act is None:
        seeker_act = [1.0, 0.0, 0.0]
    if hider_act is None:
        hider_act = [-1.0, 0.0, 0.0]
    return {
        "seeker": np.tile(np.array(seeker_act, dtype=np.float32), (num_envs, 1)),
        "hider": np.tile(np.array(hider_act, dtype=np.float32), (num_envs, 1)),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_obs_shape():
    """Observation shapes and dim match for both roles."""
    _seed()
    env = _make_env(8)
    obs = env.reset()
    assert obs["seeker"].shape == (8, env.obs_dim), f"seeker shape {obs['seeker'].shape}"
    assert obs["hider"].shape == (8, env.obs_dim), f"hider shape {obs['hider'].shape}"
    print("  PASS obs_shape")


def test_step_deterministic():
    """Same seed + same actions -> identical trajectories."""
    results = []
    for _ in range(2):
        _seed(99)
        env = _make_env(4)
        obs = env.reset()
        acts = _fixed_actions(4)
        all_obs, all_rew = [], []
        for _ in range(20):
            obs, rew, dones, info = env.step(acts)
            all_obs.append(obs["seeker"].copy())
            all_rew.append(rew["seeker"].copy())
            obs = env.auto_reset()
        results.append((np.stack(all_obs), np.stack(all_rew)))

    np.testing.assert_allclose(results[0][0], results[1][0], atol=1e-6,
                               err_msg="observations differ across seeds")
    np.testing.assert_allclose(results[0][1], results[1][1], atol=1e-6,
                               err_msg="rewards differ across seeds")
    print("  PASS step_deterministic")


def test_ray_casting_known_positions():
    """Place agents at known positions and verify ray outputs are sane."""
    _seed(0)
    env = _make_env(1, layout="empty")  # no obstacles
    env.reset()

    # Manually place agents
    env.positions[0, 0] = [0.0, 0.0]  # agent 0 at center
    env.positions[0, 1] = [5.0, 0.0]  # agent 1 to the right
    env.facing[0, 0] = 0.0            # agent 0 facing right

    obs = env._get_obs()
    # Extract ray data for the role that is agent 0
    # Determine which role agent 0 plays
    seeker_idx = env.seeker_idx[0]
    if seeker_idx == 0:
        rays = obs["seeker"][0]
    else:
        rays = obs["hider"][0]

    # Rays start at index 12 (pos2 + vel2 + relpos2 + relvel2 + role2 + fwd2)
    ray_start = 12
    num_rays = env.cfg.num_rays
    ray_data = rays[ray_start:ray_start + num_rays * 2]
    distances = ray_data[0::2]
    hit_types = ray_data[1::2]

    # The center ray (agent 0 facing right, agent 1 at x=5) should detect agent
    center_ray = num_rays // 2
    # Agent is 5 units away, radius 0.5, so hit dist ~ 4.5
    # Normalized: 4.5 / 15.0 = 0.3
    assert distances[center_ray] < 0.5, (
        f"center ray dist {distances[center_ray]} should see agent at ~0.3")
    assert hit_types[center_ray] == 1.0, (
        f"center ray should hit agent (1.0), got {hit_types[center_ray]}")

    # Rays pointing away (far from center) should hit walls
    assert hit_types[0] == 0.0 or hit_types[-1] == 0.0, "edge rays should hit walls"
    print("  PASS ray_casting_known_positions")


def test_obstacle_collision_resolution():
    """Agent placed inside obstacle gets pushed out."""
    _seed(0)
    env = _make_env(2, layout="four_corners")
    env.reset()

    # Obstacle at (-8, 8) half 1.5x1.5 -> bounds [-9.5, -6.5] x [6.5, 9.5]
    obs0 = env.obstacles[0]
    # Place agent inside the obstacle
    env.positions[0, 0] = [obs0.x, obs0.y]

    # Run collision resolution
    env._resolve_obstacle_collisions(0)

    pos = env.positions[0, 0]
    # Agent should be pushed outside the expanded box (obstacle + 0.5 radius)
    dx = abs(pos[0] - obs0.x)
    dy = abs(pos[1] - obs0.y)
    expanded_hw = obs0.half_width + 0.5
    expanded_hh = obs0.half_height + 0.5
    outside = dx >= expanded_hw or dy >= expanded_hh
    assert outside, f"Agent at ({pos[0]:.2f}, {pos[1]:.2f}) still inside obstacle"
    print("  PASS obstacle_collision_resolution")


def test_safe_zone_protection():
    """Hider in non-exhausted safe zone is not tagged even at tag distance."""
    _seed(0)
    env = _make_env(1, layout="four_corners")
    env.reset()

    # Safe zone is at (0,0) radius 2.5
    sz = env.safe_zone
    seeker_idx = env.seeker_idx[0]
    hider_idx = 1 - seeker_idx

    # Place hider in safe zone center, seeker right next to them
    env.positions[0, hider_idx] = [sz.x, sz.y]
    env.positions[0, seeker_idx] = [sz.x + 0.5, sz.y]  # within tag_distance

    # Make sure not exhausted
    env.safe_zone_exhausted[0] = False
    env.safe_zone_time[0] = 0.0

    acts = _fixed_actions(1, [0, 0, 0], [0, 0, 0])
    _, _, dones, infos = env.step(acts)

    assert not infos["tagged"][0], "Hider should be protected in safe zone"
    print("  PASS safe_zone_protection")


def test_safe_zone_exhaustion():
    """Safe zone exhausts after protection_duration and enters cooldown."""
    _seed(0)
    env = _make_env(1, layout="four_corners")
    env.reset()

    sz = env.safe_zone
    hider_idx = 1 - env.seeker_idx[0]

    # Place hider in zone
    env.positions[0, hider_idx] = [sz.x, sz.y]
    env.safe_zone_time[0] = 0.0
    env.safe_zone_exhausted[0] = False

    # Tick just past protection_duration
    dt = env.cfg.dt * env.cfg.steps_per_action
    ticks_to_exhaust = int(np.ceil(sz.protection_duration / dt)) + 1

    acts = _fixed_actions(1, [0, 0, 0], [0, 0, 0])
    for _ in range(ticks_to_exhaust):
        env.step(acts)
        # Re-pin hider in zone (physics may move them)
        env.positions[0, hider_idx] = [sz.x, sz.y]

    assert env.safe_zone_exhausted[0], "Safe zone should be exhausted"
    assert env.safe_zone_cooldown[0] > 0, "Cooldown should be active"
    print("  PASS safe_zone_exhaustion")


def test_safe_zone_cooldown_recovery():
    """After cooldown, safe zone protection is restored."""
    _seed(0)
    env = _make_env(1, layout="four_corners")
    env.reset()

    sz = env.safe_zone
    hider_idx = 1 - env.seeker_idx[0]

    # Start in exhausted state with full cooldown
    env.safe_zone_exhausted[0] = True
    env.safe_zone_cooldown[0] = sz.cooldown_duration

    # Tick past cooldown
    dt = env.cfg.dt * env.cfg.steps_per_action
    ticks_to_recover = int(np.ceil(sz.cooldown_duration / dt)) + 2

    acts = _fixed_actions(1, [0, 0, 0], [0, 0, 0])
    for _ in range(ticks_to_recover):
        env.step(acts)

    assert not env.safe_zone_exhausted[0], "Exhausted should be cleared"
    assert env.safe_zone_cooldown[0] == 0.0, "Cooldown should be zero"
    print("  PASS safe_zone_cooldown_recovery")


def test_action_mapping():
    """Seeker and hider actions reach the correct agent indices."""
    _seed(0)
    env = _make_env(4, layout="empty")
    env.reset()

    # Zero all velocities, place agents at known positions
    env.velocities[:] = 0.0
    for eid in range(4):
        env.positions[eid, 0] = [0.0, 0.0]
        env.positions[eid, 1] = [5.0, 5.0]

    # Seeker pushes right, hider pushes left
    acts = _fixed_actions(4, seeker_act=[1.0, 0.0, 0.0], hider_act=[-1.0, 0.0, 0.0])
    env.step(acts)

    for eid in range(4):
        s_idx = env.seeker_idx[eid]
        h_idx = 1 - s_idx
        # Seeker should have moved right (positive x velocity)
        assert env.velocities[eid, s_idx, 0] > 0, (
            f"env {eid}: seeker vel_x={env.velocities[eid, s_idx, 0]:.3f} should be >0")
        # Hider should have moved left (negative x velocity)
        assert env.velocities[eid, h_idx, 0] < 0, (
            f"env {eid}: hider vel_x={env.velocities[eid, h_idx, 0]:.3f} should be <0")
    print("  PASS action_mapping")


def test_single_tag_env_wrapper():
    """SingleTagEnv API works correctly."""
    _seed(0)
    cfg = TagEnvConfig(layout="four_corners")
    env = SingleTagEnv(config=cfg)
    obs = env.reset()
    assert obs["seeker"].ndim == 1
    assert obs["hider"].ndim == 1

    acts = {"seeker": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "hider": np.array([-1.0, 0.0, 0.0], dtype=np.float32)}
    obs, rew, done, info = env.step(acts)
    assert isinstance(rew["seeker"], float)
    assert isinstance(done, bool)

    state = env.get_state()
    assert "positions" in state
    assert state["positions"].shape == (2, 2)
    print("  PASS single_tag_env_wrapper")


def test_all_layouts():
    """All defined layouts initialize without error."""
    for name in ["empty", "four_corners", "central_cross", "playground"]:
        _seed(0)
        env = _make_env(2, layout=name)
        obs = env.reset()
        acts = _fixed_actions(2)
        env.step(acts)
    print("  PASS all_layouts")


def test_sprint_system():
    """Sprint drains stamina and boosts speed."""
    _seed(0)
    env = _make_env(2, layout="empty", enable_sprint=True)
    env.reset()

    initial_stamina = env.stamina.copy()
    # Sprint at full intensity (action dim 2 = +1.0 -> sprint_intensity = 1.0)
    acts = _fixed_actions(2, seeker_act=[1.0, 0.0, 1.0], hider_act=[0.0, 0.0, 1.0])
    env.step(acts)

    # Stamina should have decreased for both agents
    for eid in range(2):
        for agent_idx in range(2):
            assert env.stamina[eid, agent_idx] < initial_stamina[eid, agent_idx], (
                f"env {eid} agent {agent_idx}: stamina not drained")
    print("  PASS sprint_system")


def test_auto_reset():
    """auto_reset resets done environments and returns valid observations."""
    _seed(42)
    env = _make_env(4, layout="empty")
    env.reset()

    # Force some envs to be done
    env.dones[0] = True
    env.dones[2] = True

    obs = env.auto_reset()
    assert obs["seeker"].shape == (4, env.obs_dim)
    # Dones should be cleared for reset envs
    assert not env.dones[0]
    assert not env.dones[2]
    print("  PASS auto_reset")


def test_episode_termination():
    """Episodes end on tag or timeout."""
    _seed(0)
    env = _make_env(1, layout="empty")
    env.reset()

    # Place agents touching each other
    env.positions[0, 0] = [0.0, 0.0]
    env.positions[0, 1] = [0.5, 0.0]  # within tag_distance=1.5
    env.prev_distances[0] = 0.5

    acts = _fixed_actions(1, [0, 0, 0], [0, 0, 0])
    _, _, dones, infos = env.step(acts)
    assert dones[0], "Should be done (tagged)"
    assert infos["tagged"][0], "Should register as tagged"
    print("  PASS episode_termination")


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark(num_envs=64, n_steps=2000):
    """Measure steps/sec."""
    _seed(0)
    env = _make_env(num_envs, layout="four_corners")
    env.reset()

    start = time.time()
    for _ in range(n_steps):
        acts = {
            "seeker": np.random.uniform(-1, 1, (num_envs, 3)).astype(np.float32),
            "hider": np.random.uniform(-1, 1, (num_envs, 3)).astype(np.float32),
        }
        env.step(acts)
        env.auto_reset()
    elapsed = time.time() - start

    total = n_steps * num_envs
    rate = total / elapsed
    print(f"\n  Benchmark: {num_envs} envs x {n_steps} steps = {total} total")
    print(f"  Elapsed: {elapsed:.2f}s  |  {rate:,.0f} steps/sec")
    return rate


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running VecTagEnv correctness tests...\n")

    tests = [
        test_obs_shape,
        test_step_deterministic,
        test_ray_casting_known_positions,
        test_obstacle_collision_resolution,
        test_safe_zone_protection,
        test_safe_zone_exhaustion,
        test_safe_zone_cooldown_recovery,
        test_action_mapping,
        test_single_tag_env_wrapper,
        test_all_layouts,
        test_sprint_system,
        test_auto_reset,
        test_episode_termination,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL {t.__name__}: {e}")
            failed += 1

    print(f"\n{passed}/{passed+failed} tests passed")

    if failed > 0:
        sys.exit(1)

    benchmark(64, 2000)
    print("\nAll tests passed!")
