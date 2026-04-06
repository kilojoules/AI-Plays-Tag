"""
Competitive Ant Sumo environment for SAC self-play.

Two Ant agents compete on a circular tatami mat. Win by pushing
the opponent outside the arena boundary. Inspired by OpenAI's
RoboSumo but built on modern gymnasium/MuJoCo.

Interface matches VecTagEnv: step() returns {seeker: obs, hider: obs}
dicts so the same SAC trainer and intrinsic rewards work.
"""
from __future__ import annotations

import numpy as np
import mujoco
import os
import tempfile


# MuJoCo XML for two ants in a shared arena
SUMO_XML = """
<mujoco model="ant_sumo">
  <option timestep="0.01" integrator="RK4"/>
  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom friction="1 0.5 0.5" density="5.0" margin="0.01" condim="3"/>
  </default>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texrepeat="5 5" texuniform="true" reflectance="0.2"/>
  </asset>

  <worldbody>
    <light pos="0 0 5" dir="0 0 -1" diffuse="1 1 1"/>
    <geom name="floor" type="plane" size="10 10 0.1" material="groundplane"/>

    <!-- Agent 0 (red) -->
    <body name="agent0" pos="-1.0 0 0.75">
      <freejoint name="agent0_root"/>
      <geom name="agent0_torso" type="sphere" size="0.25" rgba="0.8 0.2 0.2 1"/>
      <body name="agent0_leg0" pos="0.2 0.2 0">
        <joint name="agent0_hip0" type="hinge" axis="0 0 1" range="-30 30"/>
        <geom name="agent0_leg0_geom" type="capsule" fromto="0 0 0 0.4 0.4 0" size="0.08" rgba="0.8 0.2 0.2 1"/>
        <body name="agent0_ankle0" pos="0.4 0.4 0">
          <joint name="agent0_ankle0" type="hinge" axis="-1 1 0" range="30 70"/>
          <geom name="agent0_ankle0_geom" type="capsule" fromto="0 0 0 0 0 -0.5" size="0.08" rgba="0.8 0.2 0.2 1"/>
        </body>
      </body>
      <body name="agent0_leg1" pos="-0.2 0.2 0">
        <joint name="agent0_hip1" type="hinge" axis="0 0 1" range="-30 30"/>
        <geom name="agent0_leg1_geom" type="capsule" fromto="0 0 0 -0.4 0.4 0" size="0.08" rgba="0.8 0.2 0.2 1"/>
        <body name="agent0_ankle1" pos="-0.4 0.4 0">
          <joint name="agent0_ankle1" type="hinge" axis="1 1 0" range="-70 -30"/>
          <geom name="agent0_ankle1_geom" type="capsule" fromto="0 0 0 0 0 -0.5" size="0.08" rgba="0.8 0.2 0.2 1"/>
        </body>
      </body>
      <body name="agent0_leg2" pos="-0.2 -0.2 0">
        <joint name="agent0_hip2" type="hinge" axis="0 0 1" range="-30 30"/>
        <geom name="agent0_leg2_geom" type="capsule" fromto="0 0 0 -0.4 -0.4 0" size="0.08" rgba="0.8 0.2 0.2 1"/>
        <body name="agent0_ankle2" pos="-0.4 -0.4 0">
          <joint name="agent0_ankle2" type="hinge" axis="-1 -1 0" range="-70 -30"/>
          <geom name="agent0_ankle2_geom" type="capsule" fromto="0 0 0 0 0 -0.5" size="0.08" rgba="0.8 0.2 0.2 1"/>
        </body>
      </body>
      <body name="agent0_leg3" pos="0.2 -0.2 0">
        <joint name="agent0_hip3" type="hinge" axis="0 0 1" range="-30 30"/>
        <geom name="agent0_leg3_geom" type="capsule" fromto="0 0 0 0.4 -0.4 0" size="0.08" rgba="0.8 0.2 0.2 1"/>
        <body name="agent0_ankle3" pos="0.4 -0.4 0">
          <joint name="agent0_ankle3" type="hinge" axis="1 -1 0" range="30 70"/>
          <geom name="agent0_ankle3_geom" type="capsule" fromto="0 0 0 0 0 -0.5" size="0.08" rgba="0.8 0.2 0.2 1"/>
        </body>
      </body>
    </body>

    <!-- Agent 1 (blue) — identical structure, mirrored start position -->
    <body name="agent1" pos="1.0 0 0.75">
      <freejoint name="agent1_root"/>
      <geom name="agent1_torso" type="sphere" size="0.25" rgba="0.2 0.2 0.8 1"/>
      <body name="agent1_leg0" pos="0.2 0.2 0">
        <joint name="agent1_hip0" type="hinge" axis="0 0 1" range="-30 30"/>
        <geom name="agent1_leg0_geom" type="capsule" fromto="0 0 0 0.4 0.4 0" size="0.08" rgba="0.2 0.2 0.8 1"/>
        <body name="agent1_ankle0" pos="0.4 0.4 0">
          <joint name="agent1_ankle0" type="hinge" axis="-1 1 0" range="30 70"/>
          <geom name="agent1_ankle0_geom" type="capsule" fromto="0 0 0 0 0 -0.5" size="0.08" rgba="0.2 0.2 0.8 1"/>
        </body>
      </body>
      <body name="agent1_leg1" pos="-0.2 0.2 0">
        <joint name="agent1_hip1" type="hinge" axis="0 0 1" range="-30 30"/>
        <geom name="agent1_leg1_geom" type="capsule" fromto="0 0 0 -0.4 0.4 0" size="0.08" rgba="0.2 0.2 0.8 1"/>
        <body name="agent1_ankle1" pos="-0.4 0.4 0">
          <joint name="agent1_ankle1" type="hinge" axis="1 1 0" range="-70 -30"/>
          <geom name="agent1_ankle1_geom" type="capsule" fromto="0 0 0 0 0 -0.5" size="0.08" rgba="0.2 0.2 0.8 1"/>
        </body>
      </body>
      <body name="agent1_leg2" pos="-0.2 -0.2 0">
        <joint name="agent1_hip2" type="hinge" axis="0 0 1" range="-30 30"/>
        <geom name="agent1_leg2_geom" type="capsule" fromto="0 0 0 -0.4 -0.4 0" size="0.08" rgba="0.2 0.2 0.8 1"/>
        <body name="agent1_ankle2" pos="-0.4 -0.4 0">
          <joint name="agent1_ankle2" type="hinge" axis="-1 -1 0" range="-70 -30"/>
          <geom name="agent1_ankle2_geom" type="capsule" fromto="0 0 0 0 0 -0.5" size="0.08" rgba="0.2 0.2 0.8 1"/>
        </body>
      </body>
      <body name="agent1_leg3" pos="0.2 -0.2 0">
        <joint name="agent1_hip3" type="hinge" axis="0 0 1" range="-30 30"/>
        <geom name="agent1_leg3_geom" type="capsule" fromto="0 0 0 0.4 -0.4 0" size="0.08" rgba="0.2 0.2 0.8 1"/>
        <body name="agent1_ankle3" pos="0.4 -0.4 0">
          <joint name="agent1_ankle3" type="hinge" axis="1 -1 0" range="30 70"/>
          <geom name="agent1_ankle3_geom" type="capsule" fromto="0 0 0 0 0 -0.5" size="0.08" rgba="0.2 0.2 0.8 1"/>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor joint="agent0_hip0" ctrlrange="-1 1" gear="150"/>
    <motor joint="agent0_ankle0" ctrlrange="-1 1" gear="150"/>
    <motor joint="agent0_hip1" ctrlrange="-1 1" gear="150"/>
    <motor joint="agent0_ankle1" ctrlrange="-1 1" gear="150"/>
    <motor joint="agent0_hip2" ctrlrange="-1 1" gear="150"/>
    <motor joint="agent0_ankle2" ctrlrange="-1 1" gear="150"/>
    <motor joint="agent0_hip3" ctrlrange="-1 1" gear="150"/>
    <motor joint="agent0_ankle3" ctrlrange="-1 1" gear="150"/>
    <motor joint="agent1_hip0" ctrlrange="-1 1" gear="150"/>
    <motor joint="agent1_ankle0" ctrlrange="-1 1" gear="150"/>
    <motor joint="agent1_hip1" ctrlrange="-1 1" gear="150"/>
    <motor joint="agent1_ankle1" ctrlrange="-1 1" gear="150"/>
    <motor joint="agent1_hip2" ctrlrange="-1 1" gear="150"/>
    <motor joint="agent1_ankle2" ctrlrange="-1 1" gear="150"/>
    <motor joint="agent1_hip3" ctrlrange="-1 1" gear="150"/>
    <motor joint="agent1_ankle3" ctrlrange="-1 1" gear="150"/>
  </actuator>
</mujoco>
"""

# Arena radius (units)
ARENA_RADIUS = 3.5
# Win/loss rewards
WIN_REWARD = 10.0
DRAW_PENALTY = -5.0
# Shaping
MOVE_TO_OPP_COEF = 0.1
ALIVE_BONUS = 0.1
CTRL_COST_COEF = 0.001
# Episode length
MAX_STEPS = 500
# Fall threshold (z-height of torso)
FALL_HEIGHT = 0.3


class AntSumoEnv:
    """Single competitive Ant Sumo environment.

    Two ant agents on a circular arena. Win by pushing opponent out
    or knocking them down. Interface matches VecTagEnv conventions.
    """

    def __init__(self):
        # Write XML to temp file and load
        self._xml_path = os.path.join(tempfile.gettempdir(), "ant_sumo.xml")
        with open(self._xml_path, 'w') as f:
            f.write(SUMO_XML)

        self.model = mujoco.MjModel.from_xml_path(self._xml_path)
        self.data = mujoco.MjData(self.model)

        # Agent body IDs
        self._agent0_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "agent0")
        self._agent1_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "agent1")

        # Each agent has 8 actuators
        self.act_dim = 8
        # Observation: own qpos/qvel (15) + relative opponent pos/vel (6) + arena info (2)
        self.obs_dim = 23

        self._step_count = 0

    def _get_agent_pos(self, agent_id):
        """Get agent torso XYZ position."""
        body_id = self._agent0_body if agent_id == 0 else self._agent1_body
        return self.data.xpos[body_id].copy()

    def _get_agent_vel(self, agent_id):
        """Get agent torso velocity."""
        body_id = self._agent0_body if agent_id == 0 else self._agent1_body
        return self.data.cvel[body_id, 3:6].copy()  # linear velocity

    def _get_obs(self, agent_id):
        """Build observation for one agent."""
        # Own state: joint positions and velocities
        if agent_id == 0:
            qpos = self.data.qpos[:15].copy()  # first agent's 7 root + 8 joints
            qvel = self.data.qvel[:14].copy()   # first agent's 6 root + 8 joint vels
        else:
            qpos = self.data.qpos[15:].copy()
            qvel = self.data.qvel[14:].copy()

        own_pos = self._get_agent_pos(agent_id)
        opp_pos = self._get_agent_pos(1 - agent_id)
        own_vel = self._get_agent_vel(agent_id)
        opp_vel = self._get_agent_vel(1 - agent_id)

        # Relative opponent
        rel_pos = (opp_pos - own_pos)[:2]  # XY only
        rel_vel = (opp_vel - own_vel)[:2]

        # Arena info
        dist_to_center = np.linalg.norm(own_pos[:2])
        dist_to_boundary = ARENA_RADIUS - dist_to_center

        # Compact observation
        obs = np.concatenate([
            own_pos[:3] / ARENA_RADIUS,          # 3: normalized position
            own_vel[:3],                           # 3: velocity
            qpos[7:15] / np.pi,                   # 8: joint angles normalized
            rel_pos / ARENA_RADIUS,               # 2: relative opponent XY
            rel_vel,                               # 2: relative opponent vel
            [dist_to_boundary / ARENA_RADIUS],    # 1: distance to arena edge
            [own_pos[2]],                          # 1: own height (fall detection)
            [self._step_count / MAX_STEPS],        # 1: episode progress
            [np.linalg.norm(rel_pos)],             # 1: opponent distance
            [0.0],                                 # 1: padding to obs_dim=23
        ]).astype(np.float32)

        return obs[:self.obs_dim]

    def reset(self):
        """Reset environment."""
        mujoco.mj_resetData(self.model, self.data)

        # Randomize starting positions slightly
        self.data.qpos[0] = -1.0 + np.random.uniform(-0.2, 0.2)  # agent0 x
        self.data.qpos[1] = np.random.uniform(-0.2, 0.2)          # agent0 y
        self.data.qpos[15] = 1.0 + np.random.uniform(-0.2, 0.2)  # agent1 x
        self.data.qpos[16] = np.random.uniform(-0.2, 0.2)         # agent1 y

        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0

        return {
            'seeker': self._get_obs(0),
            'hider': self._get_obs(1),
        }

    def step(self, actions):
        """Step environment with actions dict."""
        # Set actuator controls
        act0 = np.clip(actions['seeker'], -1, 1)
        act1 = np.clip(actions['hider'], -1, 1)
        self.data.ctrl[:8] = act0
        self.data.ctrl[8:] = act1

        # Step physics (5 substeps)
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        # Get positions
        pos0 = self._get_agent_pos(0)
        pos1 = self._get_agent_pos(1)

        dist0 = np.linalg.norm(pos0[:2])  # distance from center
        dist1 = np.linalg.norm(pos1[:2])

        # Check termination
        out0 = dist0 > ARENA_RADIUS
        out1 = dist1 > ARENA_RADIUS
        fall0 = pos0[2] < FALL_HEIGHT
        fall1 = pos1[2] < FALL_HEIGHT
        timeout = self._step_count >= MAX_STEPS

        done = out0 or out1 or fall0 or fall1 or timeout

        # Rewards
        r0 = 0.0
        r1 = 0.0

        if out1 or fall1:  # agent0 wins
            r0 += WIN_REWARD
            r1 -= WIN_REWARD
        elif out0 or fall0:  # agent1 wins
            r0 -= WIN_REWARD
            r1 += WIN_REWARD
        elif timeout:
            r0 += DRAW_PENALTY
            r1 += DRAW_PENALTY

        # Shaping: move toward opponent
        opp_dist = np.linalg.norm(pos1[:2] - pos0[:2])
        r0 -= MOVE_TO_OPP_COEF * opp_dist
        r1 -= MOVE_TO_OPP_COEF * opp_dist

        # Alive bonus
        r0 += ALIVE_BONUS
        r1 += ALIVE_BONUS

        # Control cost
        r0 -= CTRL_COST_COEF * np.sum(act0 ** 2)
        r1 -= CTRL_COST_COEF * np.sum(act1 ** 2)

        # Determine who was "tagged" (lost)
        tagged = out0 or fall0  # agent0 lost = "seeker lost"

        obs = {
            'seeker': self._get_obs(0),
            'hider': self._get_obs(1),
        }
        rewards = {
            'seeker': np.float32(r0),
            'hider': np.float32(r1),
        }
        infos = {
            'tagged': tagged,
            'timed_out': timeout and not (out0 or out1 or fall0 or fall1),
            'hider_near_wall_frac': float(dist1 > ARENA_RADIUS * 0.7),
            'hider_corner_frac': 0.0,  # no corners in circular arena
            'hider_speed_mean': float(np.linalg.norm(self._get_agent_vel(1))),
            'hider_wall_speed_mean': 0.0,
            'seeker_speed_mean': float(np.linalg.norm(self._get_agent_vel(0))),
            'hider_wall_dist_mean': float(ARENA_RADIUS - dist1),
        }

        return obs, rewards, done, infos


class VecAntSumo:
    """Vectorized wrapper over multiple AntSumoEnv instances.

    Same interface as VecTagEnv for trainer compatibility.
    """

    def __init__(self, num_envs: int = 16):
        self.num_envs = num_envs
        self._envs = [AntSumoEnv() for _ in range(num_envs)]
        self.obs_dim = self._envs[0].obs_dim
        self.act_dim = self._envs[0].act_dim
        self.dones = np.zeros(num_envs, dtype=bool)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = range(self.num_envs)
        all_obs = {
            'seeker': np.zeros((self.num_envs, self.obs_dim), dtype=np.float32),
            'hider': np.zeros((self.num_envs, self.obs_dim), dtype=np.float32),
        }
        for i in env_ids:
            obs = self._envs[i].reset()
            all_obs['seeker'][i] = obs['seeker']
            all_obs['hider'][i] = obs['hider']
            self.dones[i] = False
        return all_obs

    def step(self, actions):
        all_obs = {
            'seeker': np.zeros((self.num_envs, self.obs_dim), dtype=np.float32),
            'hider': np.zeros((self.num_envs, self.obs_dim), dtype=np.float32),
        }
        all_rewards = {
            'seeker': np.zeros(self.num_envs, dtype=np.float32),
            'hider': np.zeros(self.num_envs, dtype=np.float32),
        }
        all_dones = np.zeros(self.num_envs, dtype=bool)
        all_tagged = np.zeros(self.num_envs, dtype=bool)

        wall_fracs, speeds = [], []

        for i in range(self.num_envs):
            if self.dones[i]:
                continue
            act_i = {
                'seeker': actions['seeker'][i],
                'hider': actions['hider'][i],
            }
            obs, rew, done, info = self._envs[i].step(act_i)
            all_obs['seeker'][i] = obs['seeker']
            all_obs['hider'][i] = obs['hider']
            all_rewards['seeker'][i] = rew['seeker']
            all_rewards['hider'][i] = rew['hider']
            all_dones[i] = done
            all_tagged[i] = info['tagged']
            wall_fracs.append(info['hider_near_wall_frac'])
            speeds.append(info['hider_speed_mean'])

        self.dones = all_dones.copy()

        infos = {
            'tagged': all_tagged,
            'timed_out': all_dones & ~all_tagged,
            'hider_near_wall_frac': float(np.mean(wall_fracs)) if wall_fracs else 0,
            'hider_corner_frac': 0.0,
            'hider_speed_mean': float(np.mean(speeds)) if speeds else 0,
            'hider_wall_speed_mean': 0.0,
            'seeker_speed_mean': 0.0,
            'hider_wall_dist_mean': 0.0,
        }

        return all_obs, all_rewards, all_dones, infos

    def auto_reset(self):
        done_ids = np.where(self.dones)[0]
        if len(done_ids) > 0:
            self.reset(done_ids)
        # Return current obs
        all_obs = {
            'seeker': np.zeros((self.num_envs, self.obs_dim), dtype=np.float32),
            'hider': np.zeros((self.num_envs, self.obs_dim), dtype=np.float32),
        }
        for i in range(self.num_envs):
            obs = self._envs[i].reset() if self.dones[i] else {
                'seeker': self._envs[i]._get_obs(0),
                'hider': self._envs[i]._get_obs(1),
            }
            all_obs['seeker'][i] = obs['seeker']
            all_obs['hider'][i] = obs['hider']
        return all_obs
