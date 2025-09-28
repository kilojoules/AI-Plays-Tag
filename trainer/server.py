#!/usr/bin/env python3
"""
Production-ready-ish WebSocket bridge with a background Trainer.
Key changes:
- Dynamic obs_dim derived from first action request (vision-based obs).
- Trainer class manages buffers and updates outside the network I/O path.
"""
import asyncio
import json
import math
import os
import random
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import websockets

try:
    import torch
    from ppo import PPOAgent, PPOConfig
except Exception:
    torch = None
    PPOAgent = None
    PPOConfig = None


@dataclass
class EnvState:
    pos: np.ndarray  # (x, z)
    vel: np.ndarray  # (x, z)
    is_it: bool


def initial_state() -> EnvState:
    return EnvState(pos=np.zeros(2, dtype=np.float32), vel=np.zeros(2, dtype=np.float32), is_it=False)


def obs_from_state(s: EnvState) -> List[float]:
    return [float(s.pos[0]), float(s.pos[1]), float(s.vel[0]), float(s.vel[1]), 1.0 if s.is_it else 0.0]


def step_dynamics(s: EnvState, action: Tuple[float, float, float]) -> EnvState:
    dx, dz, jump = action
    acc = np.array([dx, dz], dtype=np.float32)
    s.vel = 0.85 * s.vel + 0.15 * acc * 5.0
    s.pos = s.pos + s.vel * 0.1
    return s


def compute_reward(s: EnvState) -> float:
    return float(np.linalg.norm(s.vel)) * 0.01


class Trainer:
    def __init__(self, batch_target: int = 2048) -> None:
        self.policy = None  # type: ignore
        self.cfg = None     # type: ignore
        self.batch_target = batch_target
        self.buf_obs: List[List[float]] = []
        self.buf_act: List[List[float]] = []
        self.buf_rew: List[float] = []
        self.buf_done: List[bool] = []
        self.updates_done = 0
        self.current_ep_rewards: List[float] = []
        self.episodes_logged = 0
        self.log_dir = os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.metrics_csv = os.path.join(self.log_dir, "metrics.csv")
        if not os.path.exists(self.metrics_csv):
            with open(self.metrics_csv, "w") as f:
                f.write("episode,reward_mean,reward_sum,steps,updates\n")
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore
            self.writer = SummaryWriter(self.log_dir)
        except Exception:
            self.writer = None
        self.lock = threading.Lock()

    def ensure_policy(self, obs_dim: int, act_dim: int = 3):
        if self.policy is not None or torch is None or PPOConfig is None or PPOAgent is None:
            return
        self.cfg = PPOConfig(obs_dim=obs_dim, act_dim=act_dim)
        self.policy = PPOAgent(self.cfg)
        # Optionally load pre-trained policy unless disabled
        if os.environ.get("DISABLE_POLICY_LOAD", "0").lower() in ("1", "true", "yes"):
            return
        policy_path = os.path.join(os.path.dirname(__file__), "policy.pt")
        try:
            if os.path.exists(policy_path):
                self.policy.load_policy(policy_path)
                print("Loaded policy.pt for action inference")
        except Exception as e:
            print("Policy load failed:", e)

    def act(self, obs: List[float]) -> List[float]:
        if self.policy is None:
            return [random.uniform(-1, 1), random.uniform(-1, 1), 0.0]
        act_np, _ = self.policy.act(np.array(obs, dtype=np.float32))
        return [float(act_np[0]), float(act_np[1]), float(act_np[2])]

    def add_transition(self, obs: List[float], act: List[float], rew: float, done: bool):
        with self.lock:
            self.buf_obs.append(obs)
            self.buf_act.append([float(act[0]), float(act[1]), float(act[2]) if len(act) > 2 else 0.0])
            self.buf_rew.append(float(rew))
            self.buf_done.append(bool(done))
            self.current_ep_rewards.append(float(rew))
            if done:
                self.log_episode()

    def add_transition_batch(self, transitions: List[Dict[str, Any]]):
        for tr in transitions:
            try:
                obs = tr.get("obs")
                action = tr.get("action")
                reward = float(tr.get("reward", 0.0))
                done = bool(tr.get("done", False))
                if isinstance(obs, list) and isinstance(action, list):
                    self.add_transition(obs, action, reward, done)
            except Exception:
                continue

    def maybe_update(self):
        if self.policy is None:
            return
        if len(self.buf_obs) < self.batch_target:
            return
        # Copy buffers under lock and clear
        with self.lock:
            obs = np.array(self.buf_obs, dtype=np.float32)
            act = np.array(self.buf_act, dtype=np.float32)
            rew = np.array(self.buf_rew, dtype=np.float32)
            done = np.array(self.buf_done, dtype=np.bool_)
            self.buf_obs, self.buf_act, self.buf_rew, self.buf_done = [], [], [], []
        try:
            # Compute value predictions
            with torch.no_grad():
                v = self.policy.vf(torch.as_tensor(obs, dtype=torch.float32)).squeeze(-1).numpy()
            # GAE-lambda advantages
            gamma, lam = 0.99, 0.95
            adv = np.zeros_like(rew)
            lastgaelam = 0.0
            for t in reversed(range(len(rew))):
                nextv = 0.0 if t == len(rew)-1 or done[t] else v[t+1]
                delta = rew[t] + gamma * nextv - v[t]
                lastgaelam = delta + gamma * lam * lastgaelam * (0.0 if done[t] else 1.0)
                adv[t] = lastgaelam
            ret = adv + v
            # Simple supervised-style PPO update placeholder
            for _ in range(10):
                idx = np.random.permutation(len(obs))
                for start in range(0, len(obs), 256):
                    j = idx[start:start+256]
                    o = torch.as_tensor(obs[j], dtype=torch.float32)
                    a = torch.as_tensor(act[j], dtype=torch.float32)
                    r = torch.as_tensor(ret[j], dtype=torch.float32)
                    adv_j = torch.as_tensor(adv[j], dtype=torch.float32)
                    logits = self.policy.pi(o)
                    pred = torch.tanh(logits)
                    pi_loss = ((pred - a) ** 2 * adv_j.unsqueeze(-1)).mean()
                    v_pred = self.policy.vf(o).squeeze(-1)
                    v_loss = ((v_pred - r) ** 2).mean()
                    self.policy.pi_opt.zero_grad(); pi_loss.backward(); self.policy.pi_opt.step()
                    self.policy.vf_opt.zero_grad(); v_loss.backward(); self.policy.vf_opt.step()
            torch.save(self.policy.pi.state_dict(), os.path.join(os.path.dirname(__file__), "policy.pt"))
            self.updates_done += 1
            print(f"Policy updated x{self.updates_done}; buffer cleared")
            if self.writer is not None:
                self.writer.add_scalar("updates/count", self.updates_done, self.updates_done)
        except Exception as e:
            print("Update failed:", e)

    def log_episode(self):
        if len(self.current_ep_rewards) == 0:
            return
        ep_sum = float(np.sum(self.current_ep_rewards))
        ep_mean = float(np.mean(self.current_ep_rewards))
        ep_steps = int(len(self.current_ep_rewards))
        self.episodes_logged += 1
        with open(self.metrics_csv, "a") as f:
            f.write(f"{self.episodes_logged},{ep_mean},{ep_sum},{ep_steps},{self.updates_done}\n")
        if self.writer is not None:
            self.writer.add_scalar("episode/reward_mean", ep_mean, self.episodes_logged)
            self.writer.add_scalar("episode/reward_sum", ep_sum, self.episodes_logged)
            self.writer.add_scalar("episode/steps", ep_steps, self.episodes_logged)
        self.current_ep_rewards = []


trainer = Trainer()


async def handle(ws: websockets.WebSocketServerProtocol):
    state = initial_state()
    await ws.send(json.dumps({"type": "observation", "obs": obs_from_state(state), "info": {}}))
    async for msg in ws:
        try:
            data = json.loads(msg)
        except json.JSONDecodeError:
            continue
        typ = data.get("type")
        if typ == "reset":
            state = initial_state()
            await ws.send(json.dumps({"type": "observation", "obs": obs_from_state(state), "info": {}}))
        elif typ == "step":
            action = data.get("action", [0.0, 0.0, 0.0])
            ax, az, aj = float(action[0]), float(action[1]), float(action[2])
            state = step_dynamics(state, (ax, az, aj))
            reward = compute_reward(state)
            done = False
            await ws.send(json.dumps({
                "type": "transition",
                "obs": obs_from_state(state),
                "reward": reward,
                "done": done,
                "info": {}
            }))
        elif typ == "act":
            obs = data.get("obs", obs_from_state(state))
            if isinstance(obs, list):
                trainer.ensure_policy(len(obs), act_dim=3)
            action = trainer.act(obs)
            await ws.send(json.dumps({
                "type": "action",
                "action": action,
                "info": {}
            }))
        elif typ == "act_batch":
            obs_dict = data.get("obs", {})
            actions = {}
            for name in obs_dict.keys():
                arr = obs_dict[name]
                if isinstance(arr, list):
                    trainer.ensure_policy(len(arr), act_dim=3)
                actions[name] = trainer.act(arr)
            await ws.send(json.dumps({
                "type": "action_batch",
                "actions": actions,
                "info": {}
            }))
        elif typ == "transition":
            obs = data.get("obs")
            action = data.get("action")
            reward = float(data.get("reward", 0.0))
            done = bool(data.get("done", False))
            if isinstance(obs, list) and isinstance(action, list):
                trainer.add_transition(obs, action, reward, done)
                trainer.maybe_update()
        elif typ == "transition_batch":
            transitions = data.get("transitions", [])
            if isinstance(transitions, list):
                trainer.add_transition_batch(transitions)
                trainer.maybe_update()
        else:
            await ws.send(json.dumps({"type": "echo", "recv": data}))


async def main(host: str = "127.0.0.1", port: int = 8765):
    async with websockets.serve(handle, host, port):
        print(f"WebSocket bridge listening on ws://{host}:{port}")
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
