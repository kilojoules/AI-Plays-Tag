#!/usr/bin/env python3
"""
Stub PPO training on the same simple dynamics used by server.py.
Trains a policy to increase velocity magnitude (toy objective) and saves it to policy.pt.
This is for demonstration; replace with a real reward (tag logic) when wiring Godot fully.
"""
import os
import math
from typing import Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception as e:
    print("PyTorch not available:", e)
    raise SystemExit(1)

from ppo import PPOAgent, PPOConfig


def step_dynamics(pos: np.ndarray, vel: np.ndarray, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    acc = action[:2]
    vel = 0.85 * vel + 0.15 * acc * 5.0
    pos = pos + vel * 0.1
    return pos, vel


def collect_rollout(agent: PPOAgent, steps: int = 256):
    obs_buf, act_buf, rew_buf, val_buf, logp_buf = [], [], [], [], []
    pos = np.zeros(2, dtype=np.float32)
    vel = np.zeros(2, dtype=np.float32)
    is_it = 0.0
    for t in range(steps):
        obs = np.array([pos[0], pos[1], vel[0], vel[1], is_it], dtype=np.float32)
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits = agent.pi(x)
            action = torch.tanh(logits)
            value = agent.vf(x)
        act = action.squeeze(0).numpy()
        pos, vel = step_dynamics(pos, vel, act)
        reward = float(np.linalg.norm(vel)) * 0.01
        obs_buf.append(obs)
        act_buf.append(act)
        rew_buf.append(reward)
        val_buf.append(value.item())
        logp_buf.append(0.0)  # placeholder
    return (
        np.array(obs_buf, dtype=np.float32),
        np.array(act_buf, dtype=np.float32),
        np.array(rew_buf, dtype=np.float32),
        np.array(val_buf, dtype=np.float32),
        np.array(logp_buf, dtype=np.float32),
    )


def ppo_update(agent: PPOAgent, obs, act, ret, adv, epochs=5, batch=64):
    n = len(obs)
    for _ in range(epochs):
        idx = np.random.permutation(n)
        for start in range(0, n, batch):
            j = idx[start:start+batch]
            o = torch.as_tensor(obs[j], dtype=torch.float32)
            a = torch.as_tensor(act[j], dtype=torch.float32)
            r = torch.as_tensor(ret[j], dtype=torch.float32)
            adv_j = torch.as_tensor(adv[j], dtype=torch.float32)

            logits = agent.pi(o)
            pred = torch.tanh(logits)
            pi_loss = ((pred - a) ** 2 * adv_j.unsqueeze(-1)).mean()

            v = agent.vf(o).squeeze(-1)
            v_loss = ((v - r) ** 2).mean()

            agent.pi_opt.zero_grad(); pi_loss.backward(); agent.pi_opt.step()
            agent.vf_opt.zero_grad(); v_loss.backward(); agent.vf_opt.step()


def main():
    # Vision-based placeholder: default to 36 rays -> 72 dims
    obs_dim = int(os.environ.get("OBS_DIM", "72"))
    cfg = PPOConfig(obs_dim=obs_dim, act_dim=3)
    agent = PPOAgent(cfg)
    gamma = 0.99
    lam = 0.95
    iters = 50
    for it in range(iters):
        obs, act, rew, val, _ = collect_rollout(agent, steps=512)
        # GAE-lambda advantages
        adv = np.zeros_like(rew)
        lastgaelam = 0.0
        for t in reversed(range(len(rew))):
            nextv = 0.0 if t == len(rew)-1 else val[t+1]
            delta = rew[t] + gamma * nextv - val[t]
            lastgaelam = delta + gamma * lam * lastgaelam
            adv[t] = lastgaelam
        ret = adv + val
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ppo_update(agent, obs, act, ret, adv, epochs=5, batch=128)
    os.makedirs("trainer", exist_ok=True)
    torch.save(agent.pi.state_dict(), os.path.join(os.path.dirname(__file__), "policy.pt"))
    print("Saved policy to trainer/policy.pt")


if __name__ == "__main__":
    main()
