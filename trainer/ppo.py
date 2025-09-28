"""
PPO scaffolding for Phase 2 training.
- Define policy network, value network, memory buffer, and update step.
- Real environment interactions should come from the Godot bridge.

This is intentionally lightweight; integrate with PyTorch as needed.
"""
from dataclasses import dataclass
from typing import Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:  # torch optional in early scaffolding
    torch = None
    nn = None
    optim = None


@dataclass
class PPOConfig:
    obs_dim: int
    act_dim: int
    gamma: float = 0.99
    lam: float = 0.95
    clip_ratio: float = 0.2
    lr: float = 3e-4
    train_iters: int = 80
    target_kl: float = 0.01


class MLP(nn.Module):
    def __init__(self, inp: int, out: int):
        super().__init__()
        self.net = nn.Sequential(
            # For ray-based vision, consider a Conv1d front-end before MLP.
            # Example (commented):
            # nn.Conv1d(in_channels=2, out_channels=16, kernel_size=3, stride=1),
            # nn.ReLU(),
            nn.Linear(inp, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, out),
        )

    def forward(self, x):
        return self.net(x)


class PPOAgent:
    def __init__(self, cfg: PPOConfig):
        assert torch is not None, "PyTorch required for PPOAgent"
        self.cfg = cfg
        self.pi = MLP(cfg.obs_dim, cfg.act_dim)
        self.vf = MLP(cfg.obs_dim, 1)
        self.pi_opt = optim.Adam(self.pi.parameters(), lr=cfg.lr)
        self.vf_opt = optim.Adam(self.vf.parameters(), lr=cfg.lr)

    def act(self, obs: np.ndarray) -> Tuple[np.ndarray, float]:
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits = self.pi(x)
            # For continuous actions, replace with Normal; here we use tanh squashing as placeholder
            action = torch.tanh(logits)
            return action.squeeze(0).numpy(), 0.0

    def load_policy(self, path: str):
        state = torch.load(path, map_location="cpu")
        self.pi.load_state_dict(state)

    # Add buffer and update methods as you connect to real rollouts.
