"""
PPO scaffolding for Phase 2 training.
- Define policy network, value network, memory buffer, and update step.
- Real environment interactions should come from the Godot bridge.

This is intentionally lightweight; integrate with PyTorch as needed.
"""
from dataclasses import dataclass
from typing import List, Tuple

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
        self.pi = MLP(cfg.obs_dim, cfg.act_dim * 2)
        self.vf = MLP(cfg.obs_dim, 1)
        self.pi_opt = optim.Adam(self.pi.parameters(), lr=cfg.lr)
        self.vf_opt = optim.Adam(self.vf.parameters(), lr=cfg.lr)

    def act(self, obs: np.ndarray) -> Tuple[np.ndarray, float, float]:
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits = self.pi(x)
            mean, log_std = torch.chunk(logits, 2, dim=-1)
            log_std = torch.clamp(log_std, -4.0, 1.5)
            std = torch.exp(log_std)
            normal = torch.distributions.Normal(mean, std)
            action = torch.tanh(normal.sample())
            pre_tanh = torch.atanh(torch.clamp(action, -0.999, 0.999))
            log_prob = normal.log_prob(pre_tanh) - torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1)
            value = self.vf(x).squeeze(-1)
            return (
                action.squeeze(0).cpu().numpy(),
                float(log_prob.item()),
                float(value.item()),
            )

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.pi(obs)
        mean, log_std = torch.chunk(logits, 2, dim=-1)
        log_std = torch.clamp(log_std, -4.0, 1.5)
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)
        pre_tanh = torch.atanh(torch.clamp(actions, -0.999, 0.999))
        log_prob = normal.log_prob(pre_tanh) - torch.log(1 - actions.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        entropy = normal.entropy().sum(dim=-1)
        value = self.vf(obs).squeeze(-1)
        return log_prob, entropy, value

    def load_policy(self, path: str):
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "pi" in state and "vf" in state:
            self.pi.load_state_dict(state["pi"])
            self.vf.load_state_dict(state["vf"])
        else:
            self.pi.load_state_dict(state)

    def save_policy(self, path: str):
        torch.save({"pi": self.pi.state_dict(), "vf": self.vf.state_dict()}, path)

    def value(self, obs: np.ndarray) -> float:
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            return float(self.vf(x).item())

    def update(self, obs: np.ndarray, actions: np.ndarray, logp_old: np.ndarray, returns: np.ndarray, advantages: np.ndarray) -> dict:
        device = torch.device("cpu")
        o = torch.as_tensor(obs, dtype=torch.float32, device=device)
        a = torch.as_tensor(actions, dtype=torch.float32, device=device)
        logp_old_t = torch.as_tensor(logp_old, dtype=torch.float32, device=device)
        ret = torch.as_tensor(returns, dtype=torch.float32, device=device)
        adv = torch.as_tensor(advantages, dtype=torch.float32, device=device)
        info = {}
        for _ in range(self.cfg.train_iters):
            logp, entropy, value = self.evaluate_actions(o, a)
            ratio = torch.exp(logp - logp_old_t)
            clip_adv = torch.clamp(ratio, 1.0 - self.cfg.clip_ratio, 1.0 + self.cfg.clip_ratio) * adv
            loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()
            loss_v = 0.5 * ((ret - value) ** 2).mean()
            approx_kl = (logp_old_t - logp).mean().item()
            self.pi_opt.zero_grad(); loss_pi.backward(); self.pi_opt.step()
            self.vf_opt.zero_grad(); loss_v.backward(); self.vf_opt.step()
            info = {
                "policy_loss": float(loss_pi.item()),
                "value_loss": float(loss_v.item()),
                "entropy": float(entropy.mean().item()),
                "approx_kl": approx_kl,
            }
            if approx_kl > 1.5 * self.cfg.target_kl:
                break
        return info
