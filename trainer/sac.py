"""
Soft Actor-Critic (SAC) implementation for tag training.

Off-policy algorithm with automatic entropy tuning.
Uses a replay buffer for sample-efficient learning.
"""
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
except Exception:
    torch = None
    nn = None
    optim = None
    F = None


@dataclass
class SACConfig:
    obs_dim: int
    act_dim: int
    hidden_dim: int = 256
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    init_alpha: float = 0.2
    buffer_size: int = 500_000
    batch_size: int = 256
    warmup_steps: int = 10_000


class SACActorNet(nn.Module):
    """Squashed Gaussian actor: obs -> (mean, log_std)."""

    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, act_dim)
        self.log_std_head = nn.Linear(hidden_dim, act_dim)

    def forward(self, obs: torch.Tensor):
        h = self.trunk(obs)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs: torch.Tensor):
        """Sample action with reparameterization trick + tanh squashing.

        Returns (action, log_prob).
        """
        mean, log_std = self.forward(obs)
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)
        # Reparameterized sample
        x_t = normal.rsample()
        action = torch.tanh(x_t)

        # Log prob with tanh correction
        log_prob = normal.log_prob(x_t) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def deterministic(self, obs: torch.Tensor):
        """Deterministic action (mean with tanh)."""
        mean, _ = self.forward(obs)
        return torch.tanh(mean)


class SACCriticNet(nn.Module):
    """Twin Q-networks: (obs, action) -> (Q1, Q2)."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)


class ReplayBuffer:
    """Fixed-size circular replay buffer with pre-allocated numpy arrays."""

    def __init__(self, obs_dim: int, act_dim: int, max_size: int = 500_000):
        self.max_size = max_size
        self.obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, act_dim), dtype=np.float32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.next_obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.dones = np.zeros(max_size, dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float,
            next_obs: np.ndarray, done: float):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def add_batch(self, obs: np.ndarray, actions: np.ndarray, rewards: np.ndarray,
                  next_obs: np.ndarray, dones: np.ndarray):
        """Add a batch of transitions."""
        n = obs.shape[0]
        if self.ptr + n <= self.max_size:
            self.obs[self.ptr:self.ptr + n] = obs
            self.actions[self.ptr:self.ptr + n] = actions
            self.rewards[self.ptr:self.ptr + n] = rewards
            self.next_obs[self.ptr:self.ptr + n] = next_obs
            self.dones[self.ptr:self.ptr + n] = dones
        else:
            # Wrap around
            first = self.max_size - self.ptr
            self.obs[self.ptr:] = obs[:first]
            self.actions[self.ptr:] = actions[:first]
            self.rewards[self.ptr:] = rewards[:first]
            self.next_obs[self.ptr:] = next_obs[:first]
            self.dones[self.ptr:] = dones[:first]
            rest = n - first
            self.obs[:rest] = obs[first:]
            self.actions[:rest] = actions[first:]
            self.rewards[:rest] = rewards[first:]
            self.next_obs[:rest] = next_obs[first:]
            self.dones[:rest] = dones[first:]
        self.ptr = (self.ptr + n) % self.max_size
        self.size = min(self.size + n, self.max_size)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        idxs = np.random.randint(0, self.size, size=batch_size)
        return {
            'obs': torch.as_tensor(self.obs[idxs]),
            'actions': torch.as_tensor(self.actions[idxs]),
            'rewards': torch.as_tensor(self.rewards[idxs]).unsqueeze(-1),
            'next_obs': torch.as_tensor(self.next_obs[idxs]),
            'dones': torch.as_tensor(self.dones[idxs]).unsqueeze(-1),
        }


class SACAgent:
    """Soft Actor-Critic agent with automatic entropy tuning."""

    def __init__(self, cfg: SACConfig):
        assert torch is not None, "PyTorch required for SACAgent"
        self.cfg = cfg

        # Networks
        self.actor = SACActorNet(cfg.obs_dim, cfg.act_dim, cfg.hidden_dim)
        self.critic = SACCriticNet(cfg.obs_dim, cfg.act_dim, cfg.hidden_dim)
        self.critic_target = SACCriticNet(cfg.obs_dim, cfg.act_dim, cfg.hidden_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Freeze target (no gradient)
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # Automatic entropy tuning
        self.target_entropy = -float(cfg.act_dim)
        self.log_alpha = torch.tensor(np.log(cfg.init_alpha), requires_grad=True)

        # Optimizers
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=cfg.alpha_lr)

    @property
    def alpha(self) -> float:
        return self.log_alpha.exp().item()

    def act(self, obs: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Get action from policy. Returns (action, log_prob, q_value).

        Interface compatible with PPOAgent.act() for zoo interop.
        """
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            action, log_prob = self.actor.sample(x)
            q1, q2 = self.critic(x, action)
            q_value = torch.min(q1, q2)
            return (
                action.squeeze(0).cpu().numpy(),
                float(log_prob.item()),
                float(q_value.item()),
            )

    def act_random(self, act_dim: int) -> np.ndarray:
        """Random action for warmup period."""
        return np.random.uniform(-1.0, 1.0, size=act_dim).astype(np.float32)

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single SAC gradient step from a replay buffer batch."""
        obs = batch['obs']
        actions = batch['actions']
        rewards = batch['rewards']
        next_obs = batch['next_obs']
        dones = batch['dones']

        alpha = self.log_alpha.exp().detach()

        # --- Critic update ---
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_obs)
            q1_target, q2_target = self.critic_target(next_obs, next_actions)
            q_target = torch.min(q1_target, q2_target) - alpha * next_log_probs
            td_target = rewards + self.cfg.gamma * (1.0 - dones) * q_target

        q1, q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_opt.step()

        # --- Actor update ---
        new_actions, log_probs = self.actor.sample(obs)
        q1_new, q2_new = self.critic(obs, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (alpha * log_probs - q_new).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_opt.step()

        # --- Alpha (entropy temperature) update ---
        alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # --- Soft target update ---
        with torch.no_grad():
            for p, p_target in zip(self.critic.parameters(),
                                   self.critic_target.parameters()):
                p_target.data.mul_(1.0 - self.cfg.tau)
                p_target.data.add_(self.cfg.tau * p.data)

        return {
            'critic_loss': float(critic_loss.item()),
            'actor_loss': float(actor_loss.item()),
            'alpha_loss': float(alpha_loss.item()),
            'alpha': float(self.log_alpha.exp().item()),
            'entropy': float(-log_probs.mean().item()),
        }

    def save_policy(self, path: str):
        torch.save({
            'type': 'sac',
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'log_alpha': self.log_alpha.detach().cpu(),
            'actor_opt': self.actor_opt.state_dict(),
            'critic_opt': self.critic_opt.state_dict(),
            'alpha_opt': self.alpha_opt.state_dict(),
            'config': {
                'obs_dim': self.cfg.obs_dim,
                'act_dim': self.cfg.act_dim,
                'hidden_dim': self.cfg.hidden_dim,
            },
        }, path)

    def load_policy(self, path: str):
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and state.get('type') == 'sac':
            self.actor.load_state_dict(state['actor'])
            self.critic.load_state_dict(state['critic'])
            self.critic_target.load_state_dict(state['critic_target'])
            if 'log_alpha' in state:
                self.log_alpha.data.copy_(state['log_alpha'])
            if 'actor_opt' in state:
                self.actor_opt.load_state_dict(state['actor_opt'])
            if 'critic_opt' in state:
                self.critic_opt.load_state_dict(state['critic_opt'])
            if 'alpha_opt' in state:
                self.alpha_opt.load_state_dict(state['alpha_opt'])
        elif isinstance(state, dict) and 'pi' in state:
            # PPO checkpoint — load actor weights only (pi network has same structure
            # if hidden dims match; otherwise this will fail gracefully)
            raise ValueError(
                f"Cannot load PPO checkpoint into SACAgent. Use PPOAgent instead."
            )
        else:
            raise ValueError(f"Unknown checkpoint format: {list(state.keys()) if isinstance(state, dict) else type(state)}")
