#!/usr/bin/env python3
"""
Self-contained SCRO (Sandwich Coral Reef Optimization) implementation.

Adapted from sandwich-reef for the tag game without external dependencies
like pettingzoo or gymnasium.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class Agent(nn.Module):
    """Actor-Critic neural network agent."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: List[int] = [128, 128]):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Build critic network
        critic_layers = []
        prev_size = obs_dim
        for size in hidden_sizes:
            critic_layers.append(nn.Linear(prev_size, size))
            critic_layers.append(nn.Tanh())
            prev_size = size
        critic_layers.append(nn.Linear(prev_size, 1))
        self.critic = nn.Sequential(*critic_layers)

        # Build actor network
        actor_layers = []
        prev_size = obs_dim
        for size in hidden_sizes:
            actor_layers.append(nn.Linear(prev_size, size))
            actor_layers.append(nn.Tanh())
            prev_size = size
        actor_layers.append(nn.Linear(prev_size, act_dim))
        self.actor_mean = nn.Sequential(*actor_layers)

        # Learnable log std
        self.actor_logstd = nn.Parameter(torch.zeros(act_dim))

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs)

    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor = None):
        mean = self.actor_mean(obs)
        std = self.actor_logstd.exp()

        dist = torch.distributions.Normal(mean, std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic(obs)

        return action, log_prob, entropy, value

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """Get action using mean (deterministic)."""
        with torch.no_grad():
            return self.actor_mean(obs)


@dataclass
class CoralCell:
    """A single cell in the coral reef grid."""
    position: Tuple[int, int]
    obs_dim: int
    act_dim: int
    hidden_sizes: List[int]
    device: torch.device
    is_protagonist: bool = True

    agent: Optional[Agent] = None
    optimizer: Optional[optim.Adam] = None
    fitness_adversarial: float = float('-inf')
    fitness_clean: float = float('-inf')
    occupied: bool = False

    def initialize_agent(self, learning_rate: float = 3e-4):
        """Create a new agent with random weights."""
        self.agent = Agent(self.obs_dim, self.act_dim, self.hidden_sizes).to(self.device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=learning_rate, eps=1e-5)
        self.occupied = True
        self.fitness_adversarial = float('-inf')
        self.fitness_clean = float('-inf')

    def set_agent(self, agent: Agent, learning_rate: float = 3e-4):
        """Set the agent for this cell."""
        self.agent = agent.to(self.device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=learning_rate, eps=1e-5)
        self.occupied = True

    def clear(self):
        """Remove the agent from this cell."""
        self.agent = None
        self.optimizer = None
        self.occupied = False
        self.fitness_adversarial = float('-inf')
        self.fitness_clean = float('-inf')

    def copy_agent(self) -> Optional[Agent]:
        """Return a deep copy of the agent."""
        if self.agent is None:
            return None
        new_agent = Agent(self.obs_dim, self.act_dim, self.hidden_sizes).to(self.device)
        new_agent.load_state_dict(copy.deepcopy(self.agent.state_dict()))
        return new_agent


class SandwichReef:
    """Two-layer coral reef structure for co-evolutionary training."""

    def __init__(
        self,
        rows: int,
        cols: int,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: List[int],
        device: torch.device,
        learning_rate: float = 3e-4,
        rng: np.random.Generator = None,
    ):
        self.rows = rows
        self.cols = cols
        self.device = device
        self.learning_rate = learning_rate
        self.rng = rng if rng is not None else np.random.default_rng()

        # Create the two layers
        self.protagonist_layer: List[List[CoralCell]] = []
        self.adversary_layer: List[List[CoralCell]] = []

        for i in range(rows):
            prot_row = []
            adv_row = []
            for j in range(cols):
                prot_cell = CoralCell(
                    position=(i, j),
                    obs_dim=obs_dim,
                    act_dim=act_dim,
                    hidden_sizes=hidden_sizes,
                    device=device,
                    is_protagonist=True,
                )
                adv_cell = CoralCell(
                    position=(i, j),
                    obs_dim=obs_dim,
                    act_dim=act_dim,
                    hidden_sizes=hidden_sizes,
                    device=device,
                    is_protagonist=False,
                )
                prot_row.append(prot_cell)
                adv_row.append(adv_cell)
            self.protagonist_layer.append(prot_row)
            self.adversary_layer.append(adv_row)

    def get_cell(self, i: int, j: int, layer: str = "protagonist") -> CoralCell:
        if layer == "protagonist":
            return self.protagonist_layer[i][j]
        else:
            return self.adversary_layer[i][j]

    def get_occupied_cells(self, layer: str = "protagonist") -> List[CoralCell]:
        if layer == "protagonist":
            cells = [cell for row in self.protagonist_layer for cell in row if cell.occupied]
        else:
            cells = [cell for row in self.adversary_layer for cell in row if cell.occupied]
        return cells

    def count_occupied(self, layer: str = "protagonist") -> int:
        return len(self.get_occupied_cells(layer))


class GauntletCROOperators:
    """CRO Operators with Gauntlet Spawning (Tournament Selection + Mutation)."""

    def __init__(
        self,
        reef: SandwichReef,
        mutation_std: float = 0.02,
        tournament_size: int = 3,
        rng: np.random.Generator = None,
    ):
        self.reef = reef
        self.mutation_std = mutation_std
        self.tournament_size = tournament_size
        self.rng = rng if rng is not None else np.random.default_rng()

    def mutate(self, parent: Agent, cell_template: CoralCell) -> Agent:
        """Create offspring through mutation."""
        child = Agent(
            cell_template.obs_dim,
            cell_template.act_dim,
            cell_template.hidden_sizes
        ).to(cell_template.device)

        parent_state = parent.state_dict()
        child_state = child.state_dict()

        for key in child_state.keys():
            noise = torch.randn_like(parent_state[key]) * self.mutation_std
            child_state[key] = parent_state[key] + noise

        child.load_state_dict(child_state)
        return child

    def gauntlet_spawning(
        self,
        layer: str,
        fraction: float,
    ) -> List[Tuple[Agent, Tuple[int, int], float]]:
        """Tournament selection + mutation."""
        cells = self.reef.get_occupied_cells(layer)
        if len(cells) < 2:
            return []

        num_spawning_events = max(1, int(len(cells) * fraction))

        larvae = []
        for _ in range(num_spawning_events):
            tournament_size = min(len(cells), self.tournament_size)
            candidates = list(self.rng.choice(cells, size=tournament_size, replace=False))
            winner_cell = max(candidates, key=lambda c: c.fitness_adversarial)

            child = self.mutate(winner_cell.agent, winner_cell)
            target = (
                self.rng.integers(self.reef.rows),
                self.rng.integers(self.reef.cols),
            )
            larvae.append((child, target, winner_cell.fitness_adversarial))

        return larvae

    def budding(
        self,
        layer: str,
        fraction: float,
    ) -> List[Tuple[Agent, Tuple[int, int], float]]:
        """Top performers by clean fitness duplicate."""
        cells = self.reef.get_occupied_cells(layer)
        if len(cells) == 0:
            return []

        num_budders = max(1, int(len(cells) * fraction))
        cells_sorted = sorted(cells, key=lambda c: c.fitness_clean, reverse=True)
        budders = cells_sorted[:num_budders]

        larvae = []
        for cell in budders:
            child = cell.copy_agent()
            if child is not None:
                target = (
                    self.rng.integers(self.reef.rows),
                    self.rng.integers(self.reef.cols),
                )
                larvae.append((child, target, cell.fitness_clean))

        return larvae

    def depredation(
        self,
        layer: str,
        fraction: float,
        probability: float,
    ) -> int:
        """Remove bottom performers by clean fitness."""
        if self.rng.random() > probability:
            return 0

        cells = self.reef.get_occupied_cells(layer)
        if len(cells) == 0:
            return 0

        num_to_remove = max(1, int(len(cells) * fraction))
        cells_sorted = sorted(cells, key=lambda c: c.fitness_clean)
        to_remove = cells_sorted[:num_to_remove]

        count = 0
        for cell in to_remove:
            cell.clear()
            count += 1

        return count

    def attempt_settlement(
        self,
        larva: Agent,
        target_position: Tuple[int, int],
        larva_fitness: float,
        layer: str,
        max_attempts: int,
        use_clean_fitness: bool = False,
    ) -> bool:
        """Attempt to settle a larva at the target position."""
        for attempt in range(max_attempts):
            i, j = target_position
            cell = self.reef.get_cell(i, j, layer)

            if not cell.occupied:
                cell.set_agent(larva, self.reef.learning_rate)
                if use_clean_fitness:
                    cell.fitness_clean = larva_fitness
                else:
                    cell.fitness_adversarial = larva_fitness
                return True
            else:
                if use_clean_fitness:
                    resident_fitness = cell.fitness_clean
                else:
                    resident_fitness = cell.fitness_adversarial

                if larva_fitness > resident_fitness:
                    cell.set_agent(larva, self.reef.learning_rate)
                    if use_clean_fitness:
                        cell.fitness_clean = larva_fitness
                    else:
                        cell.fitness_adversarial = larva_fitness
                    return True

            target_position = (
                self.rng.integers(self.reef.rows),
                self.rng.integers(self.reef.cols),
            )

        return False


@dataclass
class PPOConfig:
    """Configuration for PPO training."""
    gamma: float = 0.99
    gae_lambda: float = 0.95
    n_steps: int = 2048
    num_minibatches: int = 32
    update_epochs: int = 10
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5


def train_agent_ppo(
    agent: Agent,
    optimizer: optim.Adam,
    env_fn: Callable,
    opponent: Optional[Agent],
    device: torch.device,
    config: PPOConfig,
    is_protagonist: bool = True,
    num_steps: int = 4096,
) -> float:
    """
    Train an agent using PPO.

    Args:
        agent: The agent to train.
        optimizer: Adam optimizer for the agent.
        env_fn: Function that creates a fresh environment.
        opponent: The opponent agent (can be None).
        device: Torch device for computation.
        config: PPO hyperparameters.
        is_protagonist: Whether training protagonist (seeker) or antagonist (hider).
        num_steps: Total training steps.

    Returns:
        Average episode reward during training.
    """
    env = env_fn()

    n_steps = config.n_steps
    batch_size = n_steps
    minibatch_size = batch_size // config.num_minibatches
    num_updates = max(1, num_steps // n_steps)

    agent_role = 'seeker' if is_protagonist else 'hider'
    opponent_role = 'hider' if is_protagonist else 'seeker'

    obs_dim = env.obs_dim
    act_dim = env.act_dim

    obs_buffer = torch.zeros((n_steps, obs_dim)).to(device)
    actions_buffer = torch.zeros((n_steps, act_dim)).to(device)
    logprobs_buffer = torch.zeros(n_steps).to(device)
    rewards_buffer = torch.zeros(n_steps).to(device)
    dones_buffer = torch.zeros(n_steps).to(device)
    values_buffer = torch.zeros(n_steps).to(device)

    total_rewards = []
    obs = env.reset()
    next_obs = torch.Tensor(obs[agent_role]).to(device)
    next_done = torch.zeros(1).to(device)

    for update in range(num_updates):
        episode_reward = 0.0

        for step in range(n_steps):
            dones_buffer[step] = next_done
            obs_buffer[step] = next_obs

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs.unsqueeze(0))
                values_buffer[step] = value.flatten()
                actions_buffer[step] = action.squeeze(0)
                logprobs_buffer[step] = logprob.flatten()

                if opponent is not None:
                    opp_obs = torch.Tensor(obs[opponent_role]).to(device).unsqueeze(0)
                    opp_action = opponent.act(opp_obs).squeeze(0).cpu().numpy()
                else:
                    opp_action = np.zeros(act_dim)

            step_actions = {
                agent_role: action.squeeze(0).cpu().numpy(),
                opponent_role: opp_action,
            }

            obs, rewards, done, info = env.step(step_actions)
            reward = rewards[agent_role]
            rewards_buffer[step] = torch.tensor(reward).to(device)
            episode_reward += reward

            if done:
                total_rewards.append(episode_reward)
                episode_reward = 0.0
                obs = env.reset()

            next_obs = torch.Tensor(obs[agent_role]).to(device)
            next_done = torch.tensor([done], dtype=torch.float32).to(device)

        # PPO update using GAE
        with torch.no_grad():
            next_value = agent.get_value(next_obs.unsqueeze(0)).reshape(1, -1)
            advantages = torch.zeros_like(rewards_buffer).to(device)
            lastgaelam = 0

            for t in reversed(range(n_steps)):
                if t == n_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buffer[t + 1]
                    nextvalues = values_buffer[t + 1]

                delta = rewards_buffer[t] + config.gamma * nextvalues * nextnonterminal - values_buffer[t]
                advantages[t] = lastgaelam = delta + config.gamma * config.gae_lambda * nextnonterminal * lastgaelam

            returns = advantages + values_buffer

        b_obs = obs_buffer
        b_logprobs = logprobs_buffer
        b_actions = actions_buffer
        b_advantages = advantages
        b_returns = returns

        b_inds = np.arange(batch_size)

        for epoch in range(config.update_epochs):
            np.random.shuffle(b_inds)

            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss = torch.max(
                    -mb_advantages * ratio,
                    -mb_advantages * torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef),
                ).mean()

                v_loss = 0.5 * ((newvalue.view(-1) - b_returns[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - config.ent_coef * entropy_loss + v_loss * config.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
                optimizer.step()

    return np.mean(total_rewards) if total_rewards else 0.0


def evaluate_agent(
    agent: Agent,
    opponent: Optional[Agent],
    env_fn: Callable,
    device: torch.device,
    is_protagonist: bool = True,
    num_episodes: int = 5,
) -> float:
    """Evaluate an agent over multiple episodes."""
    env = env_fn()

    agent_role = 'seeker' if is_protagonist else 'hider'
    opponent_role = 'hider' if is_protagonist else 'seeker'

    total_reward = 0.0

    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            with torch.no_grad():
                agent_obs = torch.Tensor(obs[agent_role]).to(device).unsqueeze(0)
                agent_action = agent.act(agent_obs).squeeze(0).cpu().numpy()

                if opponent is not None:
                    opp_obs = torch.Tensor(obs[opponent_role]).to(device).unsqueeze(0)
                    opp_action = opponent.act(opp_obs).squeeze(0).cpu().numpy()
                else:
                    opp_action = np.zeros(env.act_dim)

            actions = {
                agent_role: agent_action,
                opponent_role: opp_action,
            }

            obs, rewards, done, info = env.step(actions)
            episode_reward += rewards[agent_role]

        total_reward += episode_reward

    return total_reward / num_episodes
