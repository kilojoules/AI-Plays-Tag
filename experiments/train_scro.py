#!/usr/bin/env python3
"""
Train tag agents using Sandwich CRO (Coral Reef Optimization).

This implements the spatially-structured co-evolutionary approach
for comparison with vanilla self-play.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from scro_core import (
    Agent, SandwichReef, GauntletCROOperators, PPOConfig,
    train_agent_ppo, evaluate_agent
)
from trainer.tag_env import SingleTagEnv, TagEnvConfig


@dataclass
class SCROConfig:
    """Configuration for SCRO training on tag game."""
    # Grid structure
    grid_rows: int = 3
    grid_cols: int = 3

    # CRO parameters
    gauntlet_fraction: float = 0.5
    budding_fraction: float = 0.2
    depredation_fraction: float = 0.2
    depredation_probability: float = 0.5
    settlement_attempts: int = 3
    tournament_size: int = 3
    mutation_std: float = 0.02

    # Training
    num_generations: int = 30
    training_steps_per_gen: int = 4096
    eval_episodes: int = 2  # Reduced for faster round-robin evaluation
    learning_rate: float = 3e-4
    hidden_sizes: List[int] = field(default_factory=lambda: [128, 128])

    # Output
    output_dir: str = "experiments/results/scro"
    seed: int = 42


class SCROTrainer:
    """Sandwich CRO trainer for tag game."""

    def __init__(self, config: SCROConfig, env_config: Optional[TagEnvConfig] = None):
        self.config = config
        self.env_config = env_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set seeds
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        self.rng = np.random.default_rng(config.seed)

        # Create sample environment to get dimensions
        sample_env = SingleTagEnv(config=env_config)
        self.obs_dim = sample_env.obs_dim
        self.act_dim = sample_env.act_dim

        # Create the reef
        self.reef = SandwichReef(
            rows=config.grid_rows,
            cols=config.grid_cols,
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            hidden_sizes=config.hidden_sizes,
            device=self.device,
            learning_rate=config.learning_rate,
            rng=self.rng,
        )

        # Create operators
        self.operators = GauntletCROOperators(
            reef=self.reef,
            mutation_std=config.mutation_std,
            tournament_size=config.tournament_size,
            rng=self.rng,
        )

        # PPO config
        self.ppo_config = PPOConfig(
            n_steps=min(config.training_steps_per_gen, 2048),
            num_minibatches=8,
            update_epochs=4,
        )

        # Setup output
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(config.output_dir, self.run_id)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)

        # Metrics
        self.metrics_path = os.path.join(self.output_dir, "metrics.csv")
        self._init_metrics()

        # Stats
        self.generation = 0
        self.total_timesteps = 0
        self.protagonist_wins = 0
        self.antagonist_wins = 0

    def _make_env(self):
        """Factory function to create fresh environments."""
        return SingleTagEnv(config=self.env_config)

    def _init_metrics(self):
        """Initialize metrics CSV."""
        columns = [
            "generation", "timesteps", "protagonist_pop", "antagonist_pop",
            "best_prot_adv_fitness", "best_prot_clean_fitness",
            "best_antag_adv_fitness", "best_antag_clean_fitness",
            "mean_prot_fitness", "mean_antag_fitness",
            "protagonist_win_rate", "time_elapsed"
        ]
        with open(self.metrics_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(columns)

    def _log_metrics(self, gen: int, timesteps: int, time_elapsed: float):
        """Log generation metrics."""
        prot_cells = self.reef.get_occupied_cells("protagonist")
        antag_cells = self.reef.get_occupied_cells("adversary")

        best_prot_adv = max((c.fitness_adversarial for c in prot_cells), default=0)
        best_prot_clean = max((c.fitness_clean for c in prot_cells), default=0)
        best_antag_adv = max((c.fitness_adversarial for c in antag_cells), default=0)
        best_antag_clean = max((c.fitness_clean for c in antag_cells), default=0)

        mean_prot = np.mean([c.fitness_adversarial for c in prot_cells]) if prot_cells else 0
        mean_antag = np.mean([c.fitness_adversarial for c in antag_cells]) if antag_cells else 0

        total = self.protagonist_wins + self.antagonist_wins
        win_rate = self.protagonist_wins / max(total, 1)

        row = [
            gen, timesteps, len(prot_cells), len(antag_cells),
            best_prot_adv, best_prot_clean, best_antag_adv, best_antag_clean,
            mean_prot, mean_antag, win_rate, time_elapsed
        ]

        with open(self.metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def initialize_population(self, pretrained_seeker: Optional[str] = None,
                               pretrained_hider: Optional[str] = None):
        """Initialize the reef with agents.

        If pre-trained policies are provided, all agents start from those weights
        with small random perturbations for diversity.
        """
        print(f"Initializing {self.config.grid_rows}x{self.config.grid_cols} reef...")

        # Load pre-trained weights if provided
        seeker_weights = None
        hider_weights = None

        if pretrained_seeker:
            print(f"  Loading pre-trained seeker from: {pretrained_seeker}")
            checkpoint = torch.load(pretrained_seeker, map_location='cpu', weights_only=True)
            # Convert vanilla PPO format to SCRO format
            seeker_weights = self._convert_vanilla_to_scro(checkpoint)

        if pretrained_hider:
            print(f"  Loading pre-trained hider from: {pretrained_hider}")
            checkpoint = torch.load(pretrained_hider, map_location='cpu', weights_only=True)
            hider_weights = self._convert_vanilla_to_scro(checkpoint)

        for i in range(self.reef.rows):
            for j in range(self.reef.cols):
                prot_cell = self.reef.get_cell(i, j, "protagonist")
                antag_cell = self.reef.get_cell(i, j, "adversary")

                prot_cell.initialize_agent(self.config.learning_rate)
                antag_cell.initialize_agent(self.config.learning_rate)

                # Load pre-trained weights with small perturbation for diversity
                if seeker_weights is not None:
                    self._load_with_perturbation(prot_cell.agent, seeker_weights,
                                                  perturbation_std=0.01)
                if hider_weights is not None:
                    self._load_with_perturbation(antag_cell.agent, hider_weights,
                                                  perturbation_std=0.01)

        print(f"  Protagonists (seekers): {self.reef.count_occupied('protagonist')}")
        print(f"  Antagonists (hiders): {self.reef.count_occupied('adversary')}")
        if seeker_weights or hider_weights:
            print(f"  Initialized from pre-trained policies with perturbation")

    def _convert_vanilla_to_scro(self, vanilla_checkpoint: dict) -> dict:
        """Convert vanilla PPO checkpoint format to SCRO agent format.

        Vanilla format: {"pi": {fc1.weight, fc1.bias, fc2.weight, ...}, "vf": {...}}
        SCRO format: {policy.0.weight, policy.0.bias, policy.2.weight, ...}
        """
        pi_state = vanilla_checkpoint["pi"]
        scro_state = {}

        # Map vanilla layer names to SCRO layer names
        # Vanilla: fc1, fc2, fc3 (mean layer)
        # SCRO: policy.0, policy.2, policy.4 (Sequential with ReLU between)
        layer_map = {
            "fc1.weight": "policy.0.weight",
            "fc1.bias": "policy.0.bias",
            "fc2.weight": "policy.2.weight",
            "fc2.bias": "policy.2.bias",
            "fc3.weight": "policy.4.weight",
            "fc3.bias": "policy.4.bias",
        }

        for vanilla_name, scro_name in layer_map.items():
            if vanilla_name in pi_state:
                scro_state[scro_name] = pi_state[vanilla_name]

        return scro_state

    def _load_with_perturbation(self, agent, weights: dict, perturbation_std: float = 0.01):
        """Load weights into agent with small random perturbation for diversity."""
        current_state = agent.state_dict()

        for name, param in weights.items():
            if name in current_state and current_state[name].shape == param.shape:
                # Add small perturbation
                noise = torch.randn_like(param) * perturbation_std
                current_state[name] = param + noise

        agent.load_state_dict(current_state, strict=False)

    def evaluate_population(self):
        """Evaluate all agents using round-robin against ALL opponents.

        Each seeker plays against every hider, and vice versa.
        Fitness is the average performance across all opponents.
        This prevents overfitting to a single local opponent.
        """
        prot_cells = self.reef.get_occupied_cells("protagonist")
        antag_cells = self.reef.get_occupied_cells("adversary")

        if not prot_cells or not antag_cells:
            return

        # Reset fitness accumulators
        prot_fitness_sums = {id(c): 0.0 for c in prot_cells}
        prot_fitness_counts = {id(c): 0 for c in prot_cells}
        antag_fitness_sums = {id(c): 0.0 for c in antag_cells}
        antag_fitness_counts = {id(c): 0 for c in antag_cells}

        # Round-robin: each protagonist vs each antagonist
        for prot_cell in prot_cells:
            for antag_cell in antag_cells:
                # Evaluate this matchup
                prot_reward = evaluate_agent(
                    agent=prot_cell.agent,
                    opponent=antag_cell.agent,
                    env_fn=self._make_env,
                    device=self.device,
                    is_protagonist=True,
                    num_episodes=max(1, self.config.eval_episodes // len(antag_cells)),
                )

                # Accumulate fitness for both agents
                prot_fitness_sums[id(prot_cell)] += prot_reward
                prot_fitness_counts[id(prot_cell)] += 1

                # Antagonist gets inverse reward
                antag_fitness_sums[id(antag_cell)] += (-prot_reward)
                antag_fitness_counts[id(antag_cell)] += 1

                # Track wins
                if prot_reward > 0:
                    self.protagonist_wins += 1
                else:
                    self.antagonist_wins += 1

        # Compute average fitness across all opponents
        for prot_cell in prot_cells:
            count = prot_fitness_counts[id(prot_cell)]
            if count > 0:
                prot_cell.fitness_adversarial = prot_fitness_sums[id(prot_cell)] / count

            # Clean fitness (vs no opponent) - used for budding selection
            prot_cell.fitness_clean = evaluate_agent(
                agent=prot_cell.agent,
                opponent=None,
                env_fn=self._make_env,
                device=self.device,
                is_protagonist=True,
                num_episodes=self.config.eval_episodes,
            )

        for antag_cell in antag_cells:
            count = antag_fitness_counts[id(antag_cell)]
            if count > 0:
                antag_cell.fitness_adversarial = antag_fitness_sums[id(antag_cell)] / count

            # Clean fitness
            antag_cell.fitness_clean = evaluate_agent(
                agent=antag_cell.agent,
                opponent=None,
                env_fn=self._make_env,
                device=self.device,
                is_protagonist=False,
                num_episodes=self.config.eval_episodes,
            )

    def train_population(self):
        """Train all occupied cells via PPO against random opponents.

        Each agent trains against a randomly sampled opponent from the
        opposing population, promoting diverse strategy learning.
        """
        prot_cells = self.reef.get_occupied_cells("protagonist")
        antag_cells = self.reef.get_occupied_cells("adversary")

        if not prot_cells or not antag_cells:
            return

        # Train each protagonist against a random antagonist
        for prot_cell in prot_cells:
            # Sample random opponent from antagonist population
            antag_cell = self.rng.choice(antag_cells)

            train_agent_ppo(
                agent=prot_cell.agent,
                optimizer=prot_cell.optimizer,
                env_fn=self._make_env,
                opponent=antag_cell.agent,
                device=self.device,
                config=self.ppo_config,
                is_protagonist=True,
                num_steps=self.config.training_steps_per_gen,
            )
            self.total_timesteps += self.config.training_steps_per_gen

        # Train each antagonist against a random protagonist
        for antag_cell in antag_cells:
            # Sample random opponent from protagonist population
            prot_cell = self.rng.choice(prot_cells)

            train_agent_ppo(
                agent=antag_cell.agent,
                optimizer=antag_cell.optimizer,
                env_fn=self._make_env,
                opponent=prot_cell.agent,
                device=self.device,
                config=self.ppo_config,
                is_protagonist=False,
                num_steps=self.config.training_steps_per_gen,
            )
            self.total_timesteps += self.config.training_steps_per_gen

    def apply_cro_dynamics(self, layer: str):
        """Apply CRO evolutionary operators to a layer."""
        # 1. Gauntlet Spawning
        gauntlet_larvae = self.operators.gauntlet_spawning(
            layer=layer,
            fraction=self.config.gauntlet_fraction,
        )

        for larva, target, fitness in gauntlet_larvae:
            self.operators.attempt_settlement(
                larva=larva,
                target_position=target,
                larva_fitness=fitness,
                layer=layer,
                max_attempts=self.config.settlement_attempts,
                use_clean_fitness=False,
            )

        # 2. Budding (top clean performers)
        bud_larvae = self.operators.budding(
            layer=layer,
            fraction=self.config.budding_fraction,
        )

        for larva, target, fitness in bud_larvae:
            self.operators.attempt_settlement(
                larva=larva,
                target_position=target,
                larva_fitness=fitness,
                layer=layer,
                max_attempts=self.config.settlement_attempts,
                use_clean_fitness=True,
            )

        # 3. Depredation
        self.operators.depredation(
            layer=layer,
            fraction=self.config.depredation_fraction,
            probability=self.config.depredation_probability,
        )

    def save_best_agents(self, gen: int):
        """Save the best agents from current generation."""
        prot_cells = self.reef.get_occupied_cells("protagonist")
        antag_cells = self.reef.get_occupied_cells("adversary")

        if prot_cells:
            best_prot = max(prot_cells, key=lambda c: c.fitness_adversarial)
            path = os.path.join(self.output_dir, "checkpoints", f"protagonist_gen{gen:03d}.pt")
            torch.save(best_prot.agent.state_dict(), path)

        if antag_cells:
            best_antag = max(antag_cells, key=lambda c: c.fitness_adversarial)
            path = os.path.join(self.output_dir, "checkpoints", f"antagonist_gen{gen:03d}.pt")
            torch.save(best_antag.agent.state_dict(), path)

    def save_final(self):
        """Save final best policies."""
        prot_cells = self.reef.get_occupied_cells("protagonist")
        antag_cells = self.reef.get_occupied_cells("adversary")

        if prot_cells:
            best_prot = max(prot_cells, key=lambda c: c.fitness_adversarial)
            torch.save(best_prot.agent.state_dict(),
                      os.path.join(self.output_dir, "best_protagonist.pt"))

        if antag_cells:
            best_antag = max(antag_cells, key=lambda c: c.fitness_adversarial)
            torch.save(best_antag.agent.state_dict(),
                      os.path.join(self.output_dir, "best_antagonist.pt"))

        # Save metadata
        total = self.protagonist_wins + self.antagonist_wins
        metadata = {
            'run_id': self.run_id,
            'algorithm': 'SCRO',
            'num_generations': self.generation,
            'total_timesteps': self.total_timesteps,
            'protagonist_wins': self.protagonist_wins,
            'antagonist_wins': self.antagonist_wins,
            'protagonist_win_rate': self.protagonist_wins / max(total, 1),
            'config': {
                'grid_rows': self.config.grid_rows,
                'grid_cols': self.config.grid_cols,
                'gauntlet_fraction': self.config.gauntlet_fraction,
                'mutation_std': self.config.mutation_std,
                'training_steps_per_gen': self.config.training_steps_per_gen,
                'seed': self.config.seed,
            }
        }

        with open(os.path.join(self.output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\nFinal policies saved to {self.output_dir}")

    def train(self, pretrained_seeker: Optional[str] = None,
              pretrained_hider: Optional[str] = None):
        """Main SCRO training loop."""
        print(f"Starting SCRO training for {self.config.num_generations} generations")
        print(f"  Grid: {self.config.grid_rows}x{self.config.grid_cols}")
        print(f"  Device: {self.device}")
        print(f"  Output: {self.output_dir}\n")

        start_time = time.time()
        self.initialize_population(pretrained_seeker, pretrained_hider)

        for gen in range(self.config.num_generations):
            self.generation = gen + 1
            gen_start = time.time()

            # 1. Train all pairs
            self.train_population()

            # 2. Evaluate population
            self.evaluate_population()

            # 3. Apply CRO dynamics to both layers
            self.apply_cro_dynamics("protagonist")
            self.apply_cro_dynamics("adversary")

            # 4. Log and checkpoint
            elapsed = time.time() - start_time
            self._log_metrics(gen + 1, self.total_timesteps, elapsed)

            total = self.protagonist_wins + self.antagonist_wins
            win_rate = self.protagonist_wins / max(total, 1)

            gen_time = time.time() - gen_start
            print(f"Gen {gen+1:3d}/{self.config.num_generations} | "
                  f"Pop: {self.reef.count_occupied('protagonist')}/{self.reef.count_occupied('adversary')} | "
                  f"Seeker WR: {win_rate:.1%} | "
                  f"Time: {gen_time:.1f}s")

            # Save checkpoint every 5 generations
            if (gen + 1) % 5 == 0:
                self.save_best_agents(gen + 1)

        self.save_final()

        total_time = time.time() - start_time
        total = self.protagonist_wins + self.antagonist_wins
        print(f"\nSCRO training complete!")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Total timesteps: {self.total_timesteps:,}")
        print(f"  Final seeker win rate: {self.protagonist_wins / max(total, 1):.1%}")


def main():
    parser = argparse.ArgumentParser(description="SCRO training for tag game")
    parser.add_argument("--generations", type=int, default=50,
                        help="Number of generations")
    parser.add_argument("--grid-size", type=int, default=3,
                        help="Grid size (NxN)")
    parser.add_argument("--training-steps", type=int, default=4096,
                        help="Training steps per generation per agent")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output-dir", type=str, default="experiments/results/scro",
                        help="Output directory")
    parser.add_argument("--layout", type=str, default="empty",
                        choices=["empty", "four_corners", "central_cross"],
                        help="Arena layout with obstacles (default: empty)")
    parser.add_argument("--pretrained-seeker", type=str, default=None,
                        help="Path to pre-trained seeker policy (vanilla format)")
    parser.add_argument("--pretrained-hider", type=str, default=None,
                        help="Path to pre-trained hider policy (vanilla format)")

    args = parser.parse_args()

    config = SCROConfig(
        grid_rows=args.grid_size,
        grid_cols=args.grid_size,
        num_generations=args.generations,
        training_steps_per_gen=args.training_steps,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    # Create environment config with layout
    env_config = TagEnvConfig(layout=args.layout)

    trainer = SCROTrainer(config, env_config=env_config)
    trainer.train(pretrained_seeker=args.pretrained_seeker,
                  pretrained_hider=args.pretrained_hider)


if __name__ == "__main__":
    main()
