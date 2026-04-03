---
layout: default
title: AI Plays Tag
---

# AI Plays Tag

Multi-agent reinforcement learning agents that learn to play tag through self-play.

## Studies

These studies form a progression: from reward design, to training topology and hyperparameters, to the mechanism behind SAC's dominance.

1. **[Reward Shaping Study](reward_shaping/)** — How reward function design determines agent behavior: 8 reward presets × 2 algorithms (PPO, SAC) × 3 seeds, with cross-config gauntlet evaluation. Finding: algorithm choice matters more than reward design.
2. **[Hyperparameter Tuning & Zoo Mixing Analysis](hpo_study/)** — Optuna-optimized hyperparameters, 150-run A-sweep, and cross-config gauntlet. Finding: zoo mixing doesn't help, and SAC dominates PPO 95-to-2 even with optimized PPO.
3. **[Entropy Temperature Ablation](entropy_study/)** — Why does SAC dominate? Auto-tuned entropy decays to near-zero within 500K steps. Fixed entropy is catastrophically worse than no entropy at all. The mechanism is a brief exploration phase, not sustained entropy.
