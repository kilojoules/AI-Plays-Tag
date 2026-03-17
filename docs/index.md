---
layout: default
title: AI Plays Tag
---

# AI Plays Tag

Multi-agent reinforcement learning agents that learn to play tag through self-play.

## Studies

- **[Hyperparameter Tuning & Zoo Mixing Analysis](hpo_study/)** — Optuna-optimized hyperparameters, 150-run A-sweep, and cross-config gauntlet reveal that zoo mixing doesn't help — but SAC dominates PPO 95-to-2 in cross-algorithm play.
- **[Reward Shaping Study](reward_shaping/)** — How reward function design determines agent behavior: 8 reward presets × 2 algorithms (PPO, SAC) × 3 seeds, with cross-config gauntlet evaluation.
