---
layout: default
title: Entropy Temperature Ablation
---

# Entropy Temperature Ablation

*Why SAC dominates competitive tag — and why a fixed entropy coefficient destroys learning.*

---

## Motivation

Our [prior studies](../reward_shaping/) showed that SAC dominates PPO 95-to-2 in cross-algorithm play, regardless of reward shaping or zoo training. The natural hypothesis: SAC's entropy bonus acts as implicit reward shaping, encouraging diverse behavior that prevents degenerate strategies like corner-camping.

But is entropy actually the mechanism? This study tests that directly.

---

## Experiment 1A: The Entropy Counterfactual

We trained SAC on **R4 Sparse** (no reward shaping — only terminal tag/timeout rewards) under three entropy conditions:

| Condition | Description |
|-----------|-------------|
| **alpha=0** | Entropy disabled entirely — SAC becomes a pure actor-critic |
| **alpha=0.1** | Fixed moderate entropy — no auto-tuning |
| **control** | Standard SAC with automatic entropy tuning (init=0.607) |

Each condition: 3 seeds, 5M timesteps, Optuna-optimized hyperparameters, four_corners layout with HSM=1.15.

### Alpha Trajectories

<p align="center">
  <img src="alpha_counterfactual.png" alt="Entropy temperature dynamics for the three conditions" width="700">
</p>

The auto-tuned control starts at alpha~0.57 and **crashes to ~0.003 within the first 500K steps**. Meanwhile, the fixed conditions stay flat at their assigned values.

The auto-tuned alpha settles 30x lower than the fixed alpha=0.1 condition. This is the first clue: whatever SAC needs from entropy, it's a *brief* high-entropy phase, not sustained exploration.

### Cross-Evaluation Results

All agents were cross-evaluated in an 11x11 gauntlet (50 episodes per matchup). Results for the 1A conditions:

| Condition | Seeker strength | Hider survival | Combined | Wall hugging |
|-----------|:--------------:|:--------------:|:--------:|:------------:|
| **control** (auto-tuned) | **39.3%** | **74.9%** | **57.1%** | 0.78 |
| alpha=0 (no entropy) | 30.0% | 62.0% | 46.0% | 0.67 |
| alpha=0.1 (fixed) | 1.5% | 18.4% | 9.9% | 0.43 |

**Key findings:**

1. **Auto-tuned entropy produces the strongest agents.** Both seeker and hider are substantially better than the other conditions.

2. **Fixed alpha=0.1 is catastrophically bad** — worse than *no entropy at all*. The seeker barely catches anyone (1.5%), and the hider survives only 18.4% of matches. A moderate fixed entropy coefficient doesn't just fail to help; it actively prevents learning.

3. **No entropy (alpha=0) is functional but weaker.** Agents learn reasonable strategies but can't match the auto-tuned version.

4. **Wall hugging doesn't tell the expected story.** Auto-tuned agents wall-hug *more* (0.78) than no-entropy agents (0.67). Entropy is not preventing degenerate wall-camping — the mechanism is something else.

---

## Experiment 1B: Alpha Dynamics Across Reward Presets

We trained SAC with auto-tuned entropy across all 8 reward presets (3 seeds each, 5M steps) to see whether the reward function affects the learned entropy schedule.

### Alpha Trajectories by Preset

<p align="center">
  <img src="alpha_by_preset.png" alt="Alpha dynamics across all 8 presets" width="700">
</p>

All 8 presets show **nearly identical alpha trajectories**: rapid decay from ~0.57 to ~0.003 within the first 500K steps, regardless of the reward function. The reward function does not meaningfully change the entropy schedule.

### Converged Entropy by Preset

<p align="center">
  <img src="final_alpha_by_preset.png" alt="Final alpha values by preset" width="600">
</p>

Final alpha values cluster tightly between 0.002 and 0.004. R3 (Both Shaped) converges slightly higher; R4 (Sparse) slightly lower. But the differences are small — the entropy schedule is driven by the competitive dynamics, not the reward function.

### Cross-Evaluation Rankings

| Preset | Combined strength | Description |
|--------|:-----------------:|-------------|
| R2 Hider Active | **65.1%** | Anti-degenerate shaping (wall penalty + speed bonus) |
| R7 Kitchen Sink | 60.2% | All shaping terms combined |
| R4 Sparse | 60.0% | No shaping at all |
| R1 Seeker Pursuit | 58.9% | Strong distance pursuit signal |
| R0 Baseline | 58.8% | Minimal (small time penalty + survival bonus) |
| R5 Escalating | 46.8% | Escalating time pressure |
| R3 Both Shaped | 44.6% | Seeker pursuit + hider evasion combined |
| R6 Coverage | 42.5% | Exploration/area coverage bonus |

R4 Sparse (no shaping) ranks 3rd — nearly as strong as the best shaped reward. With SAC, **reward shaping is largely unnecessary**. But some shaping actively hurts: R3 (both agents shaped), R5 (escalating), and R6 (coverage) all perform *worse* than no shaping.

---

## The Mechanistic Story

### Entropy Tracks Competitive Dynamics

<p align="center">
  <img src="alpha_vs_swr.png" alt="Alpha trajectory overlaid with seeker win rate" width="550">
</p>

The alpha trajectory (blue) decays rapidly in the first 500K steps while the seeker win rate (red) oscillates wildly throughout training. The competitive arms race plays out *after* entropy has already collapsed to near-zero.

### Why Fixed Entropy Fails

The auto-tuned schedule has three phases:

1. **High entropy (0-200K steps):** Both agents explore widely, building diverse experience in the replay buffer. This bootstraps learning from the sparse terminal rewards.

2. **Rapid decay (200K-500K steps):** As policies improve, the entropy bonus becomes a distraction. Auto-tuning reduces alpha to get out of the way.

3. **Near-zero entropy (500K+ steps):** The competitive dynamics take over. Agents refine strategies against each other with minimal entropy interference.

Fixed alpha=0.1 is catastrophic because it **prevents phase 2 from completing**. The sustained entropy bonus keeps policies stochastic long after they should be converging, and in a competitive setting this means neither agent can develop the precise, committed strategies needed to catch or evade an opponent.

Fixed alpha=0 skips phase 1 entirely, which means agents miss the initial exploration that seeds the replay buffer with diverse transitions. They still learn — terminal rewards provide enough signal — but they converge to weaker strategies.

---

## Summary

| Finding | Implication |
|---------|------------|
| Auto-tuned alpha decays to ~0.003 | SAC needs a brief exploration phase, not sustained entropy |
| Fixed alpha=0.1 is catastrophic | Moderate fixed entropy prevents policy convergence in competitive games |
| Alpha trajectory is identical across 8 presets | The entropy schedule is driven by competitive dynamics, not reward design |
| R4 Sparse ranks 3rd with SAC | Reward shaping is largely unnecessary when using auto-tuned SAC |
| Auto-tuned agents wall-hug more | Entropy is not preventing degenerate behavior — the mechanism is exploration bootstrapping |

The critical design choice is not the reward function or the entropy coefficient — it's **letting the entropy temperature adapt**. In competitive self-play, the optimal entropy schedule is a rapid decay that SAC discovers automatically.

---

## Experimental Details

- **Environment:** four_corners layout, HSM=1.15, 200-step episodes
- **Algorithm:** SAC with Optuna-optimized hyperparameters (lr=2.25e-4, gamma=0.969, tau=0.00658, init_alpha=0.607, buffer=100K, batch=256, updates_per_step=4)
- **Training:** 5M timesteps per run, 64 parallel environments
- **Evaluation:** 11x11 cross-evaluation gauntlet, 50 episodes per matchup
- **Total runs:** 33 (9 counterfactual + 24 preset dynamics)
- **Compute:** LUMI supercomputer, ~33 CPU-hours
