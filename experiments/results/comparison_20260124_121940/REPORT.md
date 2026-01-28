# Experiment Report: Vanilla Self-Play vs SCRO Training

**Date:** January 24, 2026
**Experiment ID:** comparison_20260124_121940

## Overview

This experiment compares two training approaches for training tag game agents:

1. **Vanilla Self-Play**: Standard PPO self-play where seeker and hider train against each other
2. **SCRO (Sandwich Coral Reef Optimization)**: Spatially-structured coevolutionary algorithm with population diversity

## Experimental Setup

### Training Configuration

| Parameter | Vanilla Self-Play | SCRO |
|-----------|-------------------|------|
| Total Timesteps | 552,960 | ~450,000 |
| Training Method | PPO self-play | PPO + CRO operators |
| Population Size | 1 seeker, 1 hider | 9 seekers, 9 hiders (3x3 grid) |
| Generations | N/A | 15 |
| Training Steps/Gen | N/A | 2,048 per agent |
| Seeds | 42, 123 | 42, 123 |

### Environment

- **Arena**: 15x15 units (-7.5 to 7.5)
- **Tag Distance**: 1.5 units
- **Time Limit**: 10 seconds per episode
- **Observation**: 84-dimensional (ray-based vision)
- **Action**: 3-dimensional continuous (acceleration x, y, rotation)

## Training Results

### Learning Curves

![Learning Curves](learning_curves.png)

### Final Win Rates (During Training)

| Algorithm | Seed 42 | Seed 123 | Mean |
|-----------|---------|----------|------|
| Vanilla Self-Play | 85.7%* | 91.9%* | 88.8% |
| SCRO | 27.5% | 22.9% | 25.2% |

*Note: Vanilla training crashed before completion due to numerical instability (NaN in policy network), but achieved high win rates before crashing.

### Training Stability

- **Vanilla Self-Play**: Both runs crashed with NaN values around update 240-260
- **SCRO**: Completed successfully without crashes

## Cross-Play Evaluation

To assess true agent quality, we evaluated all seeker-hider combinations:

### Win Rate Matrix (200 episodes each)

|                    | Vanilla Hider | SCRO Hider |
|--------------------|---------------|------------|
| **Vanilla Seeker** | 91.5%         | 98.5%      |
| **SCRO Seeker**    | 59.5%         | 57.5%      |

### Detailed Matchup Statistics

| Matchup | Seeker Win% | Avg Episode Length | Seeker Reward |
|---------|-------------|-------------------|---------------|
| Vanilla vs Vanilla | 91.5% | 91.3 steps | +8.73 |
| Vanilla vs SCRO | 98.5% | 76.3 steps | +9.93 |
| SCRO vs Vanilla | 59.5% | 140.6 steps | +1.99 |
| SCRO vs SCRO | 57.5% | 123.5 steps | +1.29 |

### Aggregate Performance

| Metric | Vanilla | SCRO |
|--------|---------|------|
| **Seeker avg win rate** | 95.0% | 58.5% |
| **Hider avg survival rate** | 24.5% | 22.0% |

## Key Findings

### 1. Vanilla Self-Play Produces Stronger Agents

The vanilla-trained seeker dominates all matchups with 91-99% win rates. This suggests that focused self-play training, despite maintaining high imbalance, produces more capable policies.

### 2. SCRO Maintains Balance But Not Strength

SCRO's coevolutionary dynamics maintain ~50-60% seeker win rates, but this balance doesn't translate to competitive strength. SCRO agents perform significantly worse in cross-play.

### 3. SCRO Hiders Are Surprisingly Weak

Counter-intuitively, SCRO-trained hiders are *easier* to catch (98.5% loss rate vs vanilla seeker) than vanilla-trained hiders (91.5% loss rate). The coevolutionary pressure may have led to strategies that exploit specific SCRO seeker weaknesses rather than developing generally robust evasion.

### 4. Specialization vs Generalization Trade-off

- **Vanilla**: Agents specialize against their training partner, but this specialization generalizes well
- **SCRO**: Agents co-adapt to maintain equilibrium, but this creates strategies that don't transfer

### 5. Training Stability

SCRO completed without crashes while vanilla training encountered NaN values. This suggests SCRO's population-based approach provides more stable gradients, even if individual agents are weaker.

## Conclusions

For this tag game:

1. **Use vanilla self-play for maximum performance** if you want strong agents
2. **Use SCRO for stable training** if numerical stability is a concern
3. **High win-rate during training correlates with cross-play strength**
4. **Balanced coevolution does not guarantee robust strategies**

## Future Work

- Fix numerical instability in vanilla training (gradient clipping, learning rate tuning)
- Investigate why SCRO hiders are weak (analyze learned behaviors)
- Test with longer SCRO training to see if agents eventually become stronger
- Compare with other coevolutionary algorithms (e.g., competitive coevolution without spatial structure)

## Files

- `learning_curves.png` - Training progression comparison
- `final_comparison.png` - Box plot of final win rates
- `statistics.txt` - Statistical summary
- `cross_evaluation.json` - Detailed cross-play results
- `experiment_summary.json` - Experiment configuration
- `vanilla_selfplay/` - Vanilla training checkpoints and metrics
- `scro/` - SCRO training checkpoints and metrics
