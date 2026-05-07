# Design C — Pre-Registration

**Version:** v1 (2026-05-07)
**Author:** J. Quick
**Status:** DRAFT — pending pilot result before main grid commits

This document is committed to the repo *before* any of the 24 main runs are launched.
Any change to it after the pilot result is observed must be recorded as a v2 with
explicit rationale, and the original v1 must remain in git history.

---

## 1. Question

Does zoo training (parameter `A`, the probability of training against a random
zoo snapshot rather than the live partner) substitute for reward shaping —
asymmetrically by algorithm class (PPO vs SAC) — in the 2D pursuit-evasion
("tag") environment?

Operationalised as the existence of a non-zero reward × A interaction effect
on cross-pool seeker win rate, evaluated separately for PPO and SAC.

## 2. Hypotheses

- **H₀ (per algorithm):** the reward × A interaction coefficient is zero. Reward
  shaping and zoo training contribute additively (or one of them not at all).
- **H₁ (per algorithm):** the interaction coefficient is non-zero. The benefit
  of shaping over the sparse reward changes with zoo intensity.
- **Directional prediction (not pre-registered as a hypothesis, but disclosed):**
  the substitution direction — shaping helps more at A=0 than at A=0.5 — is
  expected, with a stronger effect for SAC (replay buffer ≈ implicit zoo).

## 3. Design

Fully crossed 2 × 2 × 2 × 3 = **24 main-grid runs**:

| factor | levels | rationale |
|---|---|---|
| reward | `R4_sparse`, `R7_kitchen_sink` | extreme contrast on shaping axis (per critic) |
| zoo intensity A | 0.0, 0.5 | extreme contrast within trained-stable regime |
| algorithm | PPO, SAC | the asymmetry hypothesis |
| seed | 0, 1, 2 | minimum for variance estimate |

**Held constant:**
- Layout: `four_corners`
- HSM: 1.15 (matches Study 2)
- Training steps: 5,000,000 per run (matches Study 2)
- Zoo size: 50; zoo update interval: 50; sampling strategy: uniform
- Trainer scripts: `trainer/train_zoo.py` (PPO) and `trainer/train_zoo_sac.py` (SAC)
  used for *both* A=0 and A=0.5 cells, so the pipeline is identical across A.
  At A=0 the zoo is populated but never sampled (`zoo_prob=0`).

**Disclosed scope limitations:**
- Single HSM. Any reward × A interaction we report is conditional on HSM=1.15.
- Two reward presets, not a reward axis. We test for *existence* of an
  interaction, not a surface.
- SAC hyperparameters are not tuned for tag. The PPO vs SAC contrast is
  conditional on default-HP SAC, validated by the pilot (§5).

## 4. Outcome metric and analysis model

### 4.1 Primary outcome

Per-matchup binary outcome `y_{i,j,k} ∈ {0, 1}` = "did the seeker win?",
collected by playing each `(seeker, hider)` pair across the gauntlet
reference pool for `n_eval` episodes per matchup (see §6 for fidelity).

### 4.2 Reference pool

The cross-evaluation pool consists of:
- All 24 main-grid final seekers and 24 final hiders (one per run).
- 4 fixed **anchor policies** trained at the corners of the design space
  *before the main grid* and frozen — same data for all subsequent analyses
  (regardless of any future runs). Anchors break circular self-reference.
  Anchors: `{R4 PPO A=0, R4 SAC A=0, R7 PPO A=0.5, R7 SAC A=0.5}`, each at
  seed=42 (a seed not used in the main grid).

If usable Study 2 final checkpoints exist on disk at launch time, they are
*also* added to the pool as additional fixed reference. Their inclusion is
recorded but does not replace anchors.

### 4.3 Statistical model

A Bayesian (or REML — both reported) **logistic mixed-effects model**, fit
separately per algorithm:

```
y_{i,j,k} ~ Bernoulli(p_{i,j,k})
logit(p_{i,j,k}) = β_0
                 + β_R * reward_i              # 0 = R4, 1 = R7
                 + β_A * A_i                   # 0 = A=0,  1 = A=0.5
                 + β_RA * reward_i * A_i       # PRIMARY ESTIMAND
                 + u_{seed[i]}                 # random intercept per training seed
                 + v_{opp[j]}                  # random intercept per pool opponent
                 + γ * is_self_pair_{i,j}      # nuisance: drop or fixed effect
```

`β_RA` is the **single primary estimand** per algorithm.

Implementation: `statsmodels.BinomialBayesMixedGLM` or `pymc` for Bayesian;
`statsmodels.MixedLM` with logit link via GLMM for REML. Both reported; if
they disagree on the decision, default to the Bayesian posterior interval.

### 4.4 Self-pair handling

Diagonal entries (run i's seeker vs its own final hider) are EXCLUDED from
the analysis dataset. Recorded but not modeled, because they are not exchangeable
with the rest of the matrix.

## 5. Pre-flight pilot (gate before main grid)

A single SAC training run is launched **before** any of the 24 main-grid jobs:

- Cell: `R4_sparse`, A = 0.0, algorithm = SAC, seed = 0
- Steps: **10,000,000** (2× the main-grid budget) to verify plateau
- Output: `experiments/results/design_c/pilot/sac_R4_A0_seed0/`

### Pilot decision rule

Looking at the pilot's seeker WR vs the live partner over training:

- **PASS:** WR has plateaued by 4M steps (rolling-mean slope over the last 1M
  steps is < 0.02 / 1M-steps in absolute value). Proceed with main grid at 5M
  steps as planned.
- **EXTEND:** WR is still climbing at 4M steps but plateaus by 8M. Escalate
  main-grid budget to 7.5M steps for SAC only. Recompute LUMI cost; user
  approval required before grid launch.
- **HP-BROKEN:** WR < 0.30 at 10M steps OR loss / alpha is unstable. The
  SAC arm is HP-confounded; we do **not** infer "no interaction" from a
  null SAC result. Either tune SAC (separate effort) or drop SAC from
  Design C and document.

Pilot also serves as a wall-clock / GPU-hour calibration for the grid.

## 6. Gauntlet fidelity (successive halving on episodes)

To get tight CIs on `β_RA` without burning compute on already-decided
matchups, the gauntlet runs in stages:

| stage | episodes per matchup (cumulative) | matchups continued |
|---|---|---|
| 1 | 10 | all (576 + anchor pairs) |
| 2 | 30 | matchups whose Wilson 95% CI for p crosses 0.5 ± 0.10 |
| 3 | 100 | matchups whose CI still crosses 0.5 ± 0.05 |

This concentrates compute on the ambiguous matchups that the model is
sensitive to, and matches the critic's "successive halving where it actually
fits" recommendation.

## 7. Pre-registered decision rule

**For each algorithm independently, on the posterior of `β_RA`:**

| outcome | rule | next step |
|---|---|---|
| **PURSUE Design A** | `|β_RA|` posterior median ≥ log(1.25) ≈ 0.22 AND 95% CrI excludes 0 | for at least one algorithm | proceed to ~700 GPU-hr Design A |
| **REFINE** | `|β_RA|` between log(1.10) and log(1.25) for either algo, OR borderline CrI | add 3 more seeds (12 more main-grid runs, ~30 GPU-hr) and re-fit |
| **KILL** | `|β_RA|` < log(1.10) ≈ 0.10 AND 95% CrI includes 0 for both algorithms | document and stop. Rewards × A do not interact in this env at HSM=1.15. |

Log-odds thresholds correspond approximately to ≥5pp / 2pp differences in
seeker WR around p=0.5 — translating the critic's WR-thresholds into the
appropriate model space.

## 8. Compute budget (pre-registered)

| component | runs | GPU-hr each | subtotal |
|---|---|---|---|
| Pilot (SAC, 10M) | 1 | ~6 | 6 |
| Anchors (mixed, 5M) | 4 | ~2.5 avg | 10 |
| Main grid PPO | 12 | ~2 | 24 |
| Main grid SAC | 12 | ~3 | 36 |
| Gauntlet (3-stage) | — | — | ~2 |
| **Total** | 29 | | **~78 GPU-hr** |

If REFINE is triggered: + 12 runs ≈ +30 GPU-hr.

## 9. What we will publish regardless of outcome

- Full anonymized matchup outcomes CSV
- The fitted model coefficients (both REML and Bayesian) with CIs
- The decision rule outcome (PURSUE / REFINE / KILL)
- Disclosure of any departure from this pre-registration

A null result (KILL) is publishable as "in this env, reward shaping and zoo
training do NOT interact at the levels we tested" — that is the value of
pre-registration.

## 10. Stopping rules / amendment policy

- The pilot result IS allowed to gate the main-grid launch (this is part of
  the design, not an amendment).
- After the main grid is launched, no changes to the model spec, the decision
  rule, or the reference pool composition are allowed without a v2 of this
  document and a written rationale committed before viewing the new data.
