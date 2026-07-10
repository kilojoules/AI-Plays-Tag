# Design C — Pre-registration v3: power extension (confirmatory)

**Frozen:** 2026-06-12, before any v3 run was launched. Amends prereg v2;
everything not stated here is inherited unchanged (env, HSM = 1.15,
four_corners, 5M steps, zoo interval 50 / max-size 50 / uniform, PPO
settings, reward definitions per `design_c_grid_tasks.py` REWARD_ARGS).

## Motivation

The registered estimand β_RA landed REFINE twice with consistent
magnitude: gauntlet NUTS +1.39 [-0.13, +2.94] (Part I), anchor-panel
NUTS +1.70 [-0.46, +3.88] (Part II, `panel_mcmc_A.txt`). The posterior
between-run sigmas from the panel fit (σ_run[R4] = 0.92, σ_run[R7] =
2.30) imply the study is run-limited: se(β_RA) ≈ sqrt(2σ²_R7/n +
2σ²_R4/n) ≈ 1.1 at the current n = 5–9 per cell. This extension adds
seeds to the four prereg cells only.

## Design

| cell | existing seeds | new seeds | n after |
|---|---|---|---:|
| R4_sparse, A=0.0 | 0–4 | 5–19 | 20 |
| R4_sparse, A=0.5 | 0–4 | 5–19 | 20 |
| R7_kitchen_sink, A=0.0 | 0–8 | 9–19 | 20 |
| R7_kitchen_sink, A=0.5 | 0–8 | 9–19 | 20 |

52 new runs total (`design_c_power_extension_tasks.py`), trained on DTU
gbar (LSF, node-local /tmp pattern). Hardware change from LUMI is
acknowledged; the training stack (pixi env, identical CLI args, same
trainer commit) is unchanged, and §Analysis includes a hardware-batch
sanity check.

At n = 20/cell: se(β_RA) ≈ 0.78, 95% CrI half-width ≈ 1.53 — certifies
an effect of the size estimated twice (≥ ~1.55), with the explicit risk
that a smaller true effect again lands REFINE. No further extension
will be bolted on post hoc; a third REFINE means the paper reports
REFINE.

## Stopping rule

Fixed-n. All 52 runs train to 5M steps; failed/crashed runs are
resubmitted, never replaced with different seeds. No interim fits of
Model A before all 52 runs complete (training-progress monitoring is
allowed; win-rate analysis is not).

## Evaluation and analysis (pre-specified)

1. Eval: `design_c_anchor_panel_eval.py` — each new seeker vs the same
   4 prereg anchors (+ own hider, for the descriptive classification),
   30 episodes per pair, appended to `anchor_panel.csv`.
2. Confirmatory fit: `design_c_panel_mcmc.py --model A` exactly as
   committed at freeze time — PyMC NUTS, Binomial likelihood, random
   intercepts per run and per anchor, σ_run stratified by reward,
   priors Normal(0, 2.5) / HalfNormal(2), draws=2000 × 4 chains,
   random_seed=42.
3. Decision rule: prereg §7 unchanged — PURSUE if |β_RA| ≥ log(1.25)
   and 95% CrI excludes 0; KILL if |β_RA| < log(1.10) and CrI includes
   0; else REFINE.
4. Sanity check (reported, not gating): refit with a
   `trained_on_gbar` indicator added to the run-intercept mean for the
   R7/A cells that span both machines; flag if its CrI excludes 0.
5. The run-classification (healthy / basin / over-specialization)
   counts will be updated descriptively; no new claims hang on them.
