# Design C — Results

**Status:** Draft (2026-05-14). Pre-reg v1, v2 frozen in git; this document
reports the realized results and their interpretation. Subject to revision
once the trajectory/behavior analysis lands.

---

## TL;DR

Three findings, in increasing order of how publishable they are:

1. **Registered confirmatory test (β_RA, the reward × A interaction):**
   posterior median = 1.39 log-odds, 95% CrI [-0.13, +2.94], P(>0) = 0.963.
   By the pre-reg's literal decision rule, the 95% CrI doesn't exclude zero —
   so the formal verdict is **REFINE**, not PURSUE. P(>0) = 96.3% is just
   under the conventional one-sided 95% threshold.

2. **Reward-induced heteroscedasticity (σ_seeker[R7] ≫ σ_seeker[R4]):**
   σ_seeker[R7] = 1.75 vs σ_seeker[R4] = 0.43 (log-odds), 95% CrIs non-overlapping.
   In WR units that's ±35pp vs ±9pp around the cell mean. **R7 produces
   seekers with ~4× more between-seed variance in policy quality than R4.**

3. **SVD structure of the cross-pool WR matrix:** σ²₁ explains 61% of
   centered variance (transitive skill axis), σ²₂ 16% (residual cyclic
   structure), σ²₃ 11%, σ²₄ 6%. Effective rank ≈ 3. The R7 cluster spreads
   across multiple SVD dimensions; the R4 cluster collapses to a single
   point.

Finding 2 + 3 together = the actual paper claim. Finding 1 is the
registered claim that motivated the study; we report it honestly.

---

## Experimental design

| factor | levels |
|---|---|
| reward shaping | R4_sparse, R7_kitchen_sink |
| zoo intensity A | 0.0 (pure self-play), 0.5 (50% zoo) |
| algorithm | **PPO only** (SAC dropped after pilot, see §"SAC pilots") |
| seeds | 0, 1, 2, 3, 4 + anchor seed 42 (R4) + R7-only seeds 5, 6, 7, 8 |

**Held constant:** HSM = 1.15, layout `four_corners`, 5M timesteps, zoo size 50,
uniform sampling. Pre-reg v1, v2 record full settings; v2 documents the
PPO-only descope.

**Total training:** 32 PPO runs (10 R4 + 18 R7 + 4 anchors), ~32 CPU-hr.

**Gauntlet:** every-vs-every 32 × 32 = 1,024 matchups (less 32 self-pair) =
992 matchups × ~100 episodes via 3-stage successive halving (10 → 30 → 100
episodes, escalating on Wilson CI ambiguity). **99,200 episode outcomes
total** form the analysis dataset.

---

## §1 — Registered test: β_RA

Per-cell mean seeker WR vs the 32-policy reference pool (self-pair excluded):

|  | A = 0.0 | A = 0.5 | Δ_A |
|---|---:|---:|---:|
| R4_sparse | 0.259 | 0.253 | -0.006 |
| R7_kitchen_sink | 0.620 | 0.758 | +0.138 |

Raw-scale interaction Δ-of-Δ = **+0.144 pp WR**.

**Logistic mixed-effects model** (per pre-reg v2 §A5, with the critic-recommended
heteroscedastic σ_seeker[reward] amendment per §A6 REFINE branch):

```
y_e ~ Bernoulli(σ⁻¹(η_e))
η_e = β₀ + β_R · R_e + β_A · A_e + β_RA · (R · A)_e
    + α_seeker[i_e] + α_hider[j_e]
α_seeker[i] ~ Normal(0, σ_s[R_i])     # heteroscedastic by training reward
α_hider[j]  ~ Normal(0, σ_h)
β_*  ~ Normal(0, 2.5)
σ_*  ~ HalfNormal(2.0)
```

Fit with PyMC NUTS (4 chains × 1500 draws + 1000 tune), R̂ ≤ 1.006, ESS > 600.

| parameter | posterior median | 95% CrI |
|---|---:|---|
| β₀ (intercept) | -1.10 | [-1.48, -0.72] |
| β_R (R7 main effect) | **+1.71** | [+0.61, +2.77] |
| β_A (zoo main effect) | -0.03 | [-0.52, +0.49] |
| **β_RA (interaction)** | **+1.39** | **[-0.13, +2.94]** |
| σ_seeker[R4] | 0.43 | [0.25, 0.67] |
| σ_seeker[R7] | **1.75** | **[1.21, 2.38]** |
| σ_hider | 0.34 | [0.26, 0.44] |

**Verdict (pre-reg v2 §A6 thresholds):** REFINE. |β_RA| = 1.39 ≫ 0.223
(PURSUE magnitude threshold), but the 95% CrI just barely brackets zero
(lower bound -0.13). P(β_RA > 0) = 0.963.

**Honest disclosure:** the registered hypothesis was that zoo and shaping
*substitute* (i.e., β_RA < 0, with R7's advantage shrinking under zoo).
The data show the opposite direction at 96.3% posterior confidence —
the gap *grows*. They look like **complements**, not substitutes. The
formal CrI just barely doesn't certify even the direction.

---

## §2 — The heteroscedasticity finding

σ_seeker[R7] / σ_seeker[R4] = 1.75 / 0.43 = **4.07** (in log-odds).
95% CrIs of the two parameters are non-overlapping: R4 [0.25, 0.67]
vs R7 [1.21, 2.38].

In WR terms (at the cell mean), one standard deviation around the
seeker-level intercept is:

- R4 seekers: ±9pp around the cell mean WR
- R7 seekers: ±35pp around the cell mean WR

Concretely, examining the 18 R7 final seekers individually:
- Top: R7/A=0.5/seed_0 reaches **0.99** gauntlet WR
- Bottom: R7/A=0.0/seed_4 reaches **0.28** gauntlet WR (below mean R4!)

The R4 final seekers all sit in [0.19, 0.30] gauntlet WR. The R4 cluster
has a unique attractor; the R7 cluster is multimodal.

This finding is robust across the data refresh (24 policies → 32
policies; pooled σ → heteroscedastic σ). It is *not* an artifact of
unlucky seeds — adding 8 more R7 seeds (5–8) reproduced the spread.

---

## §3 — SVD / spinning-top structure

Singular value decomposition of the centered 32 × 32 cross-pool WR matrix:

| component | σ² share | cumulative |
|---|---:|---:|
| 1 (transitive skill) | 0.613 | 0.613 |
| 2 (residual structure) | 0.164 | 0.777 |
| 3 (residual structure) | 0.107 | 0.884 |
| 4 (residual structure) | 0.058 | 0.942 |
| 5+ | < 0.015 each | — |

Effective rank ≈ 3. Per Czarnecki et al. 2020 "Real World Games Look Like
Spinning Tops": this is the canonical "weakly non-transitive" pattern —
mostly a single skill axis, with meaningful but lower-energy cyclic
structure underneath.

The top-2 SVD embedding shows R4 as a **tight cluster** (low U1, U2 ≈ 0)
and R7 spreading into **3–4 sub-clusters** along both U1 (skill) and U2
(residual structure). The 8 new R7 seeds (5–8) distributed across the
same spread as the original R7 seeds, confirming the multimodality is
a stable property of training under R7, not an outlier artifact.

---

## §4 — SAC pilots (negative result, fully disclosed)

Three pre-flight SAC pilots ran before the main grid:

| pilot | algo | reward | A | result |
|---|---|---|---|---|
| 18497435 | SAC | R4_sparse | 0.0 | peak 55% @ 1–3M → collapse to 14% by 10M; α → 0.001 |
| 18498830 | SAC | R7_kitchen_sink | 0.0 | peak 29% @ 3M → collapse to 8.7% by 10M; α → 0.001 |
| 18498831 | PPO | R4_sparse | 0.0 | 31% @ 1M → 69% @ 5M; **stable through 10M** |

Default-HP SAC collapses in this self-play env regardless of reward
density (R4 *and* R7 collapse identically). PPO learns the same task
cleanly. Per pre-reg v1 §5 HP-BROKEN clause, SAC was dropped from
Design C. Tuning SAC HPs for self-play is a separate effort outside
this paper's scope.

---

## §5 — What we did NOT find (anti-claims)

- **Substitution.** The registered hypothesis (zoo substitutes for
  shaping, β_RA < 0) is rejected directionally. Data is consistent with
  complementarity (β_RA > 0, P(>0) = 0.963).
- **Zoo alone helps.** β_A = -0.03 with CrI [-0.52, +0.49] — under
  sparse reward, zoo training has no meaningful effect.
- **Algorithm-class asymmetry.** Cannot be tested here because SAC is
  HP-broken. Open question.

---

## §6 — Mechanism (pending)

The heteroscedasticity finding raises the question: *why* does R7
produce a wide policy-quality distribution? Two non-mutually-exclusive
hypotheses:

- **Exploration variance.** Rich reward enables broader exploration; some
  seeds find better local optima than others. Implied test: trajectory
  features of high-WR R7 seekers cluster together (same dominant
  strategy, just discovered or not).
- **Strategic diversity.** Rich reward admits multiple stable strategies
  that all perform well in self-play but differ in cross-pool
  performance. Implied test: high-WR R7 seekers cluster *separately*
  by behavioral fingerprint, indicating distinct strategies.

`experiments/design_c_trajectory_analysis.py` is being run now: for each
of the 32 final seekers, roll out 30 episodes vs a fixed reference
hider, collect trajectory features (path length, mean speed, time near
center, time near walls, mean seeker-hider distance, final position),
PCA the behavior space, and check whether behavior PC1 aligns with the
SVD outcome U1.

**Expected outcomes** (to be falsified by data, not advocated):
- High |corr(behavior PC1, SVD U1)| → exploration-variance story
- Low |corr(behavior PC1, SVD U1)| AND visible R7 sub-clustering in
  behavior space → strategic-diversity story
- Both clear → both mechanisms contribute (most likely outcome
  given SVD effective rank ≈ 3)

---

## §7 — Methodological lessons

Surfaced during the analysis arc; worth a methods section:

1. **Training-time WR is uninformative across seeds in self-play.** Every
   R7 seeker reached 88–96% training-time WR vs *its own* hider, but
   cross-pool WR ranged 28–99% across seeds. Training-time WR measures
   performance against a seed-specific opponent; it isn't commensurable
   across runs. Only the gauntlet measures policy quality.

2. **In self-play, the seed is not a noise term — it's a draw from a
   distribution over objectives.** Each seed's hider defines a different
   evaluation distribution for its seeker. A factorial design with seed
   replicates implicitly assumes homoscedasticity across cells; that
   assumption can fail dramatically (here, σ ratio = 4×), and modeling
   σ_cell explicitly is necessary, not a refinement.

3. **Variational Bayes underestimates posterior variance with hierarchical
   structure.** Our pre-reg listed VB as a fallback and required MCMC
   when available. The VB CrI for β_RA was ~10× too narrow vs proper
   NUTS. Beware.

4. **Pre-registration earned its keep.** Without v1/v2 frozen in git
   history, the directional flip (substitution → complementarity) and
   the "elevate heteroscedasticity from a nuisance to a finding" move
   would both look like post-hoc rationalization. With frozen pre-regs,
   the paper can transparently say "registered confirmatory result =
   REFINE; here's the exploratory follow-up."

---

## §8 — Compute spent (post-hoc)

| step | runs | wall-each | CPU-hr |
|---|---:|---:|---:|
| SAC pilots (3) | 3 | ~2h | 6 |
| PPO pilot | 1 | ~2h | 2 |
| Anchors | 4 | ~1h | 4 |
| Main grid (seeds 0–2) | 12 | ~1h | 12 |
| Refine round 1 (seeds 3–4) | 8 | ~1h | 8 |
| Refine round 2 (R7 seeds 5–8) | 8 | ~1h | 8 |
| Gauntlets (1st 24-pol + 2nd 32-pol) | 2 | 17–47 min | ~1 |
| Analysis (GLM + VB + MCMC ×2) | 4 | 1–3h | ~5 |
| **Total** | | | **~46 CPU-hr** |

Budget claimed up front (pre-reg v2 §A7): ~17 CPU-hr. Actual ~46.
Overrun driven by: (a) SAC pilots (6 CPU-hr) that the original
estimate undercounted, (b) two MCMC runs and the second one needed
6h walltime, (c) the second gauntlet that the refine round triggered.

Worth noting for future studies: budgets should include 2× safety
margin on Bayesian inference; NUTS scales worse than linearly in
parameter count.
