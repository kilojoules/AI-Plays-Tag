# Design C — Results

**Status:** Draft v4 (2026-07-10). Part I (§1–§8) is the 2026-05-14 draft,
kept verbatim except for bracketed `[Correction 2026-07: …]` notes. Part II
(§9–§15) reports the failure-mode arc: the 4-anchor eval correction, the
basin/over-specialization taxonomy, the A dose-response, the coverage×urgency
factorial, and the HSM flank. **§16 reports the prereg v3 power extension,
which resolves the registered confirmatory test: PURSUE**
(β_RA = +2.05 [+0.72, +3.45]). **§18 records the 2026-07 adversarial review:
errata, code bugs, pre-registration deviations, and the follow-up experiments
they triggered.** The claims ledger in §15 (updated) is the authoritative
summary, read jointly with §18.

Provenance note (corrects the earlier "frozen in git" wording): pre-reg v1
and v2 were frozen in git before their data. The v3 pre-reg and all Part II
work were authored on LUMI scratch/gbar *outside version control* — v3's
freeze (2026-06-12, before the extension launched) is evidenced by file
mtimes and the LSF submission record, not by a commit hash. Everything was
imported into git on 2026-07-10.

---

## TL;DR

Three findings, in increasing order of how publishable they are:

1. **Registered confirmatory test (β_RA, the reward × A interaction):**
   posterior median = 1.39 log-odds, 95% CrI [-0.13, +2.94], P(>0) = 0.963.
   By the pre-reg's literal decision rule, the 95% CrI doesn't exclude zero —
   so the formal verdict is **REFINE**, not PURSUE. [Correction 2026-07: the
   original text said 96.3% was "just under the conventional one-sided 95%
   threshold" — it is *above* 95%; the relevant bar is the 97.5% one-sided
   level implied by the two-sided CrI rule, which it is under. Superseded by
   the §16 PURSUE at n=20/cell.]

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
| seeds | 0, 1, 2, 3, 4 + anchor seed 42 (both rewards × both A) + R7-only seeds 5, 6, 7, 8 [Correction 2026-07: original said "anchor seed 42 (R4)"; the 4 anchors span R4/R7 × A=0/0.5 per pre-reg v2 §A4 — the row's own 10+18+4=32 arithmetic only works that way] |

**Held constant:** HSM = 1.15, layout `four_corners`, 5M timesteps, zoo size 50,
uniform sampling. Pre-reg v1, v2 record full settings; v2 documents the
PPO-only descope.

**Total training:** 32 PPO runs (10 R4 + 18 R7 + 4 anchors), ~32 CPU-hr.

**Gauntlet:** every-vs-every 32 × 32 = 1,024 matchups (less 32 self-pair) =
992 matchups × 100 episodes each. **99,200 episode outcomes total** form the
analysis dataset. [Correction 2026-07: the original described "3-stage
successive halving (10 → 30 → 100, escalating on Wilson CI ambiguity)". The
escalation thresholds were mathematically unreachable (min Wilson half-width
0.139 at n=10 > 0.10; 0.057 at n=30 > 0.05), so every matchup ran all 100
episodes and no adaptive decision ever fired — which the exact 992×100 total
betrays. Uniform n is what the downstream per-episode models assume, so the
data are fine; the *methods description* was wrong. See §18.1.]

---

## §1 — Registered test: β_RA

Per-cell mean seeker WR vs the 32-policy reference pool (self-pair excluded):

|  | A = 0.0 | A = 0.5 | Δ_A |
|---|---:|---:|---:|
| R4_sparse | 0.259 | 0.253 | -0.006 |
| R7_kitchen_sink | 0.620 | 0.758 | +0.138 |

Raw-scale interaction Δ-of-Δ = **+14.4pp WR** [Correction 2026-07: original
said "+0.144 pp"; 0.144 in WR units is 14.4 percentage points].

**Logistic mixed-effects model** [Correction 2026-07: the heteroscedastic
σ_seeker[reward] term was *not* authorized by pre-reg v2 — §A5 says
"unchanged from v1 §4.3" (pooled σ, per-seed intercepts) and §A6's REFINE
branch adds seeds, not model terms. It is a post-hoc, data-motivated model
change made without the v3 amendment the v2 Lock required, and σ_s[R7]/σ_s[R4]
— finding 2 — is precisely the unregistered parameter. Reported as exploratory;
see §18.2]:

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
- R7 seekers: ±35pp around the cell mean WR [Correction 2026-07: linearized;
  at a 0.69 cell mean a ±1.75 log-odds band is asymmetric, roughly −41/+24pp.
  The log-odds σ values are the quotable numbers]

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

> **[2026-07 status: this section's trajectory-PCA plan was never validly
> executed.** The 2026-05-14 run used features that indexed agent slot 0 as
> the seeker, but the env randomizes roles per reset — ~50% of episodes
> measured the hider (§18.3). No conclusion was ever drawn from it, and the
> mechanism question was answered by Part II's ablations instead (§10–§12).
> The fixed analysis (role-correct gather, arena-scaled thresholds, outcome
> features excluded from the PCA, 4-anchor reference panel) is queued as a
> supplementary check.]

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
   NUTS. Beware. [Correction 2026-07: partly confounded — the VB run
   (`design_c_analyze.py`) also grouped random intercepts by *seed number*
   via `C(seeker_seed_str)`, pooling unrelated policies from different
   cells into one intercept, while the NUTS fit grouped per policy. The
   ~10× factor mixes "VB underestimates" with "differently-specified
   model". The Part II panel VB-vs-NUTS contrast (§9), where the specs
   match, is the clean evidence for this lesson.]

4. **Pre-registration earned its keep.** Without v1/v2 frozen in git
   history, the directional flip (substitution → complementarity) and
   the "elevate heteroscedasticity from a nuisance to a finding" move
   would both look like post-hoc rationalization. With frozen pre-regs,
   the paper can transparently say "registered confirmatory result =
   REFINE; here's the exploratory follow-up." [Correction 2026-07: its
   *disclosure* function worked; its *binding* function was repeatedly
   violated in Part I — the registered decision rule was never evaluated
   on the registered 16-policy pool (REFINE-1 was launched ~3.5h after
   the grid, before any gauntlet existed), both refine rounds deviated
   from §A6's prescription, REFINE-2 was R7-only and data-triggered, and
   the model spec changed post-data with no v3. The v3 power extension
   (§16) was the first arm run fully by its own rules. Full list: §18.2.]

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

---
---

# Part II — The failure-mode arc (2026-05-15 → 2026-06-12)

Part I ended with "why does R7 produce a 4× wider policy-quality
distribution than R4?" Part II answers it. The short version: **R7's
seed variance is not noise — it is two discrete, mechanistically distinct
failure modes (a learning basin and an over-specialization trap), each
caused by identifiable reward terms and each rescued by zoo training at a
different dose.** Two adversarial idea-critic reviews steered this arc;
both their major objections (single-anchor evals; missing factorial
cells) were addressed with new experiments.

All Part II training cells: PPO, four_corners, 5M steps, zoo interval
50 / max-size 50 / uniform, HSM = 1.15 except §14. New reward variants:
`R7_no_coverage` (drops `area_coverage_bonus 0.05`), `R7_no_urgency`
(drops `seeker_escalating_urgency`), `R7_no_cov_urg` (drops both).

---

## §9 — Eval correction: the 4-anchor panel

Every early Part II eval scored seekers against a single reference hider
(the R4/A=0 anchor). The pre-reg (§4.2) specifies a **4-anchor reference
set** (R4/R7 × A=0/A=0.5, seed 42). The single-anchor shortcut turned out
to matter: several "collapsed" verdicts were anchor-specific. The most
important reversal: the R7/A=0.5 **seed-5 run, declared a failure under
the single anchor, scores 0.558 anchor-mean under the full panel** — it
had an anchor-specific weakness, not a broken policy. Individual
(seeker, anchor) win rates routinely span 0.0–1.0 within one seeker
(e.g. R7_no_urgency seed 0: 0.93/1.00/1.00 vs three anchors, 0.00 vs
the fourth).

The corrected panel (`design_c_anchor_panel_eval.py`) scores every run
× (4 anchors + own final hider) × 30 episodes. All Part II numbers
below use it. Episode-level Bernoulli outcomes feed the GLMMs
(`design_c_glmm_fit.py`, BinomialBayesMixedGLM, random intercepts per
run and per anchor).

**Updated registered estimand (Model A, R4 vs R7 × A), MCMC
(`design_c_panel_mcmc.py`, PyMC NUTS, R̂ ≤ 1.005, ESS > 2200):**

| parameter | NUTS median | 95% CrI (NUTS) | VB (for contrast) |
|---|---:|---|---|
| β_R (X) | +2.07 | [+0.42, +3.59] | +1.93 |
| β_A | -0.06 | [-1.16, +1.12] | -0.04 |
| **β_RA** | **+1.70** | **[-0.46, +3.88]** | +1.55 [+1.37, +1.73] |
| σ_run[R4] | 0.92 | [0.47, 1.49] | — |
| σ_run[R7] | 2.30 | [1.46, 3.26] | — |

P(β_RA > 0) = 0.94. **Verdict: REFINE** — large, directionally stable,
not certified. This is remarkably consistent with Part I's independent
gauntlet-based NUTS fit (+1.39, CrI [-0.13, +2.94], P = 0.963): two
eval populations, same answer. Two more replications fall out: β_A ≈ 0
(zoo does nothing for R4_sparse) and the heteroscedasticity
(σ_run[R7]/σ_run[R4] = 2.5× on the panel, 4× on the gauntlet).

The VB-vs-NUTS contrast is itself a finding, now demonstrated twice in
this project: the VB CrI was ~12× too narrow and flipped the verdict
(PURSUE → REFINE). **All VB intervals in this document are
untrustworthy; only NUTS intervals support verdicts.**

---

## §10 — The failure taxonomy: basin vs over-specialization

Each run is classified from (anchor-mean WR, own-hider WR):

- **basin** — anchor-mean < 0.5 AND own < 0.5: never learned pursuit.
- **over-specialization** — anchor-mean < 0.5, own ≥ 0.8: beats its own
  co-evolved hider (often 100%), loses broadly.
- **healthy** — anchor-mean ≥ 0.5.

Two supporting analyses sharpened the taxonomy:

1. **Forgetting is not a mechanism.** The running-max Forgetting Regret
   metric, re-tested against a per-column row-shuffle permutation null,
   shows **0/29 runs with regret above the null** — checkpoint
   "forgetting" patterns are indistinguishable from reshuffled noise.
2. **Basin runs never learn; over-specialized runs learn the wrong
   thing.** The seed-level checkpoint diagnostic showed an
   over-specialized run acquiring real pursuit skill by update ~300 and
   then narrowing onto its own hider, while basin runs plateau early.

**Classification counts by cell (HSM = 1.15):**

| cell | n | healthy | over-spec | basin | mixed |
|---|---:|---:|---:|---:|---:|
| R4_sparse, A=0 | 20 | 1 | 3 | 14 | 2 |
| R4_sparse, A=0.5 | 20 | 1 | 3 | 14 | 2 |
| R7_kitchen_sink, A=0 | 20 | 12 | 6 | 2 | — |
| R7_kitchen_sink, A=0.05 | 10 | 5 | 3 | 2 | — |
| R7_kitchen_sink, A=0.10 | 10 | 8 | 2 | 0 | — |
| R7_kitchen_sink, A=0.25 | 10 | 7 | 3 | 0 | — |
| R7_kitchen_sink, A=0.5 | 20 | 18 | 2 | 0 | — |
| R7_no_coverage, A=0 | 10 | 10 | 0 | 0 | — |
| R7_no_coverage, A=0.5 | 5 | 5 | 0 | 0 | — |
| R7_no_urgency, A=0 | 10 | 7 | 3 | 0 | — |
| R7_no_cov_urg, A=0 | 10 | 9 | 1 | 0 | — |

(Prereg cells at n=20 after the v3 power extension; the R4 cells'
label counts are coincidentally identical at A=0 and A=0.5 — different
seeds, same composition, which is itself the cleanest statement of
"A does nothing for R4".)

Key reading: **R4's failures are basins** (it mostly never learns, and
A does not rescue it — β_A ≈ 0). **R7's failures are mostly
over-specialization.** The earlier "R7/A=0 bad basin" language is
retired: only 2 of 20 R7 kitchen-sink A=0 runs are true basins.

---

## §11 — A is a dose, not a switch

Per-cell anchor-mean WR for R7_kitchen_sink across the A sweep
(A=0 and A=0.5 at n=20 post-v3; interior points n=10):

| A | 0 | 0.05 | 0.10 | 0.25 | 0.5 |
|---|---:|---:|---:|---:|---:|
| mean WR | 0.597 | 0.527 | 0.671 | 0.750 | 0.814 |
| sd | 0.237 | 0.310 | 0.208 | 0.275 | 0.214 |

The earlier "step function at A=0.5" claim is **retired**; so is the
transient "A=0.10 trough" (an n=5 artifact that vanished at n=10 — the
sd column explains why: per-cell means move ±0.1 on reseeding). The
robust pattern is in the failure-mode counts (§10 table), not the means:
**basins disappear by A ≥ 0.10; over-specialization shrinks with A but
has a long tail — 6/20 at A=0, 2/20 even at A=0.5.** (The pre-v3 n=9
cell showed 0/9 at A=0.5; n=20 revises "clears at 0.5" to "~3× rarer
at 0.5".) Mechanistically sensible: a small zoo dose breaks the
never-learn equilibrium, but un-narrowing a policy that already
exploits one hider requires sustained exposure to diverse opponents —
and even A=0.5 doesn't fully guarantee it.

Two findings on *how* A works:

- **It is the seeker's training exposure, not the final hider.**
  **[RETRACTED pending re-measurement, 2026-07:** this rested on hider
  behavioral dispersion from `design_c_hider_panel_eval.py`, which indexed
  agent slot 1 as the hider while the env randomizes roles per reset —
  ~50% of episodes measured the seeker. Mixing both agents into every
  fingerprint mechanically pushes dispersion toward "similar across A",
  i.e. the bug biases *toward* this claim. Re-run with role-correct
  features queued; until then the claim has no valid evidence. §18.3]
- **No evidence that A's benefit is coverage-specific.** Model B
  (no_coverage vs kitchen_sink × A), NUTS: the A main effect for
  R7_no_coverage is +1.25 [-0.59, +3.13] (directional, P ≈ 0.9) and the
  coverage-specific extra benefit is β_XA = +0.39 [-2.10, +2.91] —
  indistinguishable from zero. The earlier raw-DiD proxy (+1.48) and the
  VB interval (+0.34 [+0.16, +0.52]) both overstated certainty; at the
  run level the coverage-specific component is simply not measurable
  with 5–10 runs per cell. The defensible statement is descriptive:
  A = 0.5 cells are healthier than A = 0 cells in *both* reward
  variants (§10 counts).

---

## §12 — Coverage × urgency factorial (complete 2×2 at A=0)

Anchor-mean WR per cell (n = 9–10 each):

| | urgency present | urgency removed |
|---|---:|---:|
| **coverage present** | 0.659 (kitchen sink) | **0.597** (worst) |
| **coverage removed** | **0.823** (best) | 0.693 |

GLMM (Model C), NUTS with VB for contrast:

| parameter | NUTS median | 95% CrI (NUTS) | VB (overconfident) |
|---|---:|---|---|
| intercept (kitchen sink) | +1.05 | [-0.09, +2.17] | +1.05 |
| cov_rm | +1.11 | [-0.58, +2.73] | +1.04 [+0.94, +1.14] |
| urg_rm | -0.26 | [-1.91, +1.36] | -0.30 [-0.39, -0.20] |
| **cov_x_urg** | **-0.94** | **[-2.95, +1.13]** | -0.80 [-0.93, -0.67] |

P(cov_x_urg < 0) = 0.83. At the run level, **no individual factorial
coefficient is certified** — between-run σ in the R7-variant cells is
1.7–2.1 log-odds, so 10 runs/cell cannot pin down effects of this size.
The cell-mean pattern (coverage hurts, urgency corrals it, negative
interaction) is consistent in direction across NUTS, VB, and raw means,
but it is a *descriptive* finding pending more seeds.

What remains solid is the failure-mode contrast, which doesn't depend
on interval estimates: the no-coverage cells are **15/15 healthy**
across both A values, while every coverage-bearing R7 cell at low A
produces over-specializers (own-WR = 1.00, anchor-means as low as
0.12). Practical advice for this env is unchanged: **keep urgency, drop
coverage.** One incidental NUTS finding: σ_run[no_cov_urg] = 0.95 vs
1.7–2.1 for the other three cells — removing *both* shaped terms also
removes most of the between-seed variance, consistent with shaping
terms being the source of R7's seed lottery (Part I §2).

---

## §13 — Forgetting-regret and seed-5, resolved

Both loose threads from the early Part II narrative closed cleanly:
no genuine forgetting anywhere (shuffle null, §10), and the seed-5
"residual failure" was an artifact of single-anchor evaluation (§9).
The honest framing for the paper: single-opponent evals in self-play
are as untrustworthy as training-time WR (Part I, §7.1) — both
generalize Part I's lesson that **policy quality is only measurable
against a population.**

---

## §14 — HSM flank: the failure modes are not an HSM=1.15 artifact

Referee objection (via idea-critic): all of the above runs at
HSM = 1.15; maybe the collapse is a knife-edge property of that speed
ratio, not of the reward. Flank cells: R7_kitchen_sink, A=0, HSM ∈
{1.05, 1.20}, 8 seeds each (trained on DTU gbar). Because the prereg
anchors are HSM=1.15-native, scoring uses a **within-cell round-robin**
(`design_c_hsm_roundrobin.py`): each cell's 8 seekers vs its 8 hiders
at the cell's native HSM, 30 episodes/pair; read-out is cross-WR (mean
vs the 7 non-own hiders).

| HSM | cross-WR mean | sd | runs < 0.5 | over-spec flags | hider-dominated |
|---|---:|---:|---:|---:|---:|
| 1.05 | 0.736 | 0.220 | 1/8 | 0 | 1 |
| 1.15 | 0.629 | 0.213 | 3/8 | 2 | 0 |
| 1.20 | 0.565 | 0.216 | 5/8 | 3 | 2 |

Conclusion: **collapse exists at all three HSMs and scales monotonically
with hider speed — 1.15 is not special.** The coverage story survives;
it needs a severity-scales-with-HSM footnote rather than an
HSM-contingency caveat. The flank also surfaced a third, rarer outcome:
**hider-dominated co-evolution** (own-hider WR ≈ 0 while cross-WR is
mediocre — the *hider* won the arms race), appearing mainly at HSM=1.20
(2/8) where the speed advantage tips the equilibrium. Worth a sentence
in the paper, not a section.

---

## §15 — Claims ledger (authoritative)

**Confirmatory (pre-registered):**

| claim | status |
|---|---|
| β_RA > 0: shaping and zoo training are complements, not substitutes | **PURSUE (final, prereg v3 at n=20/cell, §16):** +2.05 [+0.72, +3.45], P(>0) = 0.9989. Trajectory of the estimate: +1.39 (gauntlet, n=5–9) → +1.70 (panel, n=5–9) → +2.05 (panel, n=20). NOTE the certified sign is *opposite* the originally registered substitution hypothesis. **2026-07 caveats (§18):** (a) survives the pair-RE overdispersion robustness fit (§18.4); (b) "reward" here changes BOTH roles' rewards — seeker-shaping vs opponent-quality attribution is being tested by the mixed-reward cells (§18.5); (c) β is a conditional log-odds coefficient, non-collapsibility-inflated relative to the marginal scale the §7 thresholds were calibrated on — marginal DiD reported alongside (§18.4). |
| β_A ≈ 0 under sparse reward | Replicated at n=20: -0.16 [-0.58, +0.25]. A does nothing for R4_sparse. |

The power extension that produced this (prereg v3, frozen 2026-06-12
before launch) added 52 runs to reach n=20 in all four prereg cells;
see §16 for the fit, hardware batch check, and robustness rerun.

**Exploratory (clearly labeled in the paper):**

| claim | evidence |
|---|---|
| R7 seed variance = two discrete failure modes, not noise | §10 taxonomy, 88 classified runs |
| R4 fails by basin; A does not rescue it | §10, β_A ≈ 0 (NUTS-certified ≈ 0) |
| R7 fails mostly by over-specialization | §10 (10 over-spec vs 3 basin across R7 cells at HSM 1.15) |
| A is a dose: small A kills basins, large A kills over-spec | §11 (failure-mode counts; descriptive) |
| A works through zoo-history exposure, not final-hider diversity | **RETRACTED pending re-measurement** — hider-panel features were role-scrambled (§18.3); the bug biased toward this claim |
| coverage bonus is an over-spec trap; no-coverage cells 15/15 healthy | §12 (failure-mode counts; descriptive but stark) |
| urgency corrals coverage / negative cov×urg interaction | §12 — **descriptive only**: NUTS CrI includes 0 (P(<0) = 0.83); direction consistent across all estimators |
| removing both shaped terms removes most between-seed variance (σ_run 0.95 vs 1.7–2.1) | §12, NUTS σ_run posteriors |
| failure modes persist across HSM 1.05–1.20, severity scales with hider speed | §14 |
| no checkpoint-level forgetting (shuffle null) | §13 |

**Retired claims** (and why):

- "A=0.5 acts as a step function" — dose-response with mode-specific
  thresholds (§11).
- "R7/A=0 sits in a bad basin" — mostly over-specialization (§10).
- "A only works by rescuing the coverage confound" — A helps
  no-coverage cells nearly as much (§11, Model B).
- "Urgency is strongly protective (−1.05)" — conditional: −0.30 with
  coverage present, −1.09 without (§12).
- Every single-anchor collapse verdict from early Part II — superseded
  by the 4-anchor panel (§9).

---

## §16 — Prereg v3 power extension: the registered test resolves to PURSUE

Prereg v3 (frozen 2026-06-12, before launch) added 52 runs to bring all
four prereg cells to n=20: R4 cells seeds 5–19, R7 cells seeds 9–19,
trained on DTU gbar, evaluated with the identical 4-anchor panel, fit
with the identical NUTS pipeline. Fixed-n stopping; no interim fits.

**Confirmatory result (Model A at n=20/cell, 320 binomial rows, 80 runs):**

| parameter | median | 95% CrI |
|---|---:|---|
| β_R | +1.57 | [+0.68, +2.45] |
| β_A | -0.16 | [-0.58, +0.25] |
| **β_RA** | **+2.05** | **[+0.72, +3.45]** |
| σ_run[R4] | 0.65 | [0.49, 0.83] |
| σ_run[R7] | 2.00 | [1.49, 2.54] |

P(β_RA > 0) = 0.9989. **Section-7 verdict: PURSUE** — |β_RA| ≥ log(1.25)
and the CrI excludes zero. Pre-specified checks, both clean:

- **Hardware batch:** β_gbar = +0.05 [-0.41, +0.52] — no detectable
  LUMI-vs-gbar batch effect; verdict unchanged with the term included.
- **Robustness:** 5 divergences at target_accept = 0.95; rerun at 0.99
  gives +1.99 [+0.68, +3.31], P = 0.9986 — PURSUE either way.

Final cell means (anchor-mean WR, n=20 each): R4 0.308/0.273 (A=0/0.5),
R7 0.597/0.814. The estimate's trajectory across the study — +1.39
(gauntlet) → +1.70 (panel) → +2.05 (panel, n=20) — never wavered in
sign; n=20 finally bought the precision to certify it.

**Honest framing for the paper:** the certified effect is *opposite in
sign* to the originally registered substitution hypothesis (prereg v1
expected β_RA < 0). The §7 rule is two-sided, so PURSUE is earned, but
the paper must say: we registered substitution, found complementarity,
and certified it only after a pre-registered power extension. The
heteroscedasticity finding (σ_run[R7]/σ_run[R4] ≈ 3.1×) replicates at
n=20 with tight intervals — Part I's headline exploratory claim is now
also effectively confirmed.

---

## §17 — Compute and infrastructure addendum

Part II training: ~81 additional runs (10 coverage-ablation + 5
no-coverage×A=0.5 + 10 urgency-ablation + 30 A-sweep + 10 urgency-only
+ 16 HSM-flank) plus the 52-run v3 power extension, ~190 run-hours by
Part I's accounting convention, plus ~17 hours of panel/round-robin
evals. Mid-arc, **LUMI's CPU allocation
exhausted** (104% used; submissions blocked) and was subsequently
**retired permanently** (2026-06-12). The final two training batches
(urgency-only, HSM flank) ran on **DTU gbar** under LSF using a
node-local /tmp pattern (env tarball + per-job unpack + trap-EXIT rsync
of results back to home). The full run archive (~2.2 GB) was migrated
off LUMI scratch to the local repo, with a second copy on gbar home.
Going forward: training on gbar, evals/analysis local. The MCMC
confirmation (§9, §11, §12) runs locally in ~20 s/model
(`design_c_panel_mcmc.py`, Binomial likelihood on per-(run, anchor)
counts) — it never needed a cluster.

---
---

# Part III — 2026-07 adversarial review: errata and follow-ups

A full adversarial review (three independent audit passes over the code,
the data provenance, and the pre-registration chain, 2026-07-09/10)
found the items below. Inline `[Correction 2026-07]` notes mark the
specific error sites in Part I/II; this section is the consolidated
record. Code fixes land in the same commit series that adds this section.

## §18.1 — The successive halving never happened

The gauntlet's Wilson-CI escalation thresholds were mathematically
unreachable (minimum achievable 95% half-width: 0.139 at n=10 vs
threshold 0.10; 0.057 at n=30 vs 0.05), and the coded criterion also
differed from pre-reg v1 §6's. Every matchup in both gauntlets ran
exactly 100 episodes — hence the exact 992×100 = 99,200 total. No bias
results (uniform n is what the unweighted per-episode GLMM/MCMC assume;
a *working* halving would have over-sampled ambiguous matchups and
biased them), but the methods text was wrong and the "concentrates
compute" claim false in practice. The dead machinery is removed
(`design_c_gauntlet.py` now runs a fixed 100 episodes/matchup).

## §18.2 — Pre-registration deviations, complete list (Part I)

1. The registered decision rule was never evaluated at its registered
   decision point: REFINE-1 (seeds 3–4) was committed ~3.5h after the
   grid launch, before any gauntlet existed; the first gauntlet ever run
   was the 24-policy one.
2. REFINE-1 ran 2 seeds/8 runs vs §A6's "3 more seeds (+6 runs)" (whose
   own arithmetic — 3 seeds × 4 cells = 12 runs — was already broken).
3. REFINE-2 was R7-only, explicitly data-triggered (targeting the arm
   with the interesting σ), and changed the reference-pool composition
   to 62.5% R7-trained without the v3 amendment the v2 Lock required.
4. The heteroscedastic σ_seeker[reward] model term was added post-data
   (also without v3); §1's citation of "§A6 REFINE branch" misattributed
   it. σ_s[R7]/σ_s[R4] is an unregistered, exploratory parameter.
5. Anchors were trained ~1h *after* the grid launch (v1 §4.2 required
   before, frozen); 4 anchors in a 32-policy pool cannot break the pool's
   self-reference anyway. (Part II's fixed 4-anchor panel is the repair.)
6. Only one of the "three pre-registered pilots" was registered in v1;
   the other two were added the same day, post-v1.
7. The registered REML arm became a GLM with one-way (seeker-only)
   cluster-robust SEs; the VB arm's random effects were grouped by seed
   number, pooling unrelated policies (see §7.3 correction).
8. v3 and all Part II work were authored outside git (see Provenance
   note); imported 2026-07-10.

Two mitigating replications: the v3 extension (§16) was fixed-n,
frozen-before-launch, with no interim fits; and the heteroscedasticity
pattern reproduced on new seeds (REFINE-2) and on the independent anchor
panel — it is real, but its *quantification* comes from an unregistered
model on a data-dependently expanded pool.

## §18.3 — Role-scrambled behavioral features (code bug)

`VecTagEnv` randomizes agent-slot/role assignment every reset;
`design_c_trajectory_analysis.py` and `design_c_hider_panel_eval.py`
indexed slots 0/1 as seeker/hider directly, so ~50% of episodes measured
the wrong agent. WR-type outputs were unaffected (`info['tagged']` is
role-independent — the anchor-panel GLMM/MCMC and all §9–§16 win-rate
results stand). Invalidated: the 2026-05-14 `behavior/` outputs (§6,
never used for claims) and §11's hider-dispersion claim (retracted
above; the scrambling biased toward it). Also fixed in the same commit:
center/wall thresholds calibrated to a 7.5 half-extent arena
(`arena_half` = 15), and `mean_tag_time` (outcome leakage) removed from
the behavior PCA. Both analyses re-run with fixes; results to be
appended.

## §18.4 — Statistical-model robustness (confirmatory PURSUE)

The §16 Model A treats each Binomial(n=30) (run, anchor) row as exact
given run+anchor intercepts, yet §9's own data show large run×anchor
interaction (one seeker spanning 0.00–1.00 across anchors) — unmodeled
overdispersion makes CrIs anti-conservative. `design_c_panel_mcmc.py`
now supports `--pair-re` (per-row random effect) and `--seed-re` (cells
are seed-paired: same seed = bit-identical init across cells), and
reports the marginal (population-averaged) interaction in WR units
alongside the conditional β_RA (which non-collapsibility inflates
relative to the marginal scale the §7 thresholds were written in).

**Robustness refit (2026-07-10, local, target_accept 0.95 and 0.99):
PURSUE survives.** Model A + pair RE + seed RE: β_RA = +2.40
[+0.77, +4.05], P(>0) = 0.998 (baseline reproduced first: +2.05
[+0.72, +3.45], P = 0.9989, matching §16 exactly). The CrI widens as
predicted but the lower bound stays above both 0 and log(1.25).
Marginal interaction: **+21pp WR, 95% CrI [+5.2, +36.9], P(>0) = 0.995**
— certified on the WR scale the thresholds were written in. Posterior
REs: σ_pair = 2.07 [1.86, 2.31] — the run×anchor interaction §9
observed anecdotally is as large as σ_run[R7] (2.18), vindicating the
robustness check; σ_seed ≈ 0.18 (seed pairing negligible);
σ_run[R4] collapses toward 0 once pair RE absorbs row noise (its
posterior funnels — R̂ ≈ 1.08 on that nuisance parameter only, 8
divergences at 0.99; β_RA itself mixes cleanly, R̂ = 1.000,
ESS ≈ 3500, and is stable across both target_accept settings).
Note σ_run[R7] ≫ σ_run[R4] persists under pair RE — the
heteroscedasticity replication is not an overdispersion artifact.
Outputs: `anchor_panel/panel_mcmc_A_pairre_seedre.txt`.

## §18.5 — The opponent-reward confound and the mixed-reward cells

The deepest unresolved issue: "reward shaping" as manipulated changes
the HIDER's reward too (R7 carries six hider-side terms; R4's hider is
terminal-only), and seeker and hider co-train in the same run. Every
R4-vs-R7 seeker contrast is therefore also an opponent-quality and
zoo-curriculum-quality contrast; both headline effects (β_RA > 0 and
the R7 σ inflation) admit an opponent-side explanation with no
seeker-shaping interaction at all. β_A ≈ 0 at R4 is equally consistent
with "a stale sparse-reward zoo hider is a useless curriculum".

Deconfounding experiment (launched with this arc, `--legacy-gae` for
comparability): **R7sk_R4hd** cells — seeker keeps R7's seeker-side
terms (pursuit distance, escalating urgency; coverage excluded per §12's
over-spec finding), hider gets exactly R4's terminal-only reward — at
A ∈ {0, 0.5} × 10 seeds. Comparison against the existing R4 cells
isolates seeker-side shaping at fixed opponent reward; comparison
against R7_kitchen_sink separates seeker-shaping from hider-shaping
contributions to the interaction and to σ.

## §18.6 — GAE done-mask off-by-one (training-code bug, all runs)

`train_zoo.py` cut GAE bootstrap/recursion by `dones[t+1]` instead of
`dones[t]`: terminal transitions bootstrapped γ·V(next episode's reset
state), the pre-terminal step was cut one step early, and the buffer
tail bootstrapped post-reset values. Shared by every run in this study
(both arms, both parts), so internal comparisons remain internally
consistent — but the bias scales with |V(reset)|, which is largest in
dense-reward runs, making it a candidate artifactual contributor to the
R7-arm variance findings. Fixed (correct masking is now the default;
`--legacy-gae` reproduces the old behavior and is recorded in run
metadata). Bounding experiment launched with this arc: R4/A=0 and
R7/A=0, 5 seeds each, fixed-GAE, evaluated on the same anchor panel —
if the R7 failure-mode mix shifts materially, the taxonomy needs a
GAE-sensitivity footnote and the σ claims a partial re-run.

## §18.7 — Smaller factual corrections

- "Zoo size 50" never binds: 5M steps ÷ 8,192 steps/update ≈ 610
  updates at interval 50 → ≤12 snapshots ever. A is uniform over ≤12
  checkpoints, several from very early training.
- "5M timesteps per run" counts both roles' rollouts; each policy
  trains on ~2.5M of its own steps.
- §10's taxonomy table has a "mixed" column that the three-way
  definition above it doesn't define (runs whose anchor-mean sits in
  0.4–0.6 with own-WR in between; borderline cases).
- Part I §3's SVD is grand-mean-centered only, so component 1 mixes
  seeker/hider main effects with "transitive skill", and a purely
  transitive game would still show σ²₂ > 0; the 61/16/11% shares are
  not a clean skill/cycle decomposition (double-centering check queued).
  The qualitative "non-transitive structure exists" reading survives (a
  calibrated transitive null predicts ~93% in component 1; observed 61%).
- §8's "Analysis (GLM + VB + MCMC ×2)" line and the "17–47 min"
  gauntlet times are unverifiable post-LUMI; treated as approximate.
