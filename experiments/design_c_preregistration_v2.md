# Design C — Pre-Registration v2 (Amendment)

**Version:** v2 (2026-05-08)
**Supersedes:** v1 (2026-05-07, `experiments/design_c_preregistration.md`)
**Status:** ACTIVE — main grid launch authorised under this version

v1 remains in git history (commit `9050e58`). This document records the
amendment between the pilot result and the main-grid launch, as required
by v1 §10. Anything not amended here inherits from v1 unchanged.

---

## What changed and why

Three pre-registered pilots ran on 2026-05-07/08:

| pilot | algo | reward | A | result |
|---|---|---|---|---|
| 18497435 | SAC | R4_sparse | 0.0 | peak 55% @ 1–3M, **collapse to 14%** by 10M, alpha → 0.001, actor loss diverged |
| 18498830 | SAC | R7_kitchen_sink | 0.0 | peak 29% @ 3M, **collapse to 8.7%** by 10M, alpha → 0.001, actor loss diverged |
| 18498831 | PPO | R4_sparse | 0.0 | 31% @ 1M → 69% @ 5M, **stable at 50–65%** through 10M |

Per v1 §5 HP-BROKEN clause (`WR < 0.30 at 10M OR loss/alpha unstable`),
**both SAC pilots tripped the gate** — including the R7 dense-shaping cell
that was hypothesised to rescue SAC. Default-HP SAC self-play collapses
in this environment regardless of reward density. PPO learned the hardest
cell cleanly.

v1 §5 prescribes for HP-BROKEN: "Either tune SAC (separate effort) or
drop SAC from Design C and document." We are choosing the latter.

## Decisions (this amendment)

### A1. Drop SAC from Design C.
The grid is now PPO-only. The algorithm-class asymmetry leg of the
question is descoped from this study. Tuning SAC for self-play in this
env is its own project and we do not pursue it here. The two SAC pilots
+ this amendment **are** our public statement on SAC: a transparent null,
not a hidden one.

### A2. Sharpened question (PPO-only).
> "In PPO self-play, does zoo training (A) substitute for reward shaping
> in this 2D pursuit-evasion environment?"

Specifically: is there a non-zero `reward × A` interaction effect on
cross-pool seeker win rate, when reward varies between R4_sparse and
R7_kitchen_sink and A varies between 0.0 and 0.5.

The algorithm-asymmetry hypothesis is removed. We retain the substitution
hypothesis, which was the primary scientific claim.

### A3. Grid (revised from v1 §3).
2 reward × 2 A × 3 seeds × **PPO only** = **12 main-grid runs**.
All other v1 §3 settings unchanged (HSM=1.15, four_corners, 5M steps,
zoo size 50, sampling = uniform).

**Zoo-asymmetry disclosure** (was implicit in v1, now made explicit):
`trainer/train_zoo.py` does NOT pass `--use-seeker-zoo`, matching the
original A-sweep methodology (`experiments/zoo_asweep_tasks.py`). The
parameter A controls **hider zoo intensity only**; the seeker always
trains against the live hider. Any "A" effect we report is a
hider-side-only zoo effect. (SAC's `train_zoo_sac.py` zoos both sides
by default, which is part of why the SAC vs PPO comparison would have
been confounded even if SAC had not collapsed.)

### A4. Anchors and reference pool (revised from v1 §4.2).
Previous anchor list spanned PPO and SAC. Revised anchors are PPO-only,
4 fixed corners trained at seed=42 (a seed not used in the main grid):

- `R4 PPO A=0 seed=42`
- `R4 PPO A=0.5 seed=42`
- `R7 PPO A=0 seed=42`
- `R7 PPO A=0.5 seed=42`

Reference pool = {12 main-grid finals} ∪ {4 anchors} = 16 seekers and
16 hiders. Self-pair exclusion as in v1 §4.4.

### A5. Statistical model (unchanged from v1 §4.3).
A single algorithm now means we drop the per-algorithm split. One model
fit, total. Estimand `β_RA` (the reward × A interaction coefficient) is
the single primary statistic.

### A6. Decision rule (unchanged thresholds, single arm).
With one algorithm, the per-algo "OR" rule from v1 §7 reduces to a
single test on `β_RA`:

| outcome | rule | next step |
|---|---|---|
| **PURSUE Design A** | `|β_RA|` posterior median ≥ log(1.25) ≈ 0.22 AND 95% CrI excludes 0 | proceed to Design A |
| **REFINE** | `|β_RA|` between log(1.10) and log(1.25), OR borderline CrI | add 3 more seeds (+6 runs, ~6 GPU-hr) |
| **KILL** | `|β_RA|` < log(1.10) ≈ 0.10 AND 95% CrI includes 0 | document and stop |

### A7. Compute budget (revised).
| component | runs | wall-each | subtotal |
|---|---|---|---|
| Pilots (already spent) | 3 | ~2h | ~6 CPU-hr (sunk) |
| Anchors (PPO, 5M, seed=42) | 4 | ~1h | ~4 CPU-hr |
| Main grid (PPO, 5M) | 12 | ~1h | ~12 CPU-hr |
| Gauntlet (3-stage, 16×16 + anchors) | — | — | ~1 CPU-hr |
| **Total going forward** | 16 | | **~17 CPU-hr** |

Substantially smaller than v1 §8 (78 GPU-hr) because (a) SAC dropped,
(b) PPO is faster per run on this env, (c) running on `small` (CPU)
rather than `small-g` (GPU we couldn't use anyway).

If REFINE: +6 runs ≈ +6 CPU-hr.

### A8. What we publish regardless of outcome (v1 §9 + amendment).
On top of v1 §9, we publish:
- The two SAC pilots and this amendment, as a methodology note showing
  default-HP SAC is unstable in this self-play env regardless of reward
  density.
- The PPO pilot as evidence that the task and reward range are tractable,
  ruling out "task too hard" as a confound for any null result.

A null result on the PPO `β_RA` is still publishable as: "in PPO
self-play, hider zoo training and reward shaping do NOT interact (in
this env, between R4 and R7, between A=0 and A=0.5)."

## Lock

Once the main-grid sbatch is submitted, no further changes to the model
spec, decision rule, or reference-pool composition without a v3 of this
document committed before viewing the grid data.
