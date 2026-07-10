#!/usr/bin/env python3
"""
Prereg-specified GLMM on the anchor panel + threshold sensitivity +
basin-vs-overspecialization classification.

Addresses idea-critic concerns:
  (f) the beta_RA "proxy" was a DiD on cell means, not the prereg section
      4.3 logistic mixed-effects model — this script fits the actual model
      (episode-level Bernoulli, random intercepts per RUN and per ANCHOR)
      and applies the section-7 PURSUE/REFINE/KILL decision rule;
  (e) confirmatory vs exploratory labeling — model A below is the only
      confirmatory analysis (prereg cells); models B and C are labeled
      exploratory;
  (b) collapse-threshold sensitivity at {0.4, 0.5, 0.6};
  (d) generalizes the seed-5 diagnostic: every run classified by
      (anchor-mean WR, own-hider WR) into healthy / basin / over-spec.

Models (all: logit(p) = b0 + bX*X + bA*A + bXA*X*A + u_run + v_anchor):
  A (CONFIRMATORY) X = reward in {R4_sparse=0, R7_kitchen_sink=1}, A in {0, 0.5}
  B (exploratory)  X = reward in {R7_no_coverage=0, R7_kitchen_sink=1}, A in {0, 0.5}
  C (exploratory)  A=0 cells only; factorial coverage_removed x urgency_removed
                   over {R7_kitchen_sink, R7_no_coverage, R7_no_urgency,
                   R7_no_cov_urg} (falls back to additive if the
                   R7_no_urgency cell is not yet trained).

Fitting: statsmodels BinomialBayesMixedGLM (variational Bayes), as named in
the prereg. NOTE: VB tends to underestimate posterior sd — borderline
decisions should be confirmed with the existing MCMC pipeline.

Usage:
  python experiments/design_c_glmm_fit.py
  python experiments/design_c_glmm_fit.py --panel <path to anchor_panel.csv>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
PANEL = ROOT / "experiments" / "results" / "design_c" / "anchor_panel" / "anchor_panel.csv"
OUT_DIR = PANEL.parent

LOG_125 = float(np.log(1.25))  # prereg section-7 PURSUE threshold ~ 0.223
LOG_110 = float(np.log(1.10))  # prereg section-7 KILL threshold   ~ 0.095


def expand_bernoulli(df: pd.DataFrame) -> pd.DataFrame:
    """One Bernoulli row per episode from (wins, episodes) counts."""
    reps = df["episodes"].to_numpy()
    out = df.loc[df.index.repeat(reps)].copy()
    y = np.zeros(int(reps.sum()), dtype=np.int8)
    pos = 0
    for w, n in zip(df["wins"].to_numpy(), reps):
        y[pos:pos + int(w)] = 1
        pos += int(n)
    out["y"] = y
    return out.reset_index(drop=True)


def fit_interaction_glmm(data: pd.DataFrame, xcol: str, label: str):
    """Fit logit(y) ~ X + A01 + X:A01 + (1|run) + (1|anchor); report X:A01."""
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    d = data.copy()
    d["XA"] = d[xcol] * d["A01"]
    model = BinomialBayesMixedGLM.from_formula(
        f"y ~ {xcol} + A01 + XA",
        {"run": "0 + C(run_id)", "anch": "0 + C(opponent)"},
        d,
    )
    res = model.fit_vb()
    names = list(res.model.exog_names)
    i = names.index("XA")
    med, sd = float(res.fe_mean[i]), float(res.fe_sd[i])
    lo, hi = med - 1.96 * sd, med + 1.96 * sd
    excl0 = (lo > 0) or (hi < 0)

    if abs(med) >= LOG_125 and excl0:
        verdict = "PURSUE"
    elif abs(med) < LOG_110 and not excl0:
        verdict = "KILL"
    else:
        verdict = "REFINE"

    print(f"\n--- {label} ---")
    print(f"rows={len(d)}  runs={d.run_id.nunique()}  anchors={d.opponent.nunique()}")
    for nm in names:
        j = names.index(nm)
        print(f"  {nm:>12s}  {res.fe_mean[j]:+.3f} (sd {res.fe_sd[j]:.3f})")
    print(f"beta_XA = {med:+.3f}  95% CrI [{lo:+.3f}, {hi:+.3f}]  "
          f"thresholds: log(1.10)={LOG_110:.3f}, log(1.25)={LOG_125:.3f}")
    print(f"Section-7 verdict: {verdict}")
    return dict(label=label, beta=med, sd=sd, lo=lo, hi=hi, verdict=verdict)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", type=Path, default=PANEL)
    args = ap.parse_args()

    df = pd.read_csv(args.panel)
    df["run_id"] = (df.reward + "_A" + (df.A * 100).astype(int).astype(str)
                    + "_s" + df.seed.astype(str))
    anch = df[df.opponent != "own"].copy()

    # ---------------- GLMM models ----------------
    results = []

    # Model A — CONFIRMATORY (prereg cells only)
    a = anch[anch.reward.isin(["R4_sparse", "R7_kitchen_sink"])
             & anch.A.isin([0.0, 0.5])].copy()
    a["X"] = (a.reward == "R7_kitchen_sink").astype(int)
    a["A01"] = (a.A == 0.5).astype(int)
    results.append(fit_interaction_glmm(
        expand_bernoulli(a), "X",
        "Model A (CONFIRMATORY, prereg 4.3): R4 vs R7 x A — beta_RA"))

    # Model B — exploratory de-confounding contrast
    b = anch[anch.reward.isin(["R7_no_coverage", "R7_kitchen_sink"])
             & anch.A.isin([0.0, 0.5])].copy()
    b["X"] = (b.reward == "R7_kitchen_sink").astype(int)
    b["A01"] = (b.A == 0.5).astype(int)
    results.append(fit_interaction_glmm(
        expand_bernoulli(b), "X",
        "Model B (exploratory): R7_no_coverage vs R7_kitchen_sink x A"))

    # Model C — exploratory factorial at A=0
    cells = ["R7_kitchen_sink", "R7_no_coverage", "R7_no_urgency", "R7_no_cov_urg"]
    c = anch[anch.reward.isin(cells) & (anch.A == 0.0)].copy()
    have_no_urg = (c.reward == "R7_no_urgency").any()
    c["cov_rm"] = c.reward.isin(["R7_no_coverage", "R7_no_cov_urg"]).astype(int)
    c["urg_rm"] = c.reward.isin(["R7_no_urgency", "R7_no_cov_urg"]).astype(int)
    cd = expand_bernoulli(c)
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    if have_no_urg:
        cd["cov_x_urg"] = cd["cov_rm"] * cd["urg_rm"]
        formula = "y ~ cov_rm + urg_rm + cov_x_urg"
        note = ""
    else:
        formula = "y ~ cov_rm + urg_rm"
        note = " (R7_no_urgency cell missing - additive only, interaction unidentified)"
    model = BinomialBayesMixedGLM.from_formula(
        formula, {"run": "0 + C(run_id)", "anch": "0 + C(opponent)"}, cd)
    res = model.fit_vb()
    print(f"\n--- Model C (exploratory factorial at A=0){note} ---")
    print(f"rows={len(cd)}  runs={cd.run_id.nunique()}")
    for j, nm in enumerate(res.model.exog_names):
        lo = res.fe_mean[j] - 1.96 * res.fe_sd[j]
        hi = res.fe_mean[j] + 1.96 * res.fe_sd[j]
        print(f"  {nm:>12s}  {res.fe_mean[j]:+.3f} (sd {res.fe_sd[j]:.3f})  "
              f"CrI [{lo:+.3f}, {hi:+.3f}]")

    # ---------------- Threshold sensitivity ----------------
    per_run = anch.groupby(["reward", "A", "run_id"]).wr.mean().reset_index()
    print("\n--- Collapse-threshold sensitivity (anchor-mean WR per run) ---")
    print(f"{'reward':16s} {'A':>5s}  {'n':>3s}   {'<0.4':>4s}  {'<0.5':>4s}  {'<0.6':>4s}")
    for (reward, A), g in per_run.groupby(["reward", "A"]):
        print(f"{reward:16s} {A:>5}  {len(g):>3d}   "
              f"{int((g.wr < 0.4).sum()):>4d}  {int((g.wr < 0.5).sum()):>4d}  "
              f"{int((g.wr < 0.6).sum()):>4d}")

    # ---------------- Basin vs over-specialization ----------------
    own = df[df.opponent == "own"].set_index("run_id").wr
    cls_rows = []
    for run_id, g in anch.groupby("run_id"):
        a_wr = g.wr.mean()
        s_wr = float(own.get(run_id, np.nan))
        if a_wr >= 0.5:
            label = "healthy"
        elif s_wr >= 0.8:
            label = "over-specialization"
        elif s_wr < 0.5:
            label = "basin"
        else:
            label = "mixed"
        cls_rows.append(dict(run_id=run_id, reward=g.reward.iloc[0], A=g.A.iloc[0],
                             seed=g.seed.iloc[0], anchor_wr=a_wr, own_wr=s_wr,
                             label=label))
    cls = pd.DataFrame(cls_rows)
    cls.to_csv(OUT_DIR / "run_classification.csv", index=False)
    print("\n--- Failure-mode classification (anchor-mean < 0.5) ---")
    bad = cls[cls.label != "healthy"].sort_values(["reward", "A", "seed"])
    for _, r in bad.iterrows():
        print(f"  {r.reward:16s} A={r.A:<5} seed {int(r.seed)}  "
              f"anchor={r.anchor_wr:.2f}  own={r.own_wr:.2f}  -> {r.label}")
    print("\nCounts by label:")
    print(cls.groupby(["reward", "A", "label"]).size().to_string())

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        colors = {"healthy": "tab:green", "over-specialization": "tab:orange",
                  "basin": "tab:red", "mixed": "tab:purple"}
        for label, g in cls.groupby("label"):
            ax.scatter(g.own_wr, g.anchor_wr, c=colors[label], label=f"{label} (n={len(g)})",
                       s=60, edgecolor="k", alpha=0.8)
        ax.axhline(0.5, color="grey", lw=0.8, ls="--")
        ax.axvline(0.8, color="grey", lw=0.8, ls=":")
        ax.axvline(0.5, color="grey", lw=0.8, ls=":")
        ax.set_xlabel("WR vs OWN final hider (in-distribution)")
        ax.set_ylabel("mean WR vs 4 prereg anchors (out-of-distribution)")
        ax.set_title("Basin vs over-specialization, all runs")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "run_classification.png", dpi=130)
        plt.close()
        print(f"\nWrote {OUT_DIR / 'run_classification.png'}")
    except Exception as e:
        print(f"(plot skipped: {e})", file=sys.stderr)

    print("\nNOTE: VB posteriors understate sd; borderline section-7 verdicts "
          "should be confirmed with the MCMC pipeline before publication.")


if __name__ == "__main__":
    main()
