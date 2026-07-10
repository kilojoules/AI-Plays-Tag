#!/usr/bin/env python3
"""
MCMC confirmation of the anchor-panel GLMMs (PyMC NUTS).

design_c_glmm_fit.py fits Models A/B/C with statsmodels variational Bayes,
which this project has measured to understate posterior sd (~10x on the
Part I gauntlet model). This script refits the same three models with NUTS
so the section-7 verdicts rest on trustworthy intervals. Codings match
design_c_glmm_fit.py exactly, so coefficients are directly comparable.

Likelihood is Binomial on the per-(run, anchor) win counts — identical to
the episode-level Bernoulli expansion, ~30x fewer rows.

Models (logit scale, random intercepts u_run + v_anchor):
  A (CONFIRMATORY) X = 1[R7_kitchen_sink] vs R4_sparse, A in {0, 0.5}
  B (exploratory)  X = 1[R7_kitchen_sink] vs R7_no_coverage, A in {0, 0.5}
  C (exploratory)  A=0 factorial: cov_rm + urg_rm + cov_x_urg
sigma_run is stratified by reward cell (Part I lesson: between-seed
variance differs by reward; pooling washes it out).

Outputs (under experiments/results/design_c/anchor_panel/):
  panel_mcmc_<model>.txt, panel_mcmc_<model>_posterior.png

Usage:
  python experiments/design_c_panel_mcmc.py            # all three models
  python experiments/design_c_panel_mcmc.py --model A
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
PANEL = ROOT / "experiments" / "results" / "design_c" / "anchor_panel" / "anchor_panel.csv"
OUT_DIR = PANEL.parent

THRESH_PURSUE = math.log(1.25)
THRESH_KILL = math.log(1.10)


def verdict(median: float, lo: float, hi: float) -> str:
    abs_b = abs(median)
    excludes_zero = (lo > 0) or (hi < 0)
    if abs_b >= THRESH_PURSUE and excludes_zero:
        return f"PURSUE (|beta|={abs_b:.3f} >= {THRESH_PURSUE:.3f}, CrI excludes 0)"
    if abs_b < THRESH_KILL and not excludes_zero:
        return f"KILL (|beta|={abs_b:.3f} < {THRESH_KILL:.3f}, CrI includes 0)"
    return f"REFINE (|beta|={abs_b:.3f}, CrI [{lo:.3f}, {hi:.3f}])"


def load_panel(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["run_id"] = (df.reward + "_A" + (df.A * 100).astype(int).astype(str)
                    + "_s" + df.seed.astype(str))
    return df[df.opponent != "own"].copy()


def select_model(anch: pd.DataFrame, which: str):
    """Return (data, fixed-effect design dict, key coefficient name)."""
    if which == "A":
        d = anch[anch.reward.isin(["R4_sparse", "R7_kitchen_sink"])
                 & anch.A.isin([0.0, 0.5])].copy()
        d["X"] = (d.reward == "R7_kitchen_sink").astype(int)
        d["A01"] = (d.A == 0.5).astype(int)
        d["XA"] = d.X * d.A01
        # prereg v3 hardware batch: seeds 5+ (R4) / 9+ (R7) trained on gbar
        d["gbar"] = (((d.reward == "R4_sparse") & (d.seed >= 5))
                     | ((d.reward == "R7_kitchen_sink") & (d.seed >= 9))
                     ).astype(int)
        return d, ["X", "A01", "XA"], "XA", \
            "Model A (CONFIRMATORY, prereg 4.3): R4 vs R7 x A"
    if which == "B":
        d = anch[anch.reward.isin(["R7_no_coverage", "R7_kitchen_sink"])
                 & anch.A.isin([0.0, 0.5])].copy()
        d["X"] = (d.reward == "R7_kitchen_sink").astype(int)
        d["A01"] = (d.A == 0.5).astype(int)
        d["XA"] = d.X * d.A01
        return d, ["X", "A01", "XA"], "XA", \
            "Model B (exploratory): R7_no_coverage vs R7_kitchen_sink x A"
    if which == "C":
        cells = ["R7_kitchen_sink", "R7_no_coverage", "R7_no_urgency",
                 "R7_no_cov_urg"]
        d = anch[anch.reward.isin(cells) & (anch.A == 0.0)].copy()
        d["cov_rm"] = d.reward.isin(["R7_no_coverage", "R7_no_cov_urg"]).astype(int)
        d["urg_rm"] = d.reward.isin(["R7_no_urgency", "R7_no_cov_urg"]).astype(int)
        d["cov_x_urg"] = d.cov_rm * d.urg_rm
        return d, ["cov_rm", "urg_rm", "cov_x_urg"], "cov_x_urg", \
            "Model C (exploratory factorial at A=0)"
    raise ValueError(which)


def fit(d: pd.DataFrame, fe_cols: list, draws: int, chains: int,
        target_accept: float):
    import pymc as pm

    run_codes, run_uniques = pd.factorize(d["run_id"])
    anch_codes, anch_uniques = pd.factorize(d["opponent"])
    # sigma_run stratified by reward cell
    cell_of_run = (d.drop_duplicates("run_id").set_index("run_id")
                   .loc[run_uniques, "reward"])
    cell_codes, cell_uniques = pd.factorize(cell_of_run)

    coords = {"run": list(run_uniques), "anchor": list(anch_uniques),
              "cell": list(cell_uniques), "obs": np.arange(len(d))}

    with pm.Model(coords=coords) as model:
        beta_0 = pm.Normal("beta_0", 0, 2.5)
        betas = {c: pm.Normal(f"beta_{c}", 0, 2.5) for c in fe_cols}

        sigma_run = pm.HalfNormal("sigma_run", 2.0, dims="cell")
        u_run = pm.Normal("u_run", 0, sigma_run[cell_codes], dims="run")
        sigma_anchor = pm.HalfNormal("sigma_anchor", 2.0)
        v_anchor = pm.Normal("v_anchor", 0, sigma_anchor, dims="anchor")

        eta = beta_0 + u_run[run_codes] + v_anchor[anch_codes]
        for c in fe_cols:
            eta = eta + betas[c] * d[c].to_numpy(np.int8)

        pm.Binomial("wins", n=d["episodes"].to_numpy(),
                    logit_p=eta, observed=d["wins"].to_numpy(), dims="obs")

        idata = pm.sample(draws=draws, tune=1000, chains=chains,
                          cores=min(chains, 4), target_accept=target_accept,
                          random_seed=42, progressbar=False)
    return idata


def report(idata, d, fe_cols, key, label, suffix, draws, chains):
    import arviz as az

    var_names = ["beta_0"] + [f"beta_{c}" for c in fe_cols] \
        + ["sigma_run", "sigma_anchor"]
    summary = az.summary(idata, var_names=var_names, hdi_prob=0.95, round_to=4)

    post = idata.posterior[f"beta_{key}"].values.flatten()
    median = float(np.median(post))
    lo, hi = np.quantile(post, [0.025, 0.975])
    p_gt_0 = float((post > 0).mean())

    lines = [f"=== {label} — PyMC NUTS ===",
             f"rows={len(d)}  runs={d.run_id.nunique()}  "
             f"anchors={d.opponent.nunique()}  draws={draws} x {chains} chains",
             "", summary.to_string(), "",
             f"--- beta_{key} (key coefficient) ---",
             f"posterior median = {median:+.4f}",
             f"95% CrI = [{lo:+.4f}, {hi:+.4f}]",
             f"P(beta > 0) = {p_gt_0:.4f}",
             "",
             f"VERDICT (section 7): {verdict(median, lo, hi)}"]
    out_txt = OUT_DIR / f"panel_mcmc_{suffix}.txt"
    out_txt.write_text("\n".join(lines))
    print("\n" + "\n".join(lines))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(post, bins=60, density=True, alpha=0.7, edgecolor="k",
                linewidth=0.4)
        ax.axvline(0, color="grey", linestyle=":")
        for t, c in [(THRESH_PURSUE, "green"), (THRESH_KILL, "red")]:
            ax.axvline(t, color=c, linestyle="--")
            ax.axvline(-t, color=c, linestyle="--")
        ax.axvline(median, color="black",
                   label=f"median = {median:+.3f}")
        ax.axvspan(lo, hi, color="grey", alpha=0.15,
                   label=f"95% CrI [{lo:+.2f}, {hi:+.2f}]")
        ax.set_xlabel(f"beta_{key} (log-odds)")
        ax.set_ylabel("posterior density")
        ax.set_title(f"{label}\n{verdict(median, lo, hi).split(' ')[0]}")
        ax.legend()
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"panel_mcmc_{suffix}_posterior.png", dpi=120)
        plt.close()
    except Exception as e:
        print(f"(plot skipped: {e})", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["A", "B", "C", "all"], default="all")
    ap.add_argument("--panel", type=Path, default=PANEL)
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--target-accept", type=float, default=0.95)
    ap.add_argument("--batch-check", action="store_true",
                    help="Model A only: add the prereg v3 trained_on_gbar "
                         "fixed effect (hardware batch sanity check).")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    anch = load_panel(args.panel)
    which = ["A", "B", "C"] if args.model == "all" else [args.model]
    for w in which:
        d, fe_cols, key, label = select_model(anch, w)
        suffix = w
        if args.batch_check and w == "A":
            fe_cols = fe_cols + ["gbar"]
            label += " + gbar batch check"
            suffix = "A_batchcheck"
        print(f"\n=== Fitting {label}: rows={len(d)} "
              f"runs={d.run_id.nunique()} ===")
        idata = fit(d, fe_cols, args.draws, args.chains, args.target_accept)
        report(idata, d, fe_cols, key, label, suffix, args.draws, args.chains)


if __name__ == "__main__":
    main()
