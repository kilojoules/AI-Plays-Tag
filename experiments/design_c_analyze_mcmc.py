#!/usr/bin/env python3
"""
Design C analysis — proper Bayesian arm via PyMC NUTS.

Replaces the variational-Bayes shortcut in design_c_analyze.py, which
was found to underestimate the posterior variance of β_RA. This is the
"Bayesian arm" the pre-reg actually called for; VB was a fallback.

Model (logistic mixed-effects on per-episode binary outcomes):
    y_e ~ Bernoulli(p_e)
    logit(p_e) = β_0 + β_R · R_seeker_e + β_A · A_seeker_e + β_RA · (R · A)_e
                  + α_seeker[id_seeker_e]
                  + α_hider[id_hider_e]
    α_seeker ~ Normal(0, σ_s)
    α_hider  ~ Normal(0, σ_h)
    σ_s, σ_h ~ HalfNormal(2)
    β_*      ~ Normal(0, 2.5)

Outputs (under experiments/results/design_c/analysis/):
    mcmc_summary.txt   — posterior summary, β_RA verdict, traces info
    mcmc_trace.png     — trace + density plots
    posterior_RA.png   — β_RA posterior with pre-reg thresholds overlaid

Usage:
    python experiments/design_c_analyze_mcmc.py
    python experiments/design_c_analyze_mcmc.py --draws 2000 --chains 4
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
GAUNTLET_DIR = ROOT / "experiments" / "results" / "design_c" / "gauntlet"
OUT_DIR = ROOT / "experiments" / "results" / "design_c" / "analysis"

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


def load_data(include_self: bool = False) -> pd.DataFrame:
    idx = pd.read_csv(GAUNTLET_DIR / "policy_index.csv")
    long = pd.read_csv(GAUNTLET_DIR / "matchups_long.csv")
    df = long.merge(idx.add_prefix("seeker_"), on="seeker_id") \
             .merge(idx.add_prefix("hider_"), on="hider_id")
    if not include_self:
        df = df[df.seeker_id != df.hider_id].copy()
    df["R"] = (df.seeker_reward == "R7_kitchen_sink").astype(int)
    df["A_bin"] = (df.seeker_A > 0).astype(int)
    df["RA"] = df.R * df.A_bin
    return df.reset_index(drop=True)


def fit_pymc(df: pd.DataFrame, draws: int = 2000, chains: int = 4, target_accept: float = 0.95):
    import pymc as pm
    import pytensor.tensor as pt

    seeker_codes, seeker_uniques = pd.factorize(df["seeker_id"])
    hider_codes, hider_uniques = pd.factorize(df["hider_id"])

    R = df["R"].to_numpy(np.int8)
    A = df["A_bin"].to_numpy(np.int8)
    RA = df["RA"].to_numpy(np.int8)
    y = df["won"].to_numpy(np.int8)

    coords = {
        "seeker": list(map(int, seeker_uniques)),
        "hider":  list(map(int, hider_uniques)),
        "obs": np.arange(len(df)),
    }

    with pm.Model(coords=coords) as model:
        beta_0 = pm.Normal("beta_0", 0, 2.5)
        beta_R = pm.Normal("beta_R", 0, 2.5)
        beta_A = pm.Normal("beta_A", 0, 2.5)
        beta_RA = pm.Normal("beta_RA", 0, 2.5)

        sigma_s = pm.HalfNormal("sigma_s", 2.0)
        sigma_h = pm.HalfNormal("sigma_h", 2.0)
        alpha_s = pm.Normal("alpha_s", 0, sigma_s, dims="seeker")
        alpha_h = pm.Normal("alpha_h", 0, sigma_h, dims="hider")

        eta = (beta_0 + beta_R * R + beta_A * A + beta_RA * RA
               + alpha_s[seeker_codes] + alpha_h[hider_codes])
        pm.Bernoulli("y_obs", logit_p=eta, observed=y, dims="obs")

        idata = pm.sample(
            draws=draws, tune=1000, chains=chains, cores=min(chains, 4),
            target_accept=target_accept,
            random_seed=42, progressbar=False,
        )
    return idata, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--target-accept", type=float, default=0.95)
    ap.add_argument("--include-self", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data(include_self=args.include_self)
    print(f"Loaded {len(df)} episodes  ({df.seeker_id.nunique()} seekers x {df.hider_id.nunique()} hiders)")

    print("\nFitting PyMC NUTS — this takes a few minutes...")
    idata, model = fit_pymc(df, draws=args.draws, chains=args.chains,
                            target_accept=args.target_accept)

    import arviz as az
    summary = az.summary(idata, var_names=["beta_0", "beta_R", "beta_A", "beta_RA",
                                            "sigma_s", "sigma_h"],
                         hdi_prob=0.95, round_to=4)
    print("\n=== Posterior summary (95% HDI) ===")
    print(summary.to_string())

    # Extract β_RA posterior
    post_RA = idata.posterior["beta_RA"].values.flatten()
    median = float(np.median(post_RA))
    lo, hi = np.quantile(post_RA, [0.025, 0.975])
    p_gt_0 = float((post_RA > 0).mean())
    p_gt_thresh = float((np.abs(post_RA) > THRESH_PURSUE).mean())

    lines = []
    lines.append("=== Design C MCMC analysis ===")
    lines.append(f"Episodes: {len(df)}, seekers: {df.seeker_id.nunique()}, hiders: {df.hider_id.nunique()}")
    lines.append(f"Sampler: PyMC NUTS  draws={args.draws}  chains={args.chains}  "
                 f"target_accept={args.target_accept}")
    lines.append("")
    lines.append("Posterior summary:")
    lines.append(summary.to_string())
    lines.append("")
    lines.append("--- β_RA (interaction) ---")
    lines.append(f"posterior median = {median:.4f}")
    lines.append(f"95% CrI = [{lo:.4f}, {hi:.4f}]")
    lines.append(f"P(β_RA > 0) = {p_gt_0:.4f}")
    lines.append(f"P(|β_RA| > log(1.25) = {THRESH_PURSUE:.3f}) = {p_gt_thresh:.4f}")
    lines.append("")
    lines.append(f"VERDICT (Bayes/MCMC): {verdict(median, lo, hi)}")
    lines.append("")
    lines.append(f"Pre-reg thresholds: PURSUE >= {THRESH_PURSUE:.3f}, KILL < {THRESH_KILL:.3f}")

    out_txt = OUT_DIR / "mcmc_summary.txt"
    out_txt.write_text("\n".join(lines))
    print("\n" + out_txt.read_text())

    # Plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ax = az.plot_trace(idata, var_names=["beta_R", "beta_A", "beta_RA",
                                              "sigma_s", "sigma_h"])
        plt.tight_layout()
        plt.savefig(OUT_DIR / "mcmc_trace.png", dpi=110)
        plt.close()

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(post_RA, bins=60, density=True, alpha=0.7, edgecolor="k", linewidth=0.4)
        ax.axvline(0, color="grey", linestyle=":", label="zero")
        ax.axvline(THRESH_PURSUE, color="green", linestyle="--", label=f"PURSUE  ({THRESH_PURSUE:.3f})")
        ax.axvline(-THRESH_PURSUE, color="green", linestyle="--")
        ax.axvline(THRESH_KILL, color="red", linestyle="--", label=f"KILL  ({THRESH_KILL:.3f})")
        ax.axvline(-THRESH_KILL, color="red", linestyle="--")
        ax.axvline(median, color="black", linestyle="-", label=f"median = {median:.3f}")
        ax.axvspan(lo, hi, color="grey", alpha=0.15, label=f"95% CrI [{lo:.2f}, {hi:.2f}]")
        ax.set_xlabel("β_RA  (log-odds interaction)")
        ax.set_ylabel("posterior density")
        ax.set_title(f"β_RA posterior — {verdict(median, lo, hi).split(' ')[0]}")
        ax.legend()
        plt.tight_layout()
        plt.savefig(OUT_DIR / "posterior_RA.png", dpi=120)
        plt.close()
        print(f"Plots: {OUT_DIR / 'mcmc_trace.png'}  {OUT_DIR / 'posterior_RA.png'}")
    except Exception as e:
        print(f"(plot skipped: {e})", file=sys.stderr)


if __name__ == "__main__":
    main()
