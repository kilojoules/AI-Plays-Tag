#!/usr/bin/env python3
"""
Design C analysis: fit logistic mixed-effects model on cross-eval outcomes,
report β_RA (reward × A interaction), apply pre-registered decision rule.

See pre-reg v2 §A5 + §A6.

Inputs (under experiments/results/design_c/gauntlet/):
  policy_index.csv    — id, source, reward, A, seed, ts_dir
  matchups_long.csv   — seeker_id, hider_id, episode_idx, won

Outputs (under experiments/results/design_c/analysis/):
  fit_summary.txt     — model coefficients, CIs, decision verdict
  cell_means.csv      — per-cell (reward × A) seeker WR vs pool
  interaction_plot.png — β_RA visual + 2x2 cell heatmap

Usage:
  python experiments/design_c_analyze.py
  python experiments/design_c_analyze.py --include-self  # keep diagonal entries
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
GAUNTLET_DIR = ROOT / "experiments" / "results" / "design_c" / "gauntlet"
OUT_DIR = ROOT / "experiments" / "results" / "design_c" / "analysis"

# Pre-registered thresholds (log-odds units).
THRESH_PURSUE = math.log(1.25)   # ≈ 0.2231
THRESH_KILL   = math.log(1.10)   # ≈ 0.0953


def verdict(beta_med: float, ci_lo: float, ci_hi: float) -> str:
    abs_b = abs(beta_med)
    excludes_zero = (ci_lo > 0) or (ci_hi < 0)
    if abs_b >= THRESH_PURSUE and excludes_zero:
        return f"PURSUE (|beta|={abs_b:.3f} >= {THRESH_PURSUE:.3f}, CI excludes 0)"
    if abs_b < THRESH_KILL and not excludes_zero:
        return f"KILL (|beta|={abs_b:.3f} < {THRESH_KILL:.3f}, CI includes 0)"
    return f"REFINE (|beta|={abs_b:.3f}, CI [{ci_lo:.3f}, {ci_hi:.3f}])"


def load_data(include_self: bool = False) -> pd.DataFrame:
    idx = pd.read_csv(GAUNTLET_DIR / "policy_index.csv")
    long = pd.read_csv(GAUNTLET_DIR / "matchups_long.csv")
    seek = idx.add_prefix("seeker_").rename(columns={"seeker_id": "seeker_id"})
    hide = idx.add_prefix("hider_").rename(columns={"hider_id": "hider_id"})
    df = long.merge(seek, on="seeker_id").merge(hide, on="hider_id")
    if not include_self:
        df = df[df.seeker_id != df.hider_id].copy()
    df["R"] = (df.seeker_reward == "R7_kitchen_sink").astype(int)
    df["A_bin"] = (df.seeker_A > 0).astype(int)
    df["RA"] = df.R * df.A_bin
    return df.reset_index(drop=True)


def fit_glm_clustered(df: pd.DataFrame) -> dict:
    """Logit GLM with cluster-robust SE on seeker_id (each seeker = one training run)."""
    import statsmodels.api as sm
    X = df[["R", "A_bin", "RA"]].astype(float)
    X = sm.add_constant(X)
    y = df["won"].astype(float)
    model = sm.GLM(y, X, family=sm.families.Binomial())
    res = model.fit(cov_type="cluster", cov_kwds={"groups": df["seeker_id"].values})
    coef = res.params["RA"]
    se = res.bse["RA"]
    z = 1.959963984540054
    return dict(
        beta=coef, se=se,
        ci_lo=coef - z * se, ci_hi=coef + z * se,
        pvalue=res.pvalues["RA"],
        full=res.summary().as_text(),
    )


def fit_bayes_mixed(df: pd.DataFrame) -> dict | None:
    """Variational Bayes binomial mixed model with random intercept on seeker_seed
       and on opponent_id (the hider being faced)."""
    try:
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    except Exception as e:
        return None
    df = df.copy()
    df["seeker_seed_str"] = df["seeker_seed"].astype(str)
    df["opp_id_str"] = df["hider_id"].astype(str)
    formula = "won ~ R + A_bin + RA"
    vc_formula = {"seed_grp": "0 + C(seeker_seed_str)", "opp_grp": "0 + C(opp_id_str)"}
    try:
        m = BinomialBayesMixedGLM.from_formula(formula, vc_formula, df)
        r = m.fit_vb()
    except Exception as e:
        return dict(error=str(e))
    # Find RA coefficient
    names = list(r.model.exog_names)
    if "RA" not in names:
        return dict(error="no RA term found")
    i = names.index("RA")
    mean = float(r.params[i])
    sd = float(r.cov_params()[i, i] ** 0.5) if hasattr(r, "cov_params") else float(r.fe_sd[i])
    z = 1.959963984540054
    return dict(beta=mean, sd=sd, ci_lo=mean - z * sd, ci_hi=mean + z * sd,
                method="VB BinomialBayesMixedGLM")


def cell_means(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["seeker_reward", "seeker_A"]).agg(
        n=("won", "count"), wr=("won", "mean"))
    z = 1.959963984540054
    g["se"] = np.sqrt(g["wr"] * (1 - g["wr"]) / g["n"])
    g["ci_lo"] = g["wr"] - z * g["se"]
    g["ci_hi"] = g["wr"] + z * g["se"]
    return g.reset_index()


def make_plot(cells: pd.DataFrame, glm_fit: dict, bayes_fit: dict | None, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Heatmap of cell means
    pivot = cells.pivot(index="seeker_reward", columns="seeker_A", values="wr")
    ax = axes[0]
    im = ax.imshow(pivot.values, vmin=0, vmax=1, cmap="RdYlGn")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"A={a}" for a in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.2f}", ha="center", va="center")
    ax.set_title("Seeker WR vs pool (per cell)")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Interaction line plot
    ax = axes[1]
    rewards = sorted(cells.seeker_reward.unique())
    a_vals = sorted(cells.seeker_A.unique())
    for r in rewards:
        sub = cells[cells.seeker_reward == r].sort_values("seeker_A")
        ax.errorbar(sub.seeker_A, sub.wr, yerr=sub.se, marker="o", capsize=4, label=r)
    ax.set_xlabel("A (zoo intensity)")
    ax.set_ylabel("Seeker WR vs pool")
    ax.set_title(f"Interaction: β_RA = {glm_fit['beta']:.3f}  CI [{glm_fit['ci_lo']:.3f}, {glm_fit['ci_hi']:.3f}]")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xticks(a_vals)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-self", action="store_true",
                    help="Include self-pair diagonal (default: drop, per pre-reg v1 §4.4)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data(include_self=args.include_self)
    n_self = (df.seeker_id == df.hider_id).sum()
    print(f"Loaded {len(df)} episodes  ({n_self} self-pair episodes {'kept' if args.include_self else 'dropped'})")
    print(f"Unique seekers: {df.seeker_id.nunique()}  unique hiders: {df.hider_id.nunique()}")

    cells = cell_means(df)
    cells.to_csv(OUT_DIR / "cell_means.csv", index=False)

    glm = fit_glm_clustered(df)
    bayes = fit_bayes_mixed(df)

    lines = []
    lines.append("=== Design C analysis ===")
    lines.append(f"Episodes: {len(df)}  (self-pairs {'kept' if args.include_self else 'excluded'})")
    lines.append(f"Pool size: {df.seeker_id.nunique()} seekers x {df.hider_id.nunique()} hiders")
    lines.append("")
    lines.append("Cell means (seeker WR vs pool):")
    lines.append(cells.to_string(index=False))
    lines.append("")
    lines.append("--- GLM (cluster-robust SE on seeker_id) ---")
    lines.append(f"β_RA = {glm['beta']:.4f}  SE = {glm['se']:.4f}  "
                 f"CI95 = [{glm['ci_lo']:.4f}, {glm['ci_hi']:.4f}]  p = {glm['pvalue']:.4g}")
    lines.append(f"VERDICT (GLM): {verdict(glm['beta'], glm['ci_lo'], glm['ci_hi'])}")
    lines.append("")
    if bayes is None:
        lines.append("--- Bayesian mixed model: skipped (BinomialBayesMixedGLM unavailable) ---")
    elif "error" in bayes:
        lines.append(f"--- Bayesian mixed model: failed ({bayes['error']}) ---")
    else:
        lines.append(f"--- Bayesian mixed model ({bayes['method']}) ---")
        lines.append(f"β_RA = {bayes['beta']:.4f}  SD = {bayes['sd']:.4f}  "
                     f"CrI95 = [{bayes['ci_lo']:.4f}, {bayes['ci_hi']:.4f}]")
        lines.append(f"VERDICT (Bayes): {verdict(bayes['beta'], bayes['ci_lo'], bayes['ci_hi'])}")
    lines.append("")
    lines.append("Pre-reg thresholds (log-odds): PURSUE >= 0.223, KILL < 0.095")
    lines.append("")
    lines.append("=== GLM full summary ===")
    lines.append(glm["full"])

    out_txt = OUT_DIR / "fit_summary.txt"
    out_txt.write_text("\n".join(lines))
    print(out_txt.read_text())

    try:
        make_plot(cells, glm, bayes, OUT_DIR / "interaction_plot.png")
        print(f"\nPlot: {OUT_DIR / 'interaction_plot.png'}")
    except Exception as e:
        print(f"(plot skipped: {e})", file=sys.stderr)


if __name__ == "__main__":
    main()
