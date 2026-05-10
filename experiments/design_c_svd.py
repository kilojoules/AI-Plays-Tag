#!/usr/bin/env python3
"""
Design C SVD diagnostic — Czarnecki "Spinning Tops" framing.

Decomposes the seeker × hider WR matrix to determine whether the seed
variance is "draws on a single skill axis" (rank-1, transitive game) or
"non-transitive strategic diversity" (rank-2+, cyclic game).

Per the idea-critic's recommendation: this single analysis decides which
paper we're writing.

Outputs (under experiments/results/design_c/svd/):
    singular_values.png   — spectrum + cumulative explained variance
    embedding.png         — top-2 singular-vector projection of policies, colored by reward × A
    svd_summary.txt       — numerical interpretation
    matrix.csv            — the 24×24 WR matrix (with self-pair NaN)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
GAUNTLET_DIR = ROOT / "experiments" / "results" / "design_c" / "gauntlet"
OUT_DIR = ROOT / "experiments" / "results" / "design_c" / "svd"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    idx = pd.read_csv(GAUNTLET_DIR / "policy_index.csv")
    sm = pd.read_csv(GAUNTLET_DIR / "matchups_summary.csv")

    n = len(idx)
    print(f"Pool: {n} policies")

    # WR matrix M[i, j] = P(seeker i wins vs hider j). Diagonal = NaN (self-pair).
    M = np.full((n, n), np.nan)
    for _, r in sm.iterrows():
        if r.self_pair == 0:
            M[int(r.seeker_id), int(r.hider_id)] = float(r.wr)

    # For SVD we need a complete matrix. Replace diagonals with the mean of
    # row + col (a soft impute that does NOT bias rank inferences much for
    # a sparse missing pattern of n entries out of n^2).
    M_filled = M.copy()
    for i in range(n):
        if np.isnan(M_filled[i, i]):
            row_mean = np.nanmean(M[i, :])
            col_mean = np.nanmean(M[:, i])
            M_filled[i, i] = (row_mean + col_mean) / 2

    # Center on overall mean — Czarnecki's standard operation. The centered
    # matrix's singular spectrum is the "skill landscape" — top component is
    # transitive skill; later components are cyclic structure.
    grand = np.nanmean(M_filled)
    Mc = M_filled - grand

    U, S, Vt = np.linalg.svd(Mc, full_matrices=False)
    explained = S ** 2 / (S ** 2).sum()
    cum = np.cumsum(explained)

    # Save raw matrix
    pd.DataFrame(M, index=idx.id, columns=idx.id).to_csv(OUT_DIR / "matrix.csv")

    # Decide rank-1 vs rank-2+ — Czarnecki's gini-like test
    # Heuristic: if first SV explains >70% of variance AND second is <15%, rank-1 dominant.
    rank1_share = float(explained[0])
    rank2_share = float(explained[1])
    rank3_share = float(explained[2]) if n >= 3 else 0.0
    transitive_dominant = rank1_share > 0.70 and rank2_share < 0.15
    nontransitive = rank2_share > 0.10  # second component is meaningful

    # Project policies onto top components.
    # Seeker embedding = U @ diag(S);  hider embedding = V @ diag(S).
    seeker_emb = U * S  # shape (n, k)
    hider_emb = (Vt.T) * S
    # The "transitive skill" of seeker i = projection on first component.
    skill_seeker = seeker_emb[:, 0]
    # If skill_seeker is anti-correlated with WR-row-sum, flip sign for readability.
    row_means = np.nanmean(M_filled, axis=1)
    if np.corrcoef(skill_seeker, row_means)[0, 1] < 0:
        skill_seeker = -skill_seeker
        U[:, 0] = -U[:, 0]
        Vt[0, :] = -Vt[0, :]

    # Build the summary table
    info = idx.copy()
    info["row_mean_wr"] = row_means
    info["col_mean_wr_as_hider"] = np.nanmean(M_filled, axis=0)
    info["U1_seeker"] = U[:, 0]
    info["U2_seeker"] = U[:, 1]
    info["V1_hider"] = Vt[0, :]
    info["V2_hider"] = Vt[1, :]
    info.to_csv(OUT_DIR / "policy_embedding.csv", index=False)

    lines = []
    lines.append(f"=== Design C SVD diagnostic ===")
    lines.append(f"Pool size: {n}")
    lines.append(f"Grand mean WR: {grand:.4f}")
    lines.append("")
    lines.append("Singular value spectrum (centered matrix):")
    for k, (s, e, c) in enumerate(zip(S, explained, cum), start=1):
        lines.append(f"  σ_{k} = {s:.4f}   explained = {e:.4f}   cumulative = {c:.4f}")
    lines.append("")
    lines.append(f"Rank-1 share (top SV² / total): {rank1_share:.3f}")
    lines.append(f"Rank-2 share:                   {rank2_share:.3f}")
    lines.append(f"Rank-3 share:                   {rank3_share:.3f}")
    lines.append("")
    if transitive_dominant:
        lines.append("INTERPRETATION: TRANSITIVE skill landscape")
        lines.append("  First singular component dominates (>70%); second weak.")
        lines.append("  R7 seed variance = lucky vs unlucky draws on a single skill axis.")
        lines.append("  Story: 'rich reward enables wider exploration; some seeds find better policies'")
        lines.append("  Paper framing: REWARD-INDUCED EXPLORATION VARIANCE")
    elif nontransitive:
        lines.append("INTERPRETATION: NON-TRANSITIVE / CYCLIC structure present")
        lines.append("  Second (and possibly later) components carry meaningful variance.")
        lines.append("  R7 seeds may be reaching genuinely different strategies that beat each other rock-paper-scissors style.")
        lines.append("  Story: 'rich reward enables strategic diversity; cyclic dominance among seeds'")
        lines.append("  Paper framing: REWARD-INDUCED STRATEGIC DIVERSITY")
    else:
        lines.append("INTERPRETATION: ambiguous — first SV not dominant, second SV not meaningful")
        lines.append("  Probably a mix; needs more careful look at the embedding plot.")
    lines.append("")
    lines.append("Per-policy rank-1 skill scores (seeker side):")
    info_sorted = info.sort_values("U1_seeker", ascending=False)[
        ["id", "source", "reward", "A", "seed", "row_mean_wr", "U1_seeker", "U2_seeker"]]
    lines.append(info_sorted.round(3).to_string(index=False))

    out_txt = OUT_DIR / "svd_summary.txt"
    out_txt.write_text("\n".join(lines))
    print("\n".join(lines))

    # Plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Singular value spectrum
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        ks = np.arange(1, len(S) + 1)
        axes[0].plot(ks, explained, "o-")
        axes[0].set_xlabel("singular value index k")
        axes[0].set_ylabel("σ_k² / Σ σ²  (explained variance)")
        axes[0].set_title("Singular value spectrum")
        axes[0].grid(alpha=0.3)
        axes[0].axhline(0.10, color="red", linestyle=":", label="0.10 (rank-2 threshold)")
        axes[0].axhline(0.70, color="green", linestyle=":", label="0.70 (rank-1 dominance)")
        axes[0].legend(); axes[0].set_yscale("log")
        axes[1].plot(ks, cum, "o-")
        axes[1].set_xlabel("number of components")
        axes[1].set_ylabel("cumulative explained variance")
        axes[1].set_title("Cumulative explained variance")
        axes[1].axhline(0.90, color="grey", linestyle=":", label="0.90")
        axes[1].axhline(0.95, color="grey", linestyle="--", label="0.95")
        axes[1].grid(alpha=0.3); axes[1].legend()
        plt.tight_layout()
        plt.savefig(OUT_DIR / "singular_values.png", dpi=120); plt.close()

        # Top-2 embedding scatter, colored by (reward, A)
        fig, ax = plt.subplots(figsize=(8, 7))
        markers = {"R4_sparse": "o", "R7_kitchen_sink": "s"}
        colors = {0.0: "tab:blue", 0.5: "tab:orange"}
        for _, r in info.iterrows():
            ax.scatter(r["U1_seeker"], r["U2_seeker"],
                       marker=markers[r["reward"]], color=colors[r["A"]],
                       s=110, edgecolor="k",
                       alpha=0.85)
            ax.annotate(f"s{int(r['seed'])}", (r["U1_seeker"], r["U2_seeker"]),
                        fontsize=8, ha="center", va="center")
        ax.set_xlabel(f"U1 (transitive skill, {explained[0]*100:.1f}% var)")
        ax.set_ylabel(f"U2 (residual structure, {explained[1]*100:.1f}% var)")
        ax.set_title("Seeker embedding (top-2 SVD components)")
        ax.grid(alpha=0.3)
        ax.axhline(0, color="grey", linewidth=0.5)
        ax.axvline(0, color="grey", linewidth=0.5)
        from matplotlib.lines import Line2D
        legend = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:blue", markersize=10, label="R4 / A=0"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:orange", markersize=10, label="R4 / A=0.5"),
            Line2D([0], [0], marker="s", color="w", markerfacecolor="tab:blue", markersize=10, label="R7 / A=0"),
            Line2D([0], [0], marker="s", color="w", markerfacecolor="tab:orange", markersize=10, label="R7 / A=0.5"),
        ]
        ax.legend(handles=legend, loc="best")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "embedding.png", dpi=120); plt.close()
        print(f"\nPlots: {OUT_DIR}")
    except Exception as e:
        print(f"(plot skipped: {e})", file=sys.stderr)


if __name__ == "__main__":
    main()
