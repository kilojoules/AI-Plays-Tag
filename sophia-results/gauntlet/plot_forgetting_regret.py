#!/usr/bin/env python3
"""Compute Forgetting Regret (FR) from gauntlet matrices and plot as heatmap."""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA = Path(__file__).parent / "gauntlet_matrices.json"


def forgetting_regret(win_matrix: np.ndarray):
    """Compute FR_full and FR_final from a gauntlet win-rate matrix.

    FR measures how much the seeker has regressed from its own
    training-trajectory peak against each hider.

    Parameters
    ----------
    win_matrix : (n, n) array of win rates, W[i,j] = P(seeker_i beats hider_j)

    Returns
    -------
    fr_full : float   – mean pointwise regret over entire matrix
    fr_final : float  – mean regret of the *final* seeker checkpoint
    """
    n = win_matrix.shape[0]
    # Running max along rows (axis=0) — forward in training time
    running_max = np.maximum.accumulate(win_matrix, axis=0)
    pointwise_regret = running_max - win_matrix  # always >= 0
    fr_full = pointwise_regret.mean()
    fr_final = pointwise_regret[-1].mean()
    return fr_full, fr_final


def main():
    with open(DATA) as f:
        data = json.load(f)

    stps = sorted({v["stp"] for v in data.values()})
    hsms = sorted({v["hsm"] for v in data.values()})

    fr_full_grid = np.full((len(stps), len(hsms)), np.nan)
    fr_final_grid = np.full((len(stps), len(hsms)), np.nan)

    for exp_name, exp in data.items():
        stp_idx = stps.index(exp["stp"])
        hsm_idx = hsms.index(exp["hsm"])
        W = np.array(exp["win_matrix"])
        fr_full, fr_final = forgetting_regret(W)
        fr_full_grid[stp_idx, hsm_idx] = fr_full
        fr_final_grid[stp_idx, hsm_idx] = fr_final

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    stp_labels = [str(s) for s in stps]
    hsm_labels = [str(h) for h in hsms]

    for ax, grid, title in [
        (axes[0], fr_full_grid, "FR (full matrix)"),
        (axes[1], fr_final_grid, "FR (final checkpoint)"),
    ]:
        im = ax.imshow(grid, cmap="YlOrRd", aspect="auto", origin="lower")
        ax.set_xticks(range(len(hsms)))
        ax.set_xticklabels(hsm_labels)
        ax.set_yticks(range(len(stps)))
        ax.set_yticklabels(stp_labels)
        ax.set_xlabel("hider_speed_mult")
        ax.set_ylabel("seeker_time_penalty")
        ax.set_title(title, fontweight="bold")
        plt.colorbar(im, ax=ax, label="Forgetting Regret")

        for i in range(len(stps)):
            for j in range(len(hsms)):
                val = grid[i, j]
                color = "white" if val > (grid.max() + grid.min()) / 2 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=9, color=color)

    fig.suptitle(
        "Forgetting Regret: how much does the seeker regress\n"
        "from its own training-trajectory peak?",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    out = Path(__file__).parent / "forgetting_regret.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    # Print table
    print(f"\n{'':>20s}  ", "  ".join(f"HSM={h}" for h in hsms))
    for i, stp in enumerate(stps):
        row = "  ".join(f"{fr_full_grid[i,j]:.3f}" for j in range(len(hsms)))
        print(f"  STP={str(stp):<6s} (full)  {row}")
        row = "  ".join(f"{fr_final_grid[i,j]:.3f}" for j in range(len(hsms)))
        print(f"  STP={str(stp):<6s} (final) {row}")


if __name__ == "__main__":
    main()
