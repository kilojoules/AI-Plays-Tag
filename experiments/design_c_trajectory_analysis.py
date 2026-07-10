#!/usr/bin/env python3
"""
Design C trajectory/behavior analysis — mechanism for σ_seeker[R7] = 4× σ_seeker[R4].

For each policy in the pool, roll out N episodes against a fixed reference
hider and collect per-episode trajectory features. Compare the behavioral
fingerprint of each seeker to its position in the SVD U1/U2 embedding to
ask: are R7 seeders' wide policy-quality variance driven by genuinely
different *behaviors* (strategic diversity, per critic's framing) or by
draws on a single skill axis (exploration variance)?

This is the diagnostic the idea-critic put at the top of their priority
list: "trajectory clustering of R7 seekers vs each fixed hider pool —
this is the single most informative analysis and it uses only existing
checkpoints."

Outputs (under experiments/results/design_c/behavior/):
    behavior_features.csv   — per-policy mean trajectory features
    behavior_pca.png        — PC1 vs PC2 of behavior space
    behavior_vs_svd.png     — behavior PC1 vs SVD U1 alignment
    behavior_summary.txt    — narrative interpretation

Usage:
    python experiments/design_c_trajectory_analysis.py
    python experiments/design_c_trajectory_analysis.py --episodes 30 --reference 21
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from trainer.tag_env import VecTagEnv, TagEnvConfig
from trainer.ppo import PPOAgent, PPOConfig

GAUNTLET_DIR = ROOT / "experiments" / "results" / "design_c" / "gauntlet"
SVD_DIR = ROOT / "experiments" / "results" / "design_c" / "svd"
OUT_DIR = ROOT / "experiments" / "results" / "design_c" / "behavior"

MAX_STEPS = 200
HSM = 1.15
LAYOUT = "four_corners"


def load_policy(path: Path, obs_dim: int, act_dim: int) -> PPOAgent:
    cfg = PPOConfig(obs_dim=obs_dim, act_dim=act_dim)
    agent = PPOAgent(cfg)
    ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
    agent.pi.load_state_dict(ckpt["pi"])
    agent.vf.load_state_dict(ckpt["vf"])
    return agent


def batch_act(policy, obs_batch):
    with torch.no_grad():
        x = torch.as_tensor(obs_batch, dtype=torch.float32)
        logits = policy.pi(x)
        mean, log_std = torch.chunk(logits, 2, dim=-1)
        log_std = torch.clamp(log_std, -2.0, 1.5)
        std = torch.exp(log_std)
        action = torch.tanh(torch.distributions.Normal(mean, std).sample())
        return action.cpu().numpy()


def collect_features(seeker: PPOAgent, hider: PPOAgent, env_config: TagEnvConfig,
                     n_episodes: int) -> dict:
    """Roll out n_episodes seeker-vs-hider; return aggregated trajectory features.

    Role indexing: VecTagEnv randomizes which agent slot (0/1) is the seeker
    on every reset (tag_env.py, self.seeker_idx), so positions must be
    gathered via env.seeker_idx. Indexing positions[:, 0] as "the seeker"
    (the pre-2026-07 version of this function) scrambled roles in ~50% of
    episodes; wr was unaffected (info['tagged'] is role-independent) but all
    positional features were.
    """
    env = VecTagEnv(num_envs=n_episodes, config=env_config)
    obs = env.reset()
    env_ids = np.arange(n_episodes)
    active = np.ones(n_episodes, dtype=bool)
    tagged = np.zeros(n_episodes, dtype=bool)
    steps_taken = np.zeros(n_episodes, dtype=np.int32)

    # Spatial thresholds as fractions of the actual arena half-extent
    # (the original constants 3.0/6.0 were calibrated to a 7.5 half-extent;
    # cfg.arena_half is 15.0). center overlaps the hider safe zone (r=2.5
    # at origin) only marginally at 0.4*15 = 6.0.
    half = env_config.arena_half
    center_thresh = 0.4 * half
    wall_thresh = 0.8 * half

    # Per-step seeker position trace, per env (role-correct gather)
    path_len = np.zeros(n_episodes, dtype=np.float32)
    last_pos = env.positions[env_ids, env.seeker_idx].copy()
    # Track summed seeker-hider distance over time
    dist_sum = np.zeros(n_episodes, dtype=np.float32)
    time_in_center = np.zeros(n_episodes, dtype=np.float32)
    time_at_wall = np.zeros(n_episodes, dtype=np.float32)
    # Final seeker position
    final_x = np.zeros(n_episodes, dtype=np.float32)
    final_y = np.zeros(n_episodes, dtype=np.float32)

    for step in range(MAX_STEPS):
        if not active.any():
            break
        s_act = batch_act(seeker, obs["seeker"])
        h_act = batch_act(hider, obs["hider"])
        obs, _, dones, info = env.step({"seeker": s_act, "hider": h_act})

        # Update active and tag flags BEFORE accumulating features so terminated
        # envs stop contributing.
        newly_done = dones & active
        if newly_done.any():
            tagged[newly_done] = info.get("tagged", np.zeros(n_episodes, dtype=bool))[newly_done]
            steps_taken[newly_done] = step + 1

        # Trajectory features over currently-active envs (role-correct gather)
        cur_pos = env.positions[env_ids, env.seeker_idx]
        hider_pos = env.positions[env_ids, 1 - env.seeker_idx]
        diff = cur_pos - last_pos
        path_len[active] += np.linalg.norm(diff[active], axis=-1)
        dist_sum[active] += np.linalg.norm((cur_pos - hider_pos)[active], axis=-1)
        in_center = (np.abs(cur_pos[:, 0]) < center_thresh) & (np.abs(cur_pos[:, 1]) < center_thresh)
        time_in_center[active & in_center] += 1
        at_wall = (np.abs(cur_pos[:, 0]) > wall_thresh) | (np.abs(cur_pos[:, 1]) > wall_thresh)
        time_at_wall[active & at_wall] += 1

        last_pos = cur_pos.copy()
        for eid in np.where(newly_done)[0]:
            final_x[eid] = cur_pos[eid, 0]
            final_y[eid] = cur_pos[eid, 1]
        active &= ~newly_done

    # Anything still active at MAX_STEPS: final pos taken now
    if active.any():
        cur_pos = env.positions[env_ids, env.seeker_idx]
        for eid in np.where(active)[0]:
            final_x[eid] = cur_pos[eid, 0]
            final_y[eid] = cur_pos[eid, 1]
            steps_taken[eid] = MAX_STEPS

    n_actual_steps = np.maximum(steps_taken, 1)
    return dict(
        wr=float(tagged.mean()),
        mean_tag_time=float(steps_taken[tagged].mean()) if tagged.any() else float(MAX_STEPS),
        mean_path_len=float(path_len.mean()),
        mean_speed=float((path_len / n_actual_steps).mean()),
        mean_dist=float((dist_sum / n_actual_steps).mean()),
        frac_center=float((time_in_center / n_actual_steps).mean()),
        frac_wall=float((time_at_wall / n_actual_steps).mean()),
        final_radius=float(np.linalg.norm(np.column_stack([final_x, final_y]), axis=1).mean()),
        n_episodes=int(n_episodes),
    )


def localize(ts_dir) -> Path:
    """policy_index.csv stores absolute paths from the machine that built it
    (e.g. dead LUMI scratch). Re-root them at this repo's experiments/results."""
    s = str(ts_dir)
    i = s.find("experiments/results")
    return ROOT / s[i:] if i >= 0 else Path(s)


def discover_run(base: Path, reward: str, A: float, seed: int) -> Path | None:
    def _a_str(A): return f"A{int(A * 100):02d}"
    run_dir = base / reward / _a_str(A) / f"seed_{seed}"
    if not run_dir.exists():
        return None
    for ts_dir in sorted(run_dir.glob("2*"), reverse=True):
        if (ts_dir / "policy_seeker_final.pt").exists() and \
           (ts_dir / "policy_hider_final.pt").exists():
            return ts_dir
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30,
                    help="Episodes per seeker per reference hider.")
    ap.add_argument("--reference", type=str, default="anchors",
                    help="Reference hider(s): 'anchors' = all prereg anchors "
                         "(default; single-opponent evals are untrustworthy, "
                         "results.md section 9/13), or comma-separated policy ids.")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    idx = pd.read_csv(GAUNTLET_DIR / "policy_index.csv")
    print(f"Pool: {len(idx)} policies")

    if args.reference == "anchors":
        ref_rows = idx[idx.source == "anchor"]
        if not len(ref_rows):
            ref_rows = idx.iloc[[0]]
    else:
        ref_ids = [int(s) for s in args.reference.split(",")]
        ref_rows = idx[idx.id.isin(ref_ids)]
    print(f"Reference hiders: policy ids {list(ref_rows.id)}")

    env_probe = VecTagEnv(num_envs=1, config=TagEnvConfig(layout=LAYOUT, hider_speed_mult=HSM))
    obs_dim, act_dim = env_probe.obs_dim, env_probe.act_dim
    env_cfg = TagEnvConfig(layout=LAYOUT, hider_speed_mult=HSM)

    ref_hiders = [(int(r.id), load_policy(localize(r.ts_dir) / "policy_hider_final.pt",
                                          obs_dim, act_dim))
                  for _, r in ref_rows.iterrows()]

    # Behavior features averaged over the reference panel; wr kept per-ref
    # in wide columns (wr_ref<id>) for dispersion diagnostics.
    rows = []
    num_cols = ["wr", "mean_tag_time", "mean_path_len", "mean_speed",
                "mean_dist", "frac_center", "frac_wall", "final_radius"]
    for _, r in idx.iterrows():
        ts = localize(r.ts_dir)
        seeker = load_policy(ts / "policy_seeker_final.pt", obs_dim, act_dim)
        per_ref = {rid: collect_features(seeker, h, env_cfg, args.episodes)
                   for rid, h in ref_hiders}
        feats = {c: float(np.mean([f[c] for f in per_ref.values()])) for c in num_cols}
        for rid, f in per_ref.items():
            feats[f"wr_ref{rid}"] = f["wr"]
        feats.update(dict(id=int(r.id), source=r.source, reward=r.reward,
                          A=float(r.A), seed=int(r.seed)))
        rows.append(feats)
        print(f"  id={r.id:>2} {r.reward[:2]}/A{r.A}/s{r.seed:<2}  "
              f"wr={feats['wr']:.2f}  path={feats['mean_path_len']:.1f}  "
              f"dist={feats['mean_dist']:.2f}  center={feats['frac_center']:.2f}  "
              f"wall={feats['frac_wall']:.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "behavior_features.csv", index=False)

    # PCA on standardized BEHAVIOR features only. mean_tag_time is excluded:
    # it is computed over tagged episodes only (a success proxy — outcome
    # leakage inside a behavior fingerprint that gets correlated with the
    # outcome SVD). It stays in the CSV for descriptive use.
    feat_cols = ["mean_path_len", "mean_speed", "mean_dist", "frac_center",
                 "frac_wall", "final_radius"]
    X = df[feat_cols].to_numpy()
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    U, S, Vt = np.linalg.svd(Xs - Xs.mean(0), full_matrices=False)
    df["beh_PC1"] = U[:, 0] * S[0]
    df["beh_PC2"] = U[:, 1] * S[1]
    explained = (S ** 2) / (S ** 2).sum()
    df.to_csv(OUT_DIR / "behavior_features.csv", index=False)

    # Cross-link with SVD outcome embedding
    try:
        svd_idx = pd.read_csv(SVD_DIR / "policy_embedding.csv")
        df = df.merge(svd_idx[["id", "U1_seeker", "U2_seeker"]], on="id", how="left")
        df.to_csv(OUT_DIR / "behavior_features.csv", index=False)
        # Correlation between behavior PC1 and outcome U1 (transitive skill axis)
        corr_pc1_u1 = float(np.corrcoef(df["beh_PC1"], df["U1_seeker"])[0, 1])
        corr_pc2_u2 = float(np.corrcoef(df["beh_PC2"], df["U2_seeker"])[0, 1])
        print(f"\nbeh_PC1 vs SVD U1 correlation: {corr_pc1_u1:+.3f}")
        print(f"beh_PC2 vs SVD U2 correlation: {corr_pc2_u2:+.3f}")
    except Exception as e:
        corr_pc1_u1 = corr_pc2_u2 = None
        print(f"(SVD cross-link skipped: {e})", file=sys.stderr)

    lines = []
    ref_desc = ",".join(str(rid) for rid, _ in ref_hiders)
    lines.append("=== Design C behavior analysis ===")
    lines.append(f"Pool: {len(idx)} policies; reference hiders: ids {ref_desc}")
    lines.append(f"Episodes per seeker per reference: {args.episodes}")
    lines.append("")
    lines.append("PC1 share of behavior variance: %.3f" % explained[0])
    lines.append("PC2 share of behavior variance: %.3f" % explained[1])
    if corr_pc1_u1 is not None:
        lines.append(f"Correlation behavior PC1 vs outcome SVD U1: {corr_pc1_u1:+.3f}")
        lines.append(f"Correlation behavior PC2 vs outcome SVD U2: {corr_pc2_u2:+.3f}")
        lines.append("  (signs of PC2/U2 are arbitrary and component order is "
                     "noise-fragile; only |corr| of PC1-U1 feeds the verdict)")
        if abs(corr_pc1_u1) > 0.7:
            lines.append("  → behavior PC1 strongly aligned with skill axis (high-WR seekers behave similarly)")
        else:
            lines.append("  → behavior PC1 NOT strongly aligned with skill axis "
                         "(high-WR seekers may use different behaviors)")
    lines.append("")
    lines.append("Per-policy mean features, sorted by behavior PC1:")
    show = df.sort_values("beh_PC1", ascending=False)[
        ["id", "source", "reward", "A", "seed",
         "wr", "mean_path_len", "mean_dist", "frac_center", "frac_wall",
         "beh_PC1", "beh_PC2"]
        + (["U1_seeker", "U2_seeker"] if corr_pc1_u1 is not None else [])
    ]
    lines.append(show.round(3).to_string(index=False))

    out_txt = OUT_DIR / "behavior_summary.txt"
    out_txt.write_text("\n".join(lines))
    print("\n" + "\n".join(lines[:12]))

    # Plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        markers = {"R4_sparse": "o", "R7_kitchen_sink": "s"}
        colors = {0.0: "tab:blue", 0.5: "tab:orange"}

        fig, ax = plt.subplots(figsize=(8, 7))
        for _, r in df.iterrows():
            ax.scatter(r["beh_PC1"], r["beh_PC2"],
                       marker=markers[r["reward"]], color=colors[r["A"]],
                       s=120, edgecolor="k", alpha=0.85)
            ax.annotate(f"s{int(r['seed'])}", (r["beh_PC1"], r["beh_PC2"]),
                        fontsize=8, ha="center", va="center")
        ax.set_xlabel(f"behavior PC1 ({explained[0]*100:.1f}% var)")
        ax.set_ylabel(f"behavior PC2 ({explained[1]*100:.1f}% var)")
        ax.set_title(f"Behavior fingerprint (vs reference hiders {ref_desc})")
        ax.grid(alpha=0.3); ax.axhline(0, color="grey", lw=0.5); ax.axvline(0, color="grey", lw=0.5)
        from matplotlib.lines import Line2D
        legend = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:blue", markersize=10, label="R4 / A=0"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:orange", markersize=10, label="R4 / A=0.5"),
            Line2D([0], [0], marker="s", color="w", markerfacecolor="tab:blue", markersize=10, label="R7 / A=0"),
            Line2D([0], [0], marker="s", color="w", markerfacecolor="tab:orange", markersize=10, label="R7 / A=0.5"),
        ]
        ax.legend(handles=legend, loc="best")
        plt.tight_layout(); plt.savefig(OUT_DIR / "behavior_pca.png", dpi=120); plt.close()

        if corr_pc1_u1 is not None:
            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            for ax, (xc, yc, lbl) in zip(axes, [
                ("beh_PC1", "U1_seeker", f"PC1↔U1  r={corr_pc1_u1:+.3f}"),
                ("beh_PC2", "U2_seeker", f"PC2↔U2  r={corr_pc2_u2:+.3f}"),
            ]):
                for _, r in df.iterrows():
                    ax.scatter(r[xc], r[yc], marker=markers[r["reward"]],
                               color=colors[r["A"]], s=110, edgecolor="k", alpha=0.85)
                ax.set_xlabel(xc); ax.set_ylabel(yc); ax.set_title(lbl)
                ax.grid(alpha=0.3); ax.axhline(0, lw=0.5, color="grey"); ax.axvline(0, lw=0.5, color="grey")
            plt.tight_layout(); plt.savefig(OUT_DIR / "behavior_vs_svd.png", dpi=120); plt.close()

        print(f"Plots: {OUT_DIR}")
    except Exception as e:
        print(f"(plot skipped: {e})", file=sys.stderr)


if __name__ == "__main__":
    main()
