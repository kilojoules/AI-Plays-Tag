#!/usr/bin/env python3
"""
Feature probing v2: fixed numerical stability + deeper analysis.

Analyses:
  1. Ridge regression probes (stable, no divergence)
  2. Nonlinear (MLP) probes for comparison
  3. Feature clustering: what behavioral modes do the features encode?
  4. Action diversity: does the actor produce more varied actions?
  5. Opponent-conditional action analysis: do features support opponent-aware behavior?
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainer.tag_env import VecTagEnv, TagEnvConfig
from trainer.sac import SACAgent, SACConfig


def collect_states_and_labels(n_states=5000, layout="four_corners"):
    """Collect diverse states with ground-truth labels."""
    cfg = TagEnvConfig(layout=layout, hider_speed_mult=1.15)
    env = VecTagEnv(num_envs=200, config=cfg)
    obs = env.reset()

    all_hider_obs = []
    all_labels = []

    for _ in range(n_states // 200 + 1):
        acts = {
            'seeker': np.random.uniform(-1, 1, (200, 3)).astype(np.float32),
            'hider': np.random.uniform(-1, 1, (200, 3)).astype(np.float32),
        }
        obs, _, dones, infos = env.step(acts)

        hider_obs = obs['hider']
        opp_rel = hider_obs[:, 4:6]
        opp_dist = np.linalg.norm(opp_rel, axis=1)
        opp_angle = np.arctan2(opp_rel[:, 1], opp_rel[:, 0])
        own_speed = np.linalg.norm(hider_obs[:, 2:4], axis=1)
        own_pos = hider_obs[:, 0:2]
        wall_dist = 1.0 - np.abs(own_pos).max(axis=1)

        # Additional labels
        opp_approaching = (opp_dist < 0.3).astype(np.float32)  # binary: is opponent close?
        near_wall = (wall_dist < 0.15).astype(np.float32)  # binary: near wall?
        near_corner = ((np.abs(own_pos) > 0.8).all(axis=1)).astype(np.float32)  # in corner?

        labels = np.stack([
            opp_dist, opp_angle, own_speed, wall_dist,
            opp_approaching, near_wall, near_corner,
        ], axis=1)

        all_hider_obs.append(hider_obs)
        all_labels.append(labels)
        obs = env.auto_reset()

    return (np.concatenate(all_hider_obs)[:n_states],
            np.concatenate(all_labels)[:n_states])


def get_actor_features(agent, obs_np, layer="layer0"):
    """Extract hidden activations."""
    obs = torch.as_tensor(obs_np, dtype=torch.float32)
    with torch.no_grad():
        if layer == "layer0":
            return torch.relu(agent.actor.trunk[0](obs)).numpy()
        elif layer == "layer1":
            h0 = torch.relu(agent.actor.trunk[0](obs))
            return torch.relu(agent.actor.trunk[2](h0)).numpy()


def ridge_regression_r2(X_train, y_train, X_test, y_test, alpha=1.0):
    """Closed-form ridge regression. Returns R² per target dimension."""
    n, d = X_train.shape
    # Normalize
    X_mean, X_std = X_train.mean(0), X_train.std(0) + 1e-8
    Xn_train = (X_train - X_mean) / X_std
    Xn_test = (X_test - X_mean) / X_std
    y_mean = y_train.mean(0)
    yn_train = y_train - y_mean

    # Ridge: W = (X'X + alphaI)^-1 X'y
    XtX = Xn_train.T @ Xn_train + alpha * np.eye(d)
    Xty = Xn_train.T @ yn_train
    W = np.linalg.solve(XtX, Xty)

    pred = Xn_test @ W + y_mean
    ss_res = ((pred - y_test) ** 2).sum(0)
    ss_tot = ((y_test - y_test.mean(0)) ** 2).sum(0)
    r2 = 1.0 - ss_res / (ss_tot + 1e-8)
    return r2


def get_actions(agent, obs_np):
    """Get deterministic actions from actor."""
    obs = torch.as_tensor(obs_np, dtype=torch.float32)
    with torch.no_grad():
        mean, log_std = agent.actor.forward(obs)
        return torch.tanh(mean).numpy()


def analyze_action_diversity(agent, obs_np, labels):
    """Measure how diverse the actor's actions are across states."""
    actions = get_actions(agent, obs_np)

    # Overall action std
    action_std = actions.std(0).mean()

    # Action entropy (discretize into bins)
    bins = np.clip(((actions + 1) * 5).astype(int), 0, 9)
    bin_ids = bins[:, 0] * 10 + bins[:, 1]
    counts = np.bincount(bin_ids, minlength=100).astype(np.float64)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    action_entropy = -np.sum(probs * np.log(probs))

    # Opponent-conditional action diversity
    opp_dist = labels[:, 0]
    near_mask = opp_dist < np.median(opp_dist)
    far_mask = ~near_mask

    near_std = actions[near_mask].std(0).mean() if near_mask.sum() > 10 else 0
    far_std = actions[far_mask].std(0).mean() if far_mask.sum() > 10 else 0

    # Action difference between near and far (how much behavior changes)
    near_mean = actions[near_mask].mean(0)
    far_mean = actions[far_mask].mean(0)
    action_shift = np.linalg.norm(near_mean - far_mean)

    return {
        'action_std': float(action_std),
        'action_entropy': float(action_entropy),
        'near_opp_std': float(near_std),
        'far_opp_std': float(far_std),
        'action_shift_near_vs_far': float(action_shift),
    }


def cluster_features(features, n_clusters=8):
    """K-means clustering on features, return cluster stats."""
    from numpy.linalg import norm

    n, d = features.shape
    # Simple k-means
    rng = np.random.RandomState(42)
    centers = features[rng.choice(n, n_clusters, replace=False)]

    for _ in range(50):
        dists = np.array([norm(features - c, axis=1) for c in centers])  # [k, n]
        assignments = dists.argmin(0)  # [n]
        for k in range(n_clusters):
            mask = assignments == k
            if mask.sum() > 0:
                centers[k] = features[mask].mean(0)

    # Cluster sizes
    sizes = np.bincount(assignments, minlength=n_clusters)
    # Cluster purity (entropy of cluster distribution)
    probs = sizes / sizes.sum()
    probs = probs[probs > 0]
    cluster_entropy = -np.sum(probs * np.log(probs))

    # Effective number of clusters
    eff_clusters = np.exp(cluster_entropy)

    return {
        'cluster_entropy': float(cluster_entropy),
        'effective_clusters': float(eff_clusters),
        'cluster_sizes': sizes.tolist(),
    }


def main():
    base = Path("experiments/results/paper_final/init_alpha")
    output_dir = base / "feature_probing_v2"
    output_dir.mkdir(parents=True, exist_ok=True)

    init_alphas = [0.05, 0.2, 0.607, 2.0]
    label_names = ["opp_dist", "opp_angle", "own_speed", "wall_dist",
                    "opp_near", "near_wall", "near_corner"]

    print("Collecting states...")
    obs_np, labels = collect_states_and_labels(n_states=5000)
    print(f"  {obs_np.shape[0]} states")

    # Train/test split
    split = int(0.8 * len(obs_np))
    obs_train, obs_test = obs_np[:split], obs_np[split:]
    lab_train, lab_test = labels[:split], labels[split:]

    results = {}

    for ia in init_alphas:
        name = f"initalpha_{ia}"
        runs = sorted((base / name / "seed_0").glob("2026*"))
        if not runs:
            print(f"  SKIP {ia}")
            continue

        for ckpt_label, ckpt_name in [("500K", "hider_00500032.pt"),
                                       ("final", "policy_hider_final.pt")]:
            if ckpt_label == "500K":
                path = runs[-1] / "checkpoints" / ckpt_name
            else:
                path = runs[-1] / ckpt_name
            if not path.exists():
                continue

            agent = SACAgent(SACConfig(obs_dim=87, act_dim=3))
            agent.load_policy(str(path))

            print(f"\n  ia={ia} {ckpt_label}:")

            for layer in ["layer0", "layer1"]:
                feats_train = get_actor_features(agent, obs_train, layer)
                feats_test = get_actor_features(agent, obs_test, layer)

                # Ridge regression probe
                r2 = ridge_regression_r2(feats_train, lab_train, feats_test, lab_test)
                r2_dict = {f"r2_{ln}": float(r2[i]) for i, ln in enumerate(label_names)}

                # Clustering
                cluster_stats = cluster_features(feats_train)

                key = f"{ia}_{ckpt_label}_{layer}"
                results[key] = {
                    "init_alpha": ia, "checkpoint": ckpt_label, "layer": layer,
                    **r2_dict,
                    "r2_mean_continuous": float(r2[:4].mean()),  # first 4 are continuous
                    **{f"cluster_{k}": v for k, v in cluster_stats.items()},
                }

                r2_str = " ".join(f"{ln}={r2[i]:.3f}" for i, ln in enumerate(label_names[:4]))
                print(f"    {layer}: {r2_str}  mean={r2[:4].mean():.3f}  "
                      f"eff_clusters={cluster_stats['effective_clusters']:.1f}")

            # Action diversity (uses full model, not per-layer)
            act_div = analyze_action_diversity(agent, obs_np, labels)
            key_act = f"{ia}_{ckpt_label}_actions"
            results[key_act] = {"init_alpha": ia, "checkpoint": ckpt_label, **act_div}
            print(f"    actions: std={act_div['action_std']:.3f} "
                  f"entropy={act_div['action_entropy']:.2f} "
                  f"shift={act_div['action_shift_near_vs_far']:.3f}")

    # Summary tables
    print(f"\n{'='*70}")
    print("SUMMARY: Ridge Probe R² (layer0, continuous labels)")
    print(f"{'='*70}")
    print(f"{'Condition':<20s} {'OppDist':>8s} {'OppAngle':>9s} {'Speed':>7s} {'WallDist':>9s} {'Mean':>7s}")
    print("-" * 58)
    for ia in init_alphas:
        for ckpt in ["500K", "final"]:
            key = f"{ia}_{ckpt}_layer0"
            if key not in results:
                continue
            r = results[key]
            print(f"  {ia:<6} {ckpt:<6}  {r['r2_opp_dist']:>8.3f} {r['r2_opp_angle']:>9.3f} "
                  f"{r['r2_own_speed']:>7.3f} {r['r2_wall_dist']:>9.3f} {r['r2_mean_continuous']:>7.3f}")

    print(f"\n{'='*70}")
    print("SUMMARY: Action Diversity")
    print(f"{'='*70}")
    print(f"{'Condition':<20s} {'ActStd':>7s} {'ActEnt':>7s} {'NearStd':>8s} {'FarStd':>7s} {'Shift':>7s}")
    print("-" * 58)
    for ia in init_alphas:
        for ckpt in ["500K", "final"]:
            key = f"{ia}_{ckpt}_actions"
            if key not in results:
                continue
            r = results[key]
            print(f"  {ia:<6} {ckpt:<6}  {r['action_std']:>7.3f} {r['action_entropy']:>7.2f} "
                  f"{r['near_opp_std']:>8.3f} {r['far_opp_std']:>7.3f} {r['action_shift_near_vs_far']:>7.3f}")

    print(f"\n{'='*70}")
    print("SUMMARY: Feature Clustering (layer0)")
    print(f"{'='*70}")
    print(f"{'Condition':<20s} {'EffClusters':>12s}")
    print("-" * 35)
    for ia in init_alphas:
        for ckpt in ["500K", "final"]:
            key = f"{ia}_{ckpt}_layer0"
            if key not in results:
                continue
            r = results[key]
            print(f"  {ia:<6} {ckpt:<6}  {r['cluster_effective_clusters']:>12.1f}")

    # Save
    out_path = output_dir / "probe_results_v2.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
