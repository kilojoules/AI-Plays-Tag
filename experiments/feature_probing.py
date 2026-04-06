#!/usr/bin/env python3
"""
Feature probing: what did the actor learn during bootstrapping?

Trains linear probes on actor layer-0 features to predict task-relevant
quantities (opponent direction, distance, own speed, wall proximity).
If init_alpha=0.607's features are more linearly separable for these
quantities, that explains WHY those features produce stronger agents.

Also computes CKA (Centered Kernel Alignment) between conditions to
measure how different the learned representations are.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainer.tag_env import VecTagEnv, TagEnvConfig
from trainer.sac import SACAgent, SACConfig


def collect_states_and_labels(n_states=5000, layout="four_corners"):
    """Collect diverse states with ground-truth labels for probing."""
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

        hider_obs = obs['hider']  # [200, 87]
        # Extract ground-truth labels from observations
        # obs[0:2] = own pos/arena_half
        # obs[2:4] = own vel/10
        # obs[4:6] = relative opponent pos/arena_half
        opp_rel = hider_obs[:, 4:6]
        opp_dist = np.linalg.norm(opp_rel, axis=1)
        opp_angle = np.arctan2(opp_rel[:, 1], opp_rel[:, 0])
        own_speed = np.linalg.norm(hider_obs[:, 2:4], axis=1)
        own_pos = hider_obs[:, 0:2]
        wall_dist = 1.0 - np.abs(own_pos).max(axis=1)  # normalized

        labels = np.stack([
            opp_dist,        # 0: opponent distance
            opp_angle,       # 1: opponent angle
            own_speed,       # 2: own speed
            wall_dist,       # 3: wall distance
        ], axis=1)  # [200, 4]

        all_hider_obs.append(hider_obs)
        all_labels.append(labels)
        obs = env.auto_reset()

    obs_all = np.concatenate(all_hider_obs)[:n_states]
    labels_all = np.concatenate(all_labels)[:n_states]
    return torch.as_tensor(obs_all, dtype=torch.float32), \
           torch.as_tensor(labels_all, dtype=torch.float32)


def get_actor_features(agent, obs, layer="layer0"):
    """Extract hidden activations from actor trunk."""
    with torch.no_grad():
        if layer == "layer0":
            h = agent.actor.trunk[0](obs)  # Linear
            return torch.relu(h)
        elif layer == "layer1":
            h0 = torch.relu(agent.actor.trunk[0](obs))
            h1 = agent.actor.trunk[2](h0)
            return torch.relu(h1)


def train_linear_probe(features, labels, n_epochs=200, lr=0.01):
    """Train a linear probe and return R² for each label dimension."""
    n, d = features.shape
    n_labels = labels.shape[1]

    # Split train/test
    split = int(0.8 * n)
    X_train, X_test = features[:split], features[split:]
    y_train, y_test = labels[:split], labels[split:]

    # Normalize
    X_mean, X_std = X_train.mean(0), X_train.std(0) + 1e-8
    X_train = (X_train - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std

    y_mean, y_std = y_train.mean(0), y_train.std(0) + 1e-8
    y_train_n = (y_train - y_mean) / y_std
    y_test_n = (y_test - y_mean) / y_std

    # Linear regression
    probe = nn.Linear(d, n_labels)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(n_epochs):
        pred = probe(X_train)
        loss = loss_fn(pred, y_train_n)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate R²
    with torch.no_grad():
        pred_test = probe(X_test)
        ss_res = ((pred_test - y_test_n) ** 2).sum(0)
        ss_tot = ((y_test_n - y_test_n.mean(0)) ** 2).sum(0)
        r2 = 1.0 - ss_res / (ss_tot + 1e-8)

    return r2.numpy()


def linear_cka(X, Y):
    """Compute Linear CKA between two feature matrices."""
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    hsic_xy = (X.T @ Y).norm() ** 2
    hsic_xx = (X.T @ X).norm() ** 2
    hsic_yy = (Y.T @ Y).norm() ** 2
    return float(hsic_xy / (torch.sqrt(hsic_xx * hsic_yy) + 1e-8))


def main():
    base = Path("experiments/results/paper_final/init_alpha")
    output_dir = base / "feature_probing"
    output_dir.mkdir(parents=True, exist_ok=True)

    obs_dim = 87
    act_dim = 3
    init_alphas = [0.05, 0.2, 0.607, 2.0]
    label_names = ["opp_dist", "opp_angle", "own_speed", "wall_dist"]

    print("Collecting diverse states...")
    obs, labels = collect_states_and_labels(n_states=5000)
    print(f"  {obs.shape[0]} states, {labels.shape[1]} label dims")

    results = {}
    features_by_condition = {}

    for ia in init_alphas:
        name = f"initalpha_{ia}"
        runs = sorted((base / name / "seed_0").glob("2026*"))
        if not runs:
            print(f"  SKIP {ia}")
            continue

        # Probe both 500K checkpoint and final
        for ckpt_label, ckpt_name in [("500K", "hider_00500032.pt"), ("final", "policy_hider_final.pt")]:
            if ckpt_label == "500K":
                path = runs[-1] / "checkpoints" / ckpt_name
            else:
                path = runs[-1] / ckpt_name

            if not path.exists():
                continue

            agent = SACAgent(SACConfig(obs_dim=obs_dim, act_dim=act_dim))
            agent.load_policy(str(path))

            for layer in ["layer0", "layer1"]:
                feats = get_actor_features(agent, obs, layer=layer)
                r2 = train_linear_probe(feats, labels)

                key = f"{ia}_{ckpt_label}_{layer}"
                results[key] = {
                    "init_alpha": ia, "checkpoint": ckpt_label, "layer": layer,
                    **{f"r2_{ln}": float(r2[i]) for i, ln in enumerate(label_names)},
                    "r2_mean": float(r2.mean()),
                }
                features_by_condition[key] = feats

                r2_str = " ".join(f"{ln}={r2[i]:.3f}" for i, ln in enumerate(label_names))
                print(f"  ia={ia:<6} {ckpt_label:<6} {layer}: {r2_str}  mean={r2.mean():.3f}")

    # CKA between conditions (layer0 final only)
    print(f"\nCKA Matrix (layer0, final checkpoint):")
    cka_keys = [f"{ia}_final_layer0" for ia in init_alphas if f"{ia}_final_layer0" in features_by_condition]
    cka_matrix = {}
    for k1 in cka_keys:
        for k2 in cka_keys:
            cka = linear_cka(features_by_condition[k1], features_by_condition[k2])
            cka_matrix[f"{k1}_vs_{k2}"] = cka

    # Print CKA
    print(f"{'':>20s}", end="")
    for k in cka_keys:
        print(f"  {k.split('_')[0]:>8s}", end="")
    print()
    for k1 in cka_keys:
        print(f"  {k1.split('_')[0]:>18s}", end="")
        for k2 in cka_keys:
            cka = cka_matrix[f"{k1}_vs_{k2}"]
            print(f"  {cka:>8.3f}", end="")
        print()

    # Save
    out_path = output_dir / "probe_results.json"
    with open(out_path, 'w') as f:
        json.dump({"probes": results, "cka": cka_matrix}, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
