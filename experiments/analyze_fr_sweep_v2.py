#!/usr/bin/env python3
"""Analyze FR sweep v2 results."""
import os
import csv
import numpy as np
from collections import defaultdict

BASE = "experiments/results/fr_sweep_v2"


def collect_results():
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for preset in os.listdir(BASE):
        ppath = os.path.join(BASE, preset)
        if not os.path.isdir(ppath) or preset == "logs":
            continue
        for config_dir in os.listdir(ppath):
            cpath = os.path.join(ppath, config_dir)
            if not os.path.isdir(cpath):
                continue
            parts = config_dir.split("_")
            a_val = float(parts[0][1:])
            algo = parts[1]
            for seed_dir in os.listdir(cpath):
                spath = os.path.join(cpath, seed_dir)
                if not os.path.isdir(spath):
                    continue
                for run_dir in os.listdir(spath):
                    mpath = os.path.join(spath, run_dir, "metrics.csv")
                    if os.path.exists(mpath):
                        with open(mpath) as f:
                            reader = csv.reader(f)
                            header = next(reader)
                            rows = list(reader)
                            if rows:
                                swr = float(rows[-1][8])
                                results[algo][preset][a_val].append(swr)
    return results


def print_table(results):
    for algo in ["ppo", "sac"]:
        print()
        print("=" * 75)
        print(f"  {algo.upper()} Results (Optimized Hyperparameters)")
        print("=" * 75)

        a_values = sorted(set(a for p in results[algo].values() for a in p.keys()))

        # SWR table
        print(f"\n  Seeker Win Rate (mean over 3 seeds):")
        header = f"  {'Preset':22s}" + "".join(f"  A={a:.2f}" for a in a_values) + "   mean"
        print(header)
        print("  " + "-" * (len(header) - 2))

        preset_means = {}
        for preset in sorted(results[algo].keys()):
            row = f"  {preset:22s}"
            vals = []
            for a in a_values:
                swrs = results[algo][preset].get(a, [])
                if swrs:
                    m = np.mean(swrs)
                    vals.append(m)
                    row += f"  {m:.3f}"
                else:
                    row += f"    N/A"
            if vals:
                preset_means[preset] = np.mean(vals)
                row += f"  {np.mean(vals):.3f}"
            print(row)

        # Balance table
        print(f"\n  Balance Score (0.5 = perfect):")
        header = f"  {'Preset':22s}" + "".join(f"  A={a:.2f}" for a in a_values) + "   mean"
        print(header)
        print("  " + "-" * (len(header) - 2))

        best_balance = 0
        best_config = ""
        for preset in sorted(results[algo].keys()):
            row = f"  {preset:22s}"
            vals = []
            for a in a_values:
                swrs = results[algo][preset].get(a, [])
                if swrs:
                    m = np.mean(swrs)
                    b = min(m, 1 - m)
                    vals.append(b)
                    row += f"  {b:.3f}"
                    if b > best_balance:
                        best_balance = b
                        best_config = f"{preset} / A={a:.2f}"
                else:
                    row += f"    N/A"
            if vals:
                row += f"  {np.mean(vals):.3f}"
            print(row)

        print(f"\n  Best balance: {best_config} (score={best_balance:.4f})")

        # Std dev for top configs
        print(f"\n  Per-seed detail for top 3 balanced configs:")
        all_configs = []
        for preset in results[algo]:
            for a in results[algo][preset]:
                swrs = results[algo][preset][a]
                m = np.mean(swrs)
                b = min(m, 1 - m)
                all_configs.append((b, preset, a, swrs))
        all_configs.sort(reverse=True)
        for b, preset, a, swrs in all_configs[:3]:
            seeds_str = ", ".join(f"{s:.3f}" for s in swrs)
            print(f"    {preset} A={a:.2f}: SWR=[{seeds_str}] mean={np.mean(swrs):.3f} std={np.std(swrs):.3f}")


def compare_with_v1():
    """Compare v2 (optimized) vs v1 (default hyperparams)."""
    v1_base = "experiments/results/fr_sweep"
    if not os.path.isdir(v1_base):
        return

    print("\n" + "=" * 75)
    print("  COMPARISON: v1 (default HPs) vs v2 (optimized HPs)")
    print("=" * 75)

    for algo in ["ppo", "sac"]:
        v1_swrs = []
        v2_swrs = []
        for preset_dir in os.listdir(v1_base):
            v1p = os.path.join(v1_base, preset_dir)
            v2p = os.path.join(BASE, preset_dir)
            if not os.path.isdir(v1p) or not os.path.isdir(v2p):
                continue
            for config_dir in os.listdir(v1p):
                if f"_{algo}" not in config_dir:
                    continue
                # v1
                for root, _, files in os.walk(os.path.join(v1p, config_dir)):
                    if "metrics.csv" in files:
                        with open(os.path.join(root, "metrics.csv")) as f:
                            rows = list(csv.reader(f))
                            if len(rows) > 1:
                                v1_swrs.append(float(rows[-1][8]))
                # v2
                for root, _, files in os.walk(os.path.join(v2p, config_dir)):
                    if "metrics.csv" in files:
                        with open(os.path.join(root, "metrics.csv")) as f:
                            rows = list(csv.reader(f))
                            if len(rows) > 1:
                                v2_swrs.append(float(rows[-1][8]))

        if v1_swrs and v2_swrs:
            v1_bal = np.mean([min(s, 1 - s) for s in v1_swrs])
            v2_bal = np.mean([min(s, 1 - s) for s in v2_swrs])
            v1_swr = np.mean(v1_swrs)
            v2_swr = np.mean(v2_swrs)
            print(f"\n  {algo.upper()}:")
            print(f"    v1: mean SWR={v1_swr:.3f}, mean balance={v1_bal:.3f} (n={len(v1_swrs)})")
            print(f"    v2: mean SWR={v2_swr:.3f}, mean balance={v2_bal:.3f} (n={len(v2_swrs)})")
            diff = v2_bal - v1_bal
            print(f"    Improvement: {'+' if diff >= 0 else ''}{diff:.3f} balance")


if __name__ == "__main__":
    results = collect_results()
    print_table(results)
    compare_with_v1()
