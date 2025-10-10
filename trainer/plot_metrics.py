#!/usr/bin/env python3
"""
Legacy-compatible plotting helper.

This script now delegates to trainer.monitor_runs to produce per-run charts.
Run selection mirrors the old behaviour but operates on the new run-directory layout.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Sequence

from monitor_runs import (
    RunMetrics,
    compute_run_summary,
    discover_runs,
    load_run_from_csv,
    load_run_metrics,
    load_legacy_run,
    render_per_run_charts,
    write_run_summary,
)


def resolve_run(
    approach: Optional[str],
    run_id: Optional[str],
    csv_path: Optional[str],
    include_all: bool,
) -> Optional[RunMetrics]:
    if csv_path:
        csv_abs = os.path.abspath(csv_path)
        if not os.path.exists(csv_abs):
            print(f"CSV file not found: {csv_abs}", file=sys.stderr)
            return None
        base_dir = os.path.dirname(csv_abs)
        inferred_run = run_id or os.path.splitext(os.path.basename(csv_abs))[0]
        inferred_approach = approach or "manual"
        return load_run_from_csv(inferred_approach, inferred_run, csv_abs, base_dir)
    if approach and run_id:
        return load_run_metrics(approach, run_id)
    runs = discover_runs([approach] if approach else [], [], include_all)
    if runs:
        return runs[0]
    return load_legacy_run()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Plot training metrics for a single run.")
    parser.add_argument("--approach", help="Approach name (used with --run).")
    parser.add_argument("--run", help="Run identifier inside trainer/logs/runs/<approach>/.")
    parser.add_argument("--csv", help="Path to a metrics CSV (overrides --approach/--run).")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to emit charts into (defaults to the run directory).",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=10,
        help="Window size for moving averages (default: 10 episodes).",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="If no run is specified, include every run and pick the most recent overall (default: latest per approach).",
    )
    args = parser.parse_args(argv)

    metrics = resolve_run(args.approach, args.run, args.csv, args.all_runs)
    if metrics is None:
        print("No metrics available. Run scripts/train.sh to generate training data.", file=sys.stderr)
        return 1

    target_dir = os.path.abspath(args.output_dir) if args.output_dir else metrics.path
    render_per_run_charts(metrics, args.smooth_window, target_base=target_dir)
    # Write a summary clone if output redirected.
    summary = compute_run_summary(metrics)
    if target_dir != metrics.path:
        temp = RunMetrics(
            approach=metrics.approach,
            run_id=metrics.run_id,
            path=target_dir,
            episodes=metrics.episodes,
            updates=metrics.updates,
            reward_mean=metrics.reward_mean,
            reward_sum=metrics.reward_sum,
            seeker_reward_sum=metrics.seeker_reward_sum,
            seeker_reward_mean=metrics.seeker_reward_mean,
            seeker_steps=metrics.seeker_steps,
            seeker_avg_distance=metrics.seeker_avg_distance,
            hider_reward_sum=metrics.hider_reward_sum,
            hider_reward_mean=metrics.hider_reward_mean,
            hider_steps=metrics.hider_steps,
            hider_avg_distance=metrics.hider_avg_distance,
            winners=metrics.winners,
            terminal_reason=metrics.terminal_reason,
            duration_sec=metrics.duration_sec,
            metadata=metrics.metadata,
        )
        write_run_summary(temp, summary)
    else:
        write_run_summary(metrics, summary)

    print(f"Charts saved under {os.path.join(target_dir, 'charts')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
