#!/usr/bin/env python3
"""
Training monitoring suite for the AI tag game.

Generates per-run dashboards, aggregates metrics across approaches,
and emits comparison plots so hider/seeker strategies can be tracked and compared.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_DIR = os.path.dirname(__file__)
LOGS_ROOT = os.path.join(PROJECT_DIR, "logs")
RUNS_ROOT = os.path.join(LOGS_ROOT, "runs")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_DIR, "charts", "monitoring")


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def moving_average(xs: Sequence[int], ys: Sequence[float], window: int) -> Tuple[List[int], List[float]]:
    if window <= 1 or len(ys) < window:
        return list(xs), [float(v) for v in ys]
    arr = np.array(ys, dtype=np.float32)
    weights = np.ones(window, dtype=np.float32) / float(window)
    smoothed = np.convolve(arr, weights, mode="valid")
    trimmed_x = list(xs)[window - 1 :]
    return trimmed_x, smoothed.astype(np.float32).tolist()


def cumulative_rate(flags: Sequence[int]) -> List[float]:
    rates: List[float] = []
    total = 0.0
    for idx, flag in enumerate(flags, start=1):
        total += flag
        rates.append(total / float(idx))
    return rates


@dataclass
class RunMetrics:
    approach: str
    run_id: str
    path: str
    episodes: List[int] = field(default_factory=list)
    updates: List[int] = field(default_factory=list)
    reward_mean: List[float] = field(default_factory=list)
    reward_sum: List[float] = field(default_factory=list)
    seeker_reward_sum: List[float] = field(default_factory=list)
    seeker_reward_mean: List[float] = field(default_factory=list)
    seeker_steps: List[int] = field(default_factory=list)
    seeker_avg_distance: List[float] = field(default_factory=list)
    hider_reward_sum: List[float] = field(default_factory=list)
    hider_reward_mean: List[float] = field(default_factory=list)
    hider_steps: List[int] = field(default_factory=list)
    hider_avg_distance: List[float] = field(default_factory=list)
    winners: List[str] = field(default_factory=list)
    terminal_reason: List[str] = field(default_factory=list)
    duration_sec: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def load_run_from_csv(approach: str, run_id: str, csv_path: str, base_dir: str) -> Optional[RunMetrics]:
    if not os.path.exists(csv_path):
        return None
    metrics = RunMetrics(approach=approach, run_id=run_id, path=base_dir)
    with open(csv_path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            metrics.episodes.append(parse_int(row.get("episode")))
            metrics.updates.append(parse_int(row.get("updates")))
            metrics.reward_mean.append(parse_float(row.get("reward_mean")))
            metrics.reward_sum.append(parse_float(row.get("reward_sum")))
            metrics.seeker_reward_sum.append(parse_float(row.get("seeker_reward_sum")))
            metrics.seeker_reward_mean.append(parse_float(row.get("seeker_reward_mean")))
            metrics.seeker_steps.append(parse_int(row.get("seeker_steps")))
            metrics.seeker_avg_distance.append(parse_float(row.get("seeker_avg_distance")))
            metrics.hider_reward_sum.append(parse_float(row.get("hider_reward_sum")))
            metrics.hider_reward_mean.append(parse_float(row.get("hider_reward_mean")))
            metrics.hider_steps.append(parse_int(row.get("hider_steps")))
            metrics.hider_avg_distance.append(parse_float(row.get("hider_avg_distance")))
            winner = (row.get("winner") or "").strip().lower()
            metrics.winners.append(winner)
            terminal = (row.get("terminal_reason") or "").strip().lower()
            metrics.terminal_reason.append(terminal)
            metrics.duration_sec.append(parse_float(row.get("duration_sec")))
    meta_path = os.path.join(base_dir, "metadata.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as meta_fh:
                metrics.metadata = json.load(meta_fh)
        except (OSError, json.JSONDecodeError):
            metrics.metadata = {}
    return metrics


def load_run_metrics(approach: str, run_id: str) -> Optional[RunMetrics]:
    run_dir = os.path.join(RUNS_ROOT, approach, run_id)
    csv_path = os.path.join(run_dir, "metrics.csv")
    return load_run_from_csv(approach, run_id, csv_path, run_dir)


def load_legacy_run() -> Optional[RunMetrics]:
    csv_path = os.path.join(LOGS_ROOT, "metrics.csv")
    if not os.path.exists(csv_path):
        return None
    base_dir = LOGS_ROOT
    # Derive pseudo run id from mtime for reproducibility.
    stamp = parse_int(os.path.getmtime(csv_path))
    run_id = f"legacy_{stamp}"
    return load_run_from_csv("legacy", run_id, csv_path, base_dir)


def compute_run_summary(run: RunMetrics) -> Dict[str, Any]:
    episodes = len(run.episodes)
    seeker_flags = [1 if w == "seeker" else 0 for w in run.winners]
    seeker_rates = cumulative_rate(seeker_flags) if seeker_flags else []
    hider_rates = [1.0 - v for v in seeker_rates]
    seeker_final = seeker_rates[-1] if seeker_rates else None
    hider_final = hider_rates[-1] if hider_rates else None
    final_reward_mean = run.reward_mean[-1] if run.reward_mean else 0.0
    best_reward_mean = max(run.reward_mean) if run.reward_mean else 0.0
    outcomes = collections.Counter([reason or "unknown" for reason in run.terminal_reason])
    avg_duration = float(np.mean(run.duration_sec)) if run.duration_sec else None
    summary = {
        "approach": run.approach,
        "run_id": run.run_id,
        "episodes": episodes,
        "final_reward_mean": final_reward_mean,
        "best_reward_mean": best_reward_mean,
        "final_seeker_win_rate": seeker_final,
        "final_hider_win_rate": hider_final,
        "timeouts": outcomes.get("timeout", 0),
        "tags": outcomes.get("tag", 0),
        "unknown_finishes": outcomes.get("unknown", 0),
        "avg_duration_sec": avg_duration,
    }
    if run.metadata:
        summary["metadata"] = run.metadata
    return summary


def write_run_summary(run: RunMetrics, summary: Dict[str, Any]) -> None:
    summary_path = os.path.join(run.path, "summary.json")
    try:
        with open(summary_path, "w") as fh:
            json.dump(summary, fh, indent=2)
    except OSError:
        pass


def plot_single_series(
    x: Sequence[int],
    y: Sequence[float],
    title: str,
    ylabel: str,
    output_path: str,
    smoothing_window: int,
    label: str = "value",
) -> None:
    if not x or not y:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, y, label=label, alpha=0.35)
    xs_smooth, smoothed = moving_average(x, y, smoothing_window)
    if len(smoothed) > 0 and len(smoothed) != len(y):
        ax.plot(xs_smooth, smoothed, label=f"{smoothing_window}-episode avg")
    else:
        ax.plot(x, y, label=label)
    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_multi_series(
    x: Sequence[int],
    series: List[Tuple[Sequence[float], str]],
    title: str,
    ylabel: str,
    output_path: str,
    smoothing_window: int,
) -> None:
    if not x or not series:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    for values, label in series:
        if not values:
            continue
        xs_smooth, smoothed = moving_average(x, values, smoothing_window)
        if len(smoothed) > 0 and len(smoothed) != len(values):
            ax.plot(xs_smooth, smoothed, label=f"{label} ({smoothing_window}-avg)")
        else:
            ax.plot(x, values, label=label)
    if not ax.lines:
        plt.close(fig)
        return
    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_win_rates(run: RunMetrics, charts_dir: str, smoothing_window: int) -> None:
    if not run.episodes or not run.winners:
        return
    seeker_flags = [1 if w == "seeker" else 0 for w in run.winners]
    seeker_rates = cumulative_rate(seeker_flags)
    hider_rates = [1.0 - v for v in seeker_rates]
    output_path = os.path.join(charts_dir, "win_rates.png")
    plot_multi_series(
        run.episodes,
        [
            (seeker_rates, "Seeker win rate"),
            (hider_rates, "Hider win rate"),
        ],
        "Cumulative win rates",
        "Win rate",
        output_path,
        smoothing_window,
    )


def plot_role_rewards(run: RunMetrics, charts_dir: str, smoothing_window: int) -> None:
    if not run.episodes:
        return
    output_path = os.path.join(charts_dir, "role_rewards.png")
    plot_multi_series(
        run.episodes,
        [
            (run.seeker_reward_sum, "Seeker reward sum"),
            (run.hider_reward_sum, "Hider reward sum"),
        ],
        "Per-episode reward sums",
        "Reward",
        output_path,
        smoothing_window,
    )


def plot_avg_distances(run: RunMetrics, charts_dir: str, smoothing_window: int) -> None:
    if not run.episodes:
        return
    if not any(run.seeker_avg_distance) and not any(run.hider_avg_distance):
        return
    output_path = os.path.join(charts_dir, "average_distance.png")
    plot_multi_series(
        run.episodes,
        [
            (run.seeker_avg_distance, "Seeker distance"),
            (run.hider_avg_distance, "Hider distance"),
        ],
        "Average distance to opponent",
        "Distance (normalized units)",
        output_path,
        smoothing_window,
    )


def plot_episode_duration(run: RunMetrics, charts_dir: str, smoothing_window: int) -> None:
    if not run.episodes or not any(run.duration_sec):
        return
    output_path = os.path.join(charts_dir, "episode_duration.png")
    plot_single_series(
        run.episodes,
        run.duration_sec,
        "Episode duration",
        "Seconds",
        output_path,
        smoothing_window,
        label="duration",
    )


def render_per_run_charts(run: RunMetrics, smoothing_window: int, target_base: Optional[str] = None) -> None:
    base_dir = target_base or run.path
    charts_dir = os.path.join(base_dir, "charts")
    ensure_dir(charts_dir)
    plot_single_series(
        run.episodes,
        run.reward_mean,
        "Reward mean per episode",
        "Reward mean",
        os.path.join(charts_dir, "reward_mean.png"),
        smoothing_window,
        label="reward_mean",
    )
    plot_role_rewards(run, charts_dir, smoothing_window)
    plot_win_rates(run, charts_dir, smoothing_window)
    plot_avg_distances(run, charts_dir, smoothing_window)
    plot_episode_duration(run, charts_dir, smoothing_window)


def render_comparison_charts(runs: List[RunMetrics], output_dir: str, smoothing_window: int) -> None:
    if not runs:
        return
    ensure_dir(output_dir)
    by_approach: Dict[str, List[RunMetrics]] = collections.defaultdict(list)
    for run in runs:
        by_approach[run.approach].append(run)
    for approach, group in by_approach.items():
        if len(group) == 1:
            # Single run still benefits from copy in shared directory.
            run = group[0]
            reward_path = os.path.join(output_dir, f"{approach}_reward_mean.png")
            plot_single_series(
                run.episodes,
                run.reward_mean,
                f"{approach} reward mean",
                "Reward mean",
                reward_path,
                smoothing_window,
                label=run.run_id,
            )
            continue
        reward_path = os.path.join(output_dir, f"{approach}_reward_mean.png")
        series = [(run.reward_mean, run.run_id) for run in group]
        plot_multi_series(
            group[0].episodes,
            series,
            f"{approach} reward mean comparison",
            "Reward mean",
            reward_path,
            smoothing_window,
        )
        win_path = os.path.join(output_dir, f"{approach}_win_rate.png")
        win_series: List[Tuple[Sequence[float], str]] = []
        for run in group:
            if not run.winners:
                continue
            seeker_flags = [1 if w == "seeker" else 0 for w in run.winners]
            win_series.append((cumulative_rate(seeker_flags), f"{run.run_id} (seeker)"))
        if win_series:
            plot_multi_series(
                group[0].episodes,
                win_series,
                f"{approach} seeker win rate comparison",
                "Win rate",
                win_path,
                smoothing_window,
            )


def write_overview(summary_rows: List[Dict[str, Any]], output_dir: str) -> None:
    if not summary_rows:
        return
    ensure_dir(output_dir)
    csv_path = os.path.join(output_dir, "run_overview.csv")
    fieldnames = [
        "approach",
        "run_id",
        "episodes",
        "final_reward_mean",
        "best_reward_mean",
        "final_seeker_win_rate",
        "final_hider_win_rate",
        "tags",
        "timeouts",
        "unknown_finishes",
        "avg_duration_sec",
    ]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            compact = {k: row.get(k) for k in fieldnames}
            writer.writerow(compact)


def discover_runs(
    approaches: Iterable[str],
    run_filters: Iterable[str],
    include_all: bool,
) -> List[RunMetrics]:
    selected: List[RunMetrics] = []
    approach_filters = {a for a in approaches if a}
    run_filter_by_approach: Dict[str, set[str]] = collections.defaultdict(set)
    run_filter_plain: set[str] = set()
    for item in run_filters:
        if ":" in item:
            approach, run_id = item.split(":", 1)
            run_filter_by_approach[approach].add(run_id)
        elif item:
            run_filter_plain.add(item)

    if os.path.isdir(RUNS_ROOT):
        for approach in sorted(os.listdir(RUNS_ROOT)):
            if approach_filters and approach not in approach_filters:
                continue
            approach_dir = os.path.join(RUNS_ROOT, approach)
            if not os.path.isdir(approach_dir):
                continue
            run_ids = sorted(
                [name for name in os.listdir(approach_dir) if os.path.isdir(os.path.join(approach_dir, name))]
            )
            if not run_ids:
                continue
            if not include_all:
                run_ids = run_ids[-1:]
            for run_id in run_ids:
                if run_filter_plain and run_id not in run_filter_plain and run_id not in run_filter_by_approach.get(approach, set()):
                    continue
                if run_filter_by_approach and approach in run_filter_by_approach and run_id not in run_filter_by_approach[approach]:
                    continue
                metrics = load_run_metrics(approach, run_id)
                if metrics:
                    selected.append(metrics)
    # Legacy fallback
    if not selected:
        legacy = load_legacy_run()
        if legacy:
            selected.append(legacy)
    return selected


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate monitoring dashboards for training runs.")
    parser.add_argument(
        "--approach",
        action="append",
        default=[],
        help="Restrict to specific training approaches (can be specified multiple times).",
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        dest="runs",
        help="Specific run id(s) to include. Use approach:run_id to disambiguate.",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Process every run instead of only the latest per approach.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for aggregated comparison charts (defaults to trainer/charts/monitoring).",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=10,
        help="Window size for moving averages (default: 10 episodes).",
    )
    parser.add_argument(
        "--skip-per-run",
        action="store_true",
        help="Skip per-run dashboard generation and only build aggregate outputs.",
    )
    args = parser.parse_args(argv)

    runs = discover_runs(args.approach, args.runs, args.all_runs)
    if not runs:
        print("No training runs found. Start a run with scripts/train.sh to generate logs.", file=sys.stderr)
        return 1

    summary_rows: List[Dict[str, Any]] = []
    for run in runs:
        summary = compute_run_summary(run)
        summary_rows.append(summary)
        write_run_summary(run, summary)
        if not args.skip_per_run:
            render_per_run_charts(run, args.smooth_window)

    write_overview(summary_rows, args.output_dir)
    render_comparison_charts(runs, args.output_dir, args.smooth_window)

    print(f"Processed {len(runs)} run(s). Overviews stored in {args.output_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
