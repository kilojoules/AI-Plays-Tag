#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. && pwd)"

echo "[plot] Preparing metrics from trajectories (if needed) ..."
cd "$ROOT_DIR"
pixi run -e train python trainer/metrics_from_trajectories.py
echo "[plot] Generating charts from trainer/logs/metrics.csv ..."
pixi run -e train python trainer/plot_metrics.py
