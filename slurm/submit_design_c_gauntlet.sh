#!/bin/bash
#SBATCH --job-name=dc_gauntlet
#SBATCH --account=project_465002609
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/dc_gauntlet_%j.out
#SBATCH --error=logs/dc_gauntlet_%j.err

# Design C cross-evaluation gauntlet.
# Run only after grid + anchors finish (16 policies expected in pool).
# 3-stage successive halving on episodes; should complete in well under 1h.

set -euo pipefail

cd /scratch/project_465002609/julian/AI-Plays-Tag

echo "=== Design C gauntlet ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Job: $SLURM_JOB_ID"
echo "Git HEAD: $(git rev-parse --short HEAD)"
echo ""

mkdir -p logs

pixi run python experiments/design_c_gauntlet.py

echo ""
echo "Done: $(date)"
