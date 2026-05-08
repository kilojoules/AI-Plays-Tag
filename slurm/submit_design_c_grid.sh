#!/bin/bash
#SBATCH --job-name=dc_grid
#SBATCH --account=project_465002609
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-11
#SBATCH --output=logs/dc_grid_%A_%a.out
#SBATCH --error=logs/dc_grid_%A_%a.err

# Design C main grid (PPO-only, 12 runs).
# See experiments/design_c_preregistration_v2.md
#
# 2 rewards (R4, R7) × 2 A (0.0, 0.5) × 3 seeds (0, 1, 2) = 12 tasks
# Each task: 5M PPO steps, ~1h wall on 8 CPUs (PPO@R4 pilot did 10M in ~2h).
# 4h walltime cap = generous safety margin.

set -euo pipefail

cd /scratch/project_465002609/julian/AI-Plays-Tag

echo "=== Design C grid task $SLURM_ARRAY_TASK_ID / 11 ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Job: $SLURM_JOB_ID  Array: $SLURM_ARRAY_JOB_ID[$SLURM_ARRAY_TASK_ID]"
echo "Git HEAD: $(git rev-parse --short HEAD)"
echo ""

mkdir -p logs

pixi run python experiments/design_c_grid_tasks.py --task-id $SLURM_ARRAY_TASK_ID

echo ""
echo "Done: $(date)"
