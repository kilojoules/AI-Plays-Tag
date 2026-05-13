#!/bin/bash
#SBATCH --job-name=dc_anchors
#SBATCH --account=project_465002609
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-3%3
#SBATCH --output=logs/dc_anchors_%A_%a.out
#SBATCH --error=logs/dc_anchors_%A_%a.err

# Design C anchor policies — 4 corners at seed=42.
# Independent of main grid; can run in parallel with it.

set -euo pipefail

cd /scratch/project_465002609/julian/AI-Plays-Tag

echo "=== Design C anchor task $SLURM_ARRAY_TASK_ID / 3 ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Job: $SLURM_JOB_ID  Array: $SLURM_ARRAY_JOB_ID[$SLURM_ARRAY_TASK_ID]"
echo "Git HEAD: $(git rev-parse --short HEAD)"
echo ""

mkdir -p logs

pixi run python experiments/design_c_anchors_tasks.py --task-id $SLURM_ARRAY_TASK_ID

echo ""
echo "Done: $(date)"
