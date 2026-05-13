#!/bin/bash
#SBATCH --job-name=dc_refine2
#SBATCH --account=project_465002609
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-7%3
#SBATCH --output=logs/dc_refine2_%A_%a.out
#SBATCH --error=logs/dc_refine2_%A_%a.err

# Design C REFINE round 2 — R7 only, seeds 5–8.
# Targets the dominant uncertainty (σ_seeker = 1.41 for R7) without
# wasting compute on R4 cells whose σ is already well-estimated.

set -euo pipefail
cd /scratch/project_465002609/julian/AI-Plays-Tag

echo "=== Design C REFINE2 task $SLURM_ARRAY_TASK_ID / 7 ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Job: $SLURM_JOB_ID  Array: $SLURM_ARRAY_JOB_ID[$SLURM_ARRAY_TASK_ID]"
echo "Git HEAD: $(git rev-parse --short HEAD)"

mkdir -p logs

pixi run python experiments/design_c_refine2_tasks.py --task-id $SLURM_ARRAY_TASK_ID

echo "Done: $(date)"
