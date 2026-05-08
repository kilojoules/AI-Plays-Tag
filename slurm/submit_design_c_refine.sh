#!/bin/bash
#SBATCH --job-name=dc_refine
#SBATCH --account=project_465002609
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/dc_refine_%A_%a.out
#SBATCH --error=logs/dc_refine_%A_%a.err

# Design C REFINE round.
# Adds seeds 3, 4 to the main grid (8 new runs) per pre-reg v2 §A6.
# Once all 8 finish, re-run the gauntlet and analyze; the gauntlet now
# auto-discovers seeds, so it picks up the new policies without code edits.

set -euo pipefail

cd /scratch/project_465002609/julian/AI-Plays-Tag

echo "=== Design C REFINE task $SLURM_ARRAY_TASK_ID / 7 ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Job: $SLURM_JOB_ID  Array: $SLURM_ARRAY_JOB_ID[$SLURM_ARRAY_TASK_ID]"
echo "Git HEAD: $(git rev-parse --short HEAD)"
echo ""

mkdir -p logs

pixi run python experiments/design_c_refine_tasks.py --task-id $SLURM_ARRAY_TASK_ID

echo ""
echo "Done: $(date)"
