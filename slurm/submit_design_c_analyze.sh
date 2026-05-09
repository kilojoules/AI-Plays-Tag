#!/bin/bash
#SBATCH --job-name=dc_analyze
#SBATCH --account=project_465002609
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/dc_analyze_%j.out
#SBATCH --error=logs/dc_analyze_%j.err

# Design C analysis: fits logit GLM + variational Bayes mixed model on the
# gauntlet matchup outcomes and prints the PURSUE/REFINE/KILL verdict.
# Triggered with --dependency=afterok:<gauntlet_job_id>.

set -euo pipefail

cd /scratch/project_465002609/julian/AI-Plays-Tag

echo "=== Design C analysis ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Job: $SLURM_JOB_ID"
echo "Git HEAD: $(git rev-parse --short HEAD)"
echo ""

mkdir -p logs

pixi run -e train python experiments/design_c_analyze.py

echo ""
echo "Done: $(date)"
