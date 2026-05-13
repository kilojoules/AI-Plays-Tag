#!/bin/bash
#SBATCH --job-name=dc_mcmc
#SBATCH --account=project_465002609
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/dc_mcmc_%j.out
#SBATCH --error=logs/dc_mcmc_%j.err

# Design C — proper Bayesian arm via PyMC NUTS.
# Replaces the variational-Bayes shortcut whose CrI was too tight.

set -euo pipefail

cd /scratch/project_465002609/julian/AI-Plays-Tag

echo "=== Design C MCMC analysis ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Job: $SLURM_JOB_ID"
echo "Git HEAD: $(git rev-parse --short HEAD)"
echo ""

mkdir -p logs

pixi run -e train python experiments/design_c_analyze_mcmc.py --draws 1500 --chains 4

echo ""
echo "Done: $(date)"
