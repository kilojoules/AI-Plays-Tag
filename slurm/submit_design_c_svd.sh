#!/bin/bash
#SBATCH --job-name=dc_svd
#SBATCH --account=project_465002609
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/dc_svd_%j.out
#SBATCH --error=logs/dc_svd_%j.err

set -euo pipefail
cd /scratch/project_465002609/julian/AI-Plays-Tag

echo "=== Design C SVD ==="
echo "Date: $(date)"
echo "Git HEAD: $(git rev-parse --short HEAD)"
mkdir -p logs

pixi run -e train python experiments/design_c_svd.py

echo "Done: $(date)"
