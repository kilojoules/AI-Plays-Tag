#!/bin/bash
#SBATCH --job-name=dc_traj
#SBATCH --account=project_465002609
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/dc_traj_%j.out
#SBATCH --error=logs/dc_traj_%j.err

set -euo pipefail
cd /scratch/project_465002609/julian/AI-Plays-Tag

echo "=== Design C trajectory/behavior analysis ==="
echo "Date: $(date)"
echo "Git HEAD: $(git rev-parse --short HEAD)"
mkdir -p logs

pixi run -e train python experiments/design_c_trajectory_analysis.py --episodes 30

echo "Done: $(date)"
