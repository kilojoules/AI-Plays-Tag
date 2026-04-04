#!/bin/bash
#SBATCH --job-name=tag-resp-gauntlet
#SBATCH --account=project_465002609
#SBATCH --partition=small
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=experiments/results/responsiveness_sweep/logs/gauntlet_%j.out
#SBATCH --error=experiments/results/responsiveness_sweep/logs/gauntlet_%j.err

cd /scratch/project_465002609/julian/AI-Plays-Tag

echo "=== Responsiveness Gauntlet ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo ""

pixi run python experiments/responsiveness_gauntlet.py

echo ""
echo "Done: $(date)"
