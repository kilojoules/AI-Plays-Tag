#!/bin/bash
#SBATCH --job-name=tag-fin-gaunt
#SBATCH --account=project_465002609
#SBATCH --partition=small
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=experiments/results/paper_final/logs/gauntlet_%j.out
#SBATCH --error=experiments/results/paper_final/logs/gauntlet_%j.err

cd /scratch/project_465002609/julian/AI-Plays-Tag

echo "=== Final Paper Gauntlet ==="
echo "Date: $(date)"
echo ""

pixi run python experiments/paper_final_gauntlet.py

echo ""
echo "Done: $(date)"
