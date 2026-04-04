#!/bin/bash
#SBATCH --job-name=tag-lay-gaunt
#SBATCH --account=project_465002609
#SBATCH --partition=small
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=experiments/results/layout_generalization/logs/gauntlet_%j.out
#SBATCH --error=experiments/results/layout_generalization/logs/gauntlet_%j.err

cd /scratch/project_465002609/julian/AI-Plays-Tag

echo "=== Layout Gauntlet ==="
echo "Date: $(date)"
echo ""

pixi run python experiments/layout_gauntlet.py

echo ""
echo "Done: $(date)"
