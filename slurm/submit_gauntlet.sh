#!/bin/bash
#SBATCH --job-name=tag-gauntlet
#SBATCH --account=project_465002609
#SBATCH --partition=small
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=experiments/results/paper_ablations/logs/gauntlet_%j.out
#SBATCH --error=experiments/results/paper_ablations/logs/gauntlet_%j.err

# Cross-evaluation gauntlet for paper ablations
# 11 configs x 11 configs x 50 episodes = 6,050 episodes
# ~30-60 min on CPU

cd /scratch/project_465002609/julian/AI-Plays-Tag

echo "=== Gauntlet ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo ""

pixi run python experiments/paper_ablation_gauntlet.py

echo ""
echo "Done: $(date)"
