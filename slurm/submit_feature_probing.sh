#!/bin/bash
#SBATCH --job-name=tag-probe
#SBATCH --account=project_465002609
#SBATCH --partition=small
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=experiments/results/paper_final/logs/probe_%j.out
#SBATCH --error=experiments/results/paper_final/logs/probe_%j.err

cd /scratch/project_465002609/julian/AI-Plays-Tag

echo "=== Feature Probing ==="
echo "Date: $(date)"

pixi run python experiments/feature_probing.py

echo "Done: $(date)"
