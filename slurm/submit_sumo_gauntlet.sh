#!/bin/bash
#SBATCH --job-name=sumo-gauntlet
#SBATCH --account=project_465002609
#SBATCH --partition=small
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=experiments/results/ant_sumo/logs/gauntlet_%j.out
#SBATCH --error=experiments/results/ant_sumo/logs/gauntlet_%j.err

cd /scratch/project_465002609/julian/AI-Plays-Tag

echo "=== Ant Sumo Gauntlet ==="
echo "Date: $(date)"
echo ""

pixi run python experiments/ant_sumo_gauntlet.py

echo ""
echo "Done: $(date)"
