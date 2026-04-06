#!/bin/bash
#SBATCH --job-name=tag-sumo
#SBATCH --account=project_465002609
#SBATCH --partition=small
#SBATCH --array=0-17
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=experiments/results/ant_sumo/logs/sumo_%A_%a.out
#SBATCH --error=experiments/results/ant_sumo/logs/sumo_%A_%a.err

# Ant Sumo generalization: 18 tasks
# 6 entropy conditions x 3 seeds, 500K steps each
# MuJoCo is slower than our NumPy env — allow 8h walltime

mkdir -p experiments/results/ant_sumo/logs

cd /scratch/project_465002609/julian/AI-Plays-Tag

echo "=== Task $SLURM_ARRAY_TASK_ID / 17 ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo ""

pixi run python experiments/ant_sumo_tasks.py --task-id $SLURM_ARRAY_TASK_ID

echo ""
echo "Done: $(date)"
