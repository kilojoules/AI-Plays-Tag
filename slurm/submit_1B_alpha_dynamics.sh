#!/bin/bash
#SBATCH --job-name=tag-1B
#SBATCH --account=project_465002609
#SBATCH --partition=small
#SBATCH --array=9-32
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=experiments/results/paper_ablations/logs/1B_%A_%a.out
#SBATCH --error=experiments/results/paper_ablations/logs/1B_%A_%a.err

# Experiment 1B: Alpha dynamics across all 8 reward presets
# 24 tasks: 8 presets x 3 seeds, SAC selfplay with auto-tuned alpha
# ~40-60 min each at 5M steps with 64 envs

mkdir -p experiments/results/paper_ablations/logs

cd /scratch/project_465002609/julian/AI-Plays-Tag

echo "=== Task $SLURM_ARRAY_TASK_ID / 32 ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo ""

pixi run python experiments/paper_ablation_tasks.py --task-id $SLURM_ARRAY_TASK_ID

echo ""
echo "Done: $(date)"
