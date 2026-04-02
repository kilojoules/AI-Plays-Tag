#!/bin/bash
#SBATCH --job-name=tag-1A
#SBATCH --account=project_465002609
#SBATCH --partition=small
#SBATCH --array=0-8
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=experiments/results/paper_ablations/logs/1A_%A_%a.out
#SBATCH --error=experiments/results/paper_ablations/logs/1A_%A_%a.err

# Experiment 1A: SAC alpha=0 counterfactual
# 9 tasks: 3 alpha conditions (0.0, 0.1, auto) x 3 seeds on R4_sparse
# ~40-60 min each at 5M steps with 64 envs

mkdir -p experiments/results/paper_ablations/logs

cd /scratch/project_465002609/julian/AI-Plays-Tag

echo "=== Task $SLURM_ARRAY_TASK_ID / 8 ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo ""

pixi run python experiments/paper_ablation_tasks.py --task-id $SLURM_ARRAY_TASK_ID

echo ""
echo "Done: $(date)"
