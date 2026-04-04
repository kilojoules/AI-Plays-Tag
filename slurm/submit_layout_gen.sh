#!/bin/bash
#SBATCH --job-name=tag-layout
#SBATCH --account=project_465002609
#SBATCH --partition=small
#SBATCH --array=0-17
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=experiments/results/layout_generalization/logs/layout_%A_%a.out
#SBATCH --error=experiments/results/layout_generalization/logs/layout_%A_%a.err

# Layout generalization: does KL responsiveness transfer across arena layouts?
# 18 tasks: 3 layouts x 2 conditions (baseline, kl_0.2) x 3 seeds

mkdir -p experiments/results/layout_generalization/logs

cd /scratch/project_465002609/julian/AI-Plays-Tag

echo "=== Task $SLURM_ARRAY_TASK_ID / 17 ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo ""

pixi run python experiments/layout_generalization_tasks.py --task-id $SLURM_ARRAY_TASK_ID

echo ""
echo "Done: $(date)"
