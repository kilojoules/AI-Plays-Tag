#!/bin/bash
#SBATCH --job-name=tag-ia-lay
#SBATCH --account=project_465002609
#SBATCH --partition=small
#SBATCH --array=0-23
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=experiments/results/init_alpha_layouts/logs/ia_%A_%a.out
#SBATCH --error=experiments/results/init_alpha_layouts/logs/ia_%A_%a.err

# Init-alpha replication on empty + central_cross
# 24 tasks: 4 init_alpha x 2 layouts x 3 seeds

mkdir -p experiments/results/init_alpha_layouts/logs
cd /scratch/project_465002609/julian/AI-Plays-Tag

echo "=== Task $SLURM_ARRAY_TASK_ID / 23 ==="
echo "Date: $(date)"

pixi run python experiments/init_alpha_layouts_tasks.py --task-id $SLURM_ARRAY_TASK_ID

echo "Done: $(date)"
