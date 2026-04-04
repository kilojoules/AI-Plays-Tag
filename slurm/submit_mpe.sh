#!/bin/bash
#SBATCH --job-name=tag-mpe
#SBATCH --account=project_465002609
#SBATCH --partition=small
#SBATCH --array=0-11
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=experiments/results/mpe_tag/logs/mpe_%A_%a.out
#SBATCH --error=experiments/results/mpe_tag/logs/mpe_%A_%a.err

# MPE simple_tag generalization experiment
# 12 tasks: 4 conditions (baseline + 3 KL scales) x 3 seeds
# 500K steps each, ~30-60 min

mkdir -p experiments/results/mpe_tag/logs

cd /scratch/project_465002609/julian/AI-Plays-Tag

echo "=== Task $SLURM_ARRAY_TASK_ID / 11 ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo ""

pixi run python experiments/mpe_tasks.py --task-id $SLURM_ARRAY_TASK_ID

echo ""
echo "Done: $(date)"
