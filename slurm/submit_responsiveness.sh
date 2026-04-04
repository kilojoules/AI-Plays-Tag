#!/bin/bash
#SBATCH --job-name=tag-resp
#SBATCH --account=project_465002609
#SBATCH --partition=small
#SBATCH --array=0-32
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=experiments/results/responsiveness_sweep/logs/resp_%A_%a.out
#SBATCH --error=experiments/results/responsiveness_sweep/logs/resp_%A_%a.err

# Responsiveness intrinsic reward sweep
# 33 tasks: 11 conditions x 3 seeds, SAC selfplay, 5M steps each
# Tests TE, KL, and both as replacements for hand-crafted anti-degenerate shaping

mkdir -p experiments/results/responsiveness_sweep/logs

cd /scratch/project_465002609/julian/AI-Plays-Tag

echo "=== Task $SLURM_ARRAY_TASK_ID / 32 ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo ""

pixi run python experiments/responsiveness_tasks.py --task-id $SLURM_ARRAY_TASK_ID

echo ""
echo "Done: $(date)"
