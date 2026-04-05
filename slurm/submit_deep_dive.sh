#!/bin/bash
#SBATCH --job-name=tag-deep
#SBATCH --account=project_465002609
#SBATCH --partition=small
#SBATCH --array=0-37
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=experiments/results/paper_final/logs/deep_%A_%a.out
#SBATCH --error=experiments/results/paper_final/logs/deep_%A_%a.err

# Mechanism deep dive: 38 tasks
#   0-19:  5-seed transplant replication (4 conditions x 5 seeds)
#   20:    Effective rank analysis (no training)
#   21-25: Random critic transplant (5 seeds)
#   26-34: Layer-wise transplant (3 conditions x 3 seeds)
#   35-37: Freeze actor (3 seeds)

cd /scratch/project_465002609/julian/AI-Plays-Tag

echo "=== Task $SLURM_ARRAY_TASK_ID / 37 ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo ""

pixi run python experiments/mechanism_deep_dive.py --task-id $SLURM_ARRAY_TASK_ID

echo ""
echo "Done: $(date)"
