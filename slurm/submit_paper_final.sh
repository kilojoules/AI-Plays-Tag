#!/bin/bash
#SBATCH --job-name=tag-final
#SBATCH --account=project_465002609
#SBATCH --partition=small
#SBATCH --array=0-38
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=experiments/results/paper_final/logs/final_%A_%a.out
#SBATCH --error=experiments/results/paper_final/logs/final_%A_%a.err

# Final paper experiments:
#   Tasks 0-26: Geometry study (9 layouts x 3 seeds)
#   Tasks 27-38: Init-alpha ablation (4 values x 3 seeds)
# All runs also log buffer diversity metrics (Experiment 2)

mkdir -p experiments/results/paper_final/logs

cd /scratch/project_465002609/julian/AI-Plays-Tag

echo "=== Task $SLURM_ARRAY_TASK_ID / 38 ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo ""

pixi run python experiments/paper_final_tasks.py --task-id $SLURM_ARRAY_TASK_ID

echo ""
echo "Done: $(date)"
