#!/bin/bash
#SBATCH --job-name=tag-mechanism
#SBATCH --account=project_465002609
#SBATCH --partition=small
#SBATCH --array=0-4
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=experiments/results/paper_final/logs/mechanism_%A_%a.out
#SBATCH --error=experiments/results/paper_final/logs/mechanism_%A_%a.err

# Mechanism experiments:
#   Task 0: Q-value correlation analysis (all 4 init_alpha conditions)
#   Tasks 1-4: Critic transplant (4 conditions x 1M steps each)

cd /scratch/project_465002609/julian/AI-Plays-Tag

echo "=== Task $SLURM_ARRAY_TASK_ID ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo ""

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    echo "Running Q-value correlation analysis..."
    pixi run python experiments/qvalue_correlation.py
else
    TRANSPLANT_ID=$((SLURM_ARRAY_TASK_ID - 1))
    echo "Running critic transplant task $TRANSPLANT_ID..."
    pixi run python experiments/critic_transplant.py --task-id $TRANSPLANT_ID --timesteps 1000000 --seed 0
fi

echo ""
echo "Done: $(date)"
