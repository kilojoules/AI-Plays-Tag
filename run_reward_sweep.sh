#!/bin/bash
#SBATCH --job-name=reward_sweep
#SBATCH --partition=windq,workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --array=0-47%28
#SBATCH --output=experiments/results/reward_sweep/logs/slurm_%A_%a.out
#SBATCH --error=experiments/results/reward_sweep/logs/slurm_%A_%a.err

# Reward shaping sweep: 8 presets × 2 algorithms (PPO, SAC) × 3 seeds = 48 tasks

cd /work/users/juqu/AI-Plays-Tag

echo "=== Task $SLURM_ARRAY_TASK_ID / 47 ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo ""

pixi run python experiments/reward_sweep_tasks.py --task-id $SLURM_ARRAY_TASK_ID

echo ""
echo "Done: $(date)"
