#!/bin/bash
#SBATCH --job-name=fr_sweep_v2
#SBATCH --partition=windq,workq
#SBATCH --array=0-149%28
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=experiments/results/fr_sweep_v2/logs/slurm_%A_%a.out

mkdir -p experiments/results/fr_sweep_v2/logs

cd /work/users/juqu/AI-Plays-Tag

pixi run python experiments/fr_sweep_tasks.py --task-id $SLURM_ARRAY_TASK_ID
