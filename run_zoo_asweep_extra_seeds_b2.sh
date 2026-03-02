#!/bin/bash
#SBATCH --partition=windq
#SBATCH --job-name="zoo_asweep_s3-9b"
#SBATCH --time=4-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --array=0-979%32
#SBATCH --output=experiments/results/zoo_asweep/logs/extra_task_b2_%a.out
#SBATCH --error=experiments/results/zoo_asweep/logs/extra_task_b2_%a.err
#
# Batch 2 of extra seeds: tasks 980-1959 (offset by 980)

. ~/.bashrc

mkdir -p experiments/results/zoo_asweep/logs

REAL_TASK_ID=$((SLURM_ARRAY_TASK_ID + 980))
pixi run python experiments/zoo_asweep_extra_seeds_tasks.py --task-id $REAL_TASK_ID
