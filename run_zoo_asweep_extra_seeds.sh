#!/bin/bash
#SBATCH --partition=windq
#SBATCH --job-name="zoo_asweep_s3-9"
#SBATCH --time=4-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --array=0-979%32
#SBATCH --output=experiments/results/zoo_asweep/logs/extra_task_%a.out
#SBATCH --error=experiments/results/zoo_asweep/logs/extra_task_%a.err
#
# Zoo A-sweep EXTRA SEEDS: 1960 tasks total, split into 2 batches of 980
# (MaxArraySize=1001 on this cluster)
# Batch 1: tasks 0-979    → run_zoo_asweep_extra_seeds.sh   (--array=0-979)
# Batch 2: tasks 980-1959 → run_zoo_asweep_extra_seeds_b2.sh (--array=0-979, offset +980)
#
# Submit both:
#   sbatch run_zoo_asweep_extra_seeds.sh
#   sbatch run_zoo_asweep_extra_seeds_b2.sh

. ~/.bashrc

mkdir -p experiments/results/zoo_asweep/logs

pixi run python experiments/zoo_asweep_extra_seeds_tasks.py --task-id $SLURM_ARRAY_TASK_ID
