#!/bin/bash
#SBATCH --partition=windq
#SBATCH --job-name="zoo_gauntlet"
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --array=0-279%32
#SBATCH --output=experiments/results/zoo_asweep/gauntlet/logs/task_%a.out
#SBATCH --error=experiments/results/zoo_asweep/gauntlet/logs/task_%a.err
#
# Zoo A-sweep gauntlet: 280 tasks (20 configs × 7 A × 2 sampling, seed 0 only)
# Each task: ~13 checkpoints → 169 matchups × 20 episodes ≈ 10-30 min
#
# After all tasks complete, aggregate:
#   pixi run python experiments/zoo_asweep_gauntlet.py --aggregate

. ~/.bashrc

mkdir -p experiments/results/zoo_asweep/gauntlet/logs

pixi run python experiments/zoo_asweep_gauntlet.py --task-id $SLURM_ARRAY_TASK_ID
