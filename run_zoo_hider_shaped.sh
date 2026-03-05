#!/bin/bash
#SBATCH --partition=windq
#SBATCH --job-name="zoo_hshaped"
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --array=0-599%240
#SBATCH --output=experiments/results/zoo_hider_shaped/logs/task_%a.out
#SBATCH --error=experiments/results/zoo_hider_shaped/logs/task_%a.err
#
# Zoo A-sweep with hider distance-change shaping (reduced sweep):
#   600 tasks = 20 configs × 5 A × 2 sampling × 3 seeds
#   hider_dist_reward=0.14, hider_abs_dist_reward=0.1
#   %240 throttle for ~30 nodes (8 tasks/node × 30)
#
# Estimated: ~9 hours wall time at 240 concurrent

. ~/.bashrc

mkdir -p experiments/results/zoo_hider_shaped/logs

pixi run python experiments/zoo_hider_shaped_tasks.py --task-id $SLURM_ARRAY_TASK_ID
