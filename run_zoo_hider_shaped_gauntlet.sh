#!/bin/bash
#SBATCH --partition=windq
#SBATCH --job-name="hs_gauntlet"
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --array=0-599%64
#SBATCH --output=experiments/results/zoo_hider_shaped/gauntlet/logs/task_%a.out
#SBATCH --error=experiments/results/zoo_hider_shaped/gauntlet/logs/task_%a.err
#
# Zoo hider-shaped gauntlet: 600 tasks (20 configs × 5 A × 2 sampling × 3 seeds)
# Each task: ~13 checkpoints → 169 matchups × 20 episodes ≈ 2-3 min
#
# After all tasks complete, aggregate:
#   pixi run python experiments/zoo_hider_shaped_gauntlet.py --aggregate

. ~/.bashrc

mkdir -p experiments/results/zoo_hider_shaped/gauntlet/logs

pixi run python experiments/zoo_hider_shaped_gauntlet.py --task-id $SLURM_ARRAY_TASK_ID
