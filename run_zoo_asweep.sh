#!/bin/bash
#SBATCH --partition=windq
#SBATCH --job-name="zoo_asweep"
#SBATCH --time=4-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --array=0-839%32
#SBATCH --output=experiments/results/zoo_asweep/logs/task_%a.out
#SBATCH --error=experiments/results/zoo_asweep/logs/task_%a.err
#
# Zoo A-sweep: 840 tasks (20 configs × 7 A × 2 sampling × 3 seeds)
# --array=0-839%32  →  up to 32 tasks run simultaneously
#
# Prerequisites (not yet implemented in train_zoo.py):
#   1. --seeker-time-penalty  (port from train_selfplay.py)
#   2. --sampling-strategy thompson_loss  (port from RPS_RL/zoo.py + invert)
#
# Adjust %32 throttle to match node availability. At 32 concurrent
# with ~3 days each, full sweep completes in ~80 days wall time.
# Bump to %64 or %128 if more nodes are available.
#
# Quick sanity check before submitting:
#   pixi run python experiments/zoo_asweep_tasks.py --list | head -20
#   pixi run python experiments/zoo_asweep_tasks.py --task-id 0 --dry-run

. ~/.bashrc

mkdir -p experiments/results/zoo_asweep/logs

pixi run python experiments/zoo_asweep_tasks.py --task-id $SLURM_ARRAY_TASK_ID
