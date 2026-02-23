#!/bin/bash
#SBATCH --partition=windq
#SBATCH --job-name="sp_sweep"
#SBATCH --time=4-00:00:00
#SBATCH --ntasks-per-core 1
#SBATCH --ntasks-per-node 32
#SBATCH --nodes=1
#SBATCH --exclusive

. ~/.bashrc

pixi run python experiments/run_selfplay_sweep.py --max-parallel 16
