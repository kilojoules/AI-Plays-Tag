#!/bin/bash
#SBATCH --partition=windfatq
#SBATCH --job-name="tag"
#SBATCH --time=2-00:00:00
#SBATCH --ntasks-per-core 1
#SBATCH --ntasks-per-node 32
#SBATCH --nodes=1
#SBATCH --exclusive

. ~/.bashrc
#conEnv
#eval "$(pixi shell-hook)"

pixi run python experiments/run_zoo_sweep.py --max-parallel 10 --resume
