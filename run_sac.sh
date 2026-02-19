#!/bin/bash
#SBATCH --partition=windq
#SBATCH --job-name="sac-tag"
#SBATCH --time=4-00:00:00
#SBATCH --ntasks-per-core 1
#SBATCH --ntasks-per-node 32
#SBATCH --nodes=1
#SBATCH --exclusive

. ~/.bashrc

pixi run python experiments/run_zoo_sweep.py --algorithm sac --max-parallel 10 --output-dir experiments/results/sac_sweep
