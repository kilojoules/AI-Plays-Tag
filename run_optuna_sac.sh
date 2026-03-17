#!/bin/bash
#SBATCH --job-name=optuna_sac
#SBATCH --partition=windq,workq
#SBATCH --array=0-19
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=experiments/results/optuna/logs/sac_%A_%a.out

mkdir -p experiments/results/optuna/logs

cd /work/users/juqu/AI-Plays-Tag

pixi run python experiments/optuna_sac.py \
    --n-trials 5 \
    --study-name sac_hpo_v3 \
    --storage-dir experiments/results/optuna
