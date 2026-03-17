#!/bin/bash
#SBATCH --job-name=optuna_ppo
#SBATCH --partition=windq,workq
#SBATCH --array=0-19
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=experiments/results/optuna/logs/ppo_%A_%a.out

mkdir -p experiments/results/optuna/logs

cd /work/users/juqu/AI-Plays-Tag

pixi run python experiments/optuna_ppo.py \
    --n-trials 5 \
    --study-name ppo_hpo_v5 \
    --storage-dir experiments/results/optuna
