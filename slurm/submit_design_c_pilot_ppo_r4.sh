#!/bin/bash
#SBATCH --job-name=dc_pilot_ppo_r4
#SBATCH --account=project_465002609
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/dc_pilot_ppo_r4_%j.out
#SBATCH --error=logs/dc_pilot_ppo_r4_%j.err

# Design C pilot #3: PPO at R4_sparse, A=0, seed=0, 10M steps.
# Confirms PPO floor for the hardest reward cell. If PPO also collapses at
# R4_sparse, the issue is task-difficulty (R4 + HSM=1.15 unlearnable for
# anyone) and we have to widen our reward contrast or back off HSM.
# If PPO learns and SAC doesn't, the algorithm-class asymmetry is real.
#
# train_zoo.py uses --latest-prob (= 1 - A). A=0 means latest_prob=1.0.

set -euo pipefail
cd /scratch/project_465002609/julian/AI-Plays-Tag

echo "=== Design C PPO pilot @ R4_sparse ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Job: $SLURM_JOB_ID"
echo "Git HEAD: $(git rev-parse --short HEAD)"
echo ""

OUT=experiments/results/design_c/pilot/ppo_R4_A0_seed0
mkdir -p "$OUT" logs

pixi run python trainer/train_zoo.py \
    --timesteps 10000000 \
    --seed 0 \
    --layout four_corners \
    --hider-speed-mult 1.15 \
    --latest-prob 1.0 \
    --zoo-interval 50 \
    --zoo-max-size 50 \
    --sampling-strategy uniform \
    --seeker-time-penalty 0.0 \
    --distance-reward-scale 0.0 \
    --hider-dist-reward 0.0 \
    --hider-abs-dist-reward 0.0 \
    --runner-survival-bonus 0.0 \
    --batch-size 4096 \
    --num-envs 64 \
    --output-dir "$OUT"

echo ""
echo "Done: $(date)"
