#!/bin/bash
#SBATCH --job-name=dc_pilot_sac_r7
#SBATCH --account=project_465002609
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/dc_pilot_sac_r7_%j.out
#SBATCH --error=logs/dc_pilot_sac_r7_%j.err

# Design C pilot #2: SAC at R7_kitchen_sink, A=0, seed=0, 10M steps.
# First pilot (R4_sparse) showed SAC collapse: 55% -> 14% WR, alpha->0,
# actor loss diverged. R7 has dense shaping; if SAC stabilises here, the
# instability is a reward × algorithm interaction we want to study (not
# an HP defect).

set -euo pipefail
cd /scratch/project_465002609/julian/AI-Plays-Tag

echo "=== Design C SAC pilot @ R7_kitchen_sink ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Job: $SLURM_JOB_ID"
echo "Git HEAD: $(git rev-parse --short HEAD)"
echo ""

OUT=experiments/results/design_c/pilot/sac_R7_A0_seed0
mkdir -p "$OUT" logs

pixi run python trainer/train_zoo_sac.py \
    --timesteps 10000000 \
    --seed 0 \
    --layout four_corners \
    --hider-speed-mult 1.15 \
    --zoo-prob 0.0 \
    --zoo-interval 50 \
    --zoo-max-size 50 \
    --seeker-time-penalty -0.015 \
    --seeker-escalating-urgency \
    --distance-reward-scale 0.2 \
    --hider-dist-reward 0.14 \
    --hider-abs-dist-reward 0.1 \
    --hider-wall-prox-penalty -0.02 \
    --hider-min-speed-reward 0.005 \
    --area-coverage-bonus 0.05 \
    --runner-survival-bonus 0.01 \
    --warmup-steps 10000 \
    --log-interval 5000 \
    --save-interval 100000 \
    --output-dir "$OUT"

echo ""
echo "Done: $(date)"
