#!/bin/bash
#SBATCH --job-name=dc_pilot_sac
#SBATCH --account=project_465002609
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/dc_pilot_sac_%j.out
#SBATCH --error=logs/dc_pilot_sac_%j.err

# Design C pre-flight pilot (see experiments/design_c_preregistration.md §5).
#
# Single SAC run, 10M steps, R4_sparse reward, A=0 (zoo_prob=0), seed=0.
# Purpose: verify SAC reaches a learning plateau within the 5M-step budget
# planned for the main grid. PASS / EXTEND / HP-BROKEN gate documented in
# the pre-registration.
#
# Partition: `small` (CPU). pixi.toml ships standard PyTorch from
# conda-forge — no ROCm — so an MI250X GCD on `small-g` would not be used
# anyway. Existing tag slurm scripts (submit_gauntlet.sh etc.) use `small`.
# Wall-clock estimate: 8-16h for 10M SAC steps on 8 CPUs.

set -euo pipefail

cd /scratch/project_465002609/julian/AI-Plays-Tag

echo "=== Design C SAC pilot ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Job: $SLURM_JOB_ID"
echo "Git HEAD: $(git rev-parse --short HEAD)"
echo ""

OUT=experiments/results/design_c/pilot/sac_R4_A0_seed0
mkdir -p "$OUT" logs

# R4_sparse: only terminal rewards (no distance shaping, no survival/time).
# All reward CLI flags explicit so the run is fully reproducible from this
# sbatch alone — no dependency on reward_presets.py drifting later.
pixi run python trainer/train_zoo_sac.py \
    --timesteps 10000000 \
    --seed 0 \
    --layout four_corners \
    --hider-speed-mult 1.15 \
    --zoo-prob 0.0 \
    --zoo-interval 50 \
    --zoo-max-size 50 \
    --seeker-time-penalty 0.0 \
    --distance-reward-scale 0.0 \
    --hider-dist-reward 0.0 \
    --hider-abs-dist-reward 0.0 \
    --runner-survival-bonus 0.0 \
    --warmup-steps 10000 \
    --log-interval 5000 \
    --save-interval 100000 \
    --output-dir "$OUT"

echo ""
echo "Done: $(date)"
