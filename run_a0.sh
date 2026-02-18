#!/bin/bash
#SBATCH --partition=windfatq
#SBATCH --job-name="zoo_A0"
#SBATCH --time=2-00:00:00
#SBATCH --ntasks-per-core 1
#SBATCH --ntasks-per-node 32
#SBATCH --nodes=1
#SBATCH --exclusive

. ~/.bashrc

OUTPUT_BASE="experiments/results/zoo_sweep"

# A00_hider_only: latest_prob=0.0, no seeker zoo
pixi run python trainer/train_zoo.py \
    --timesteps 10000000 \
    --latest-prob 0.0 \
    --output-dir "${OUTPUT_BASE}/A00_hider_only" \
    --layout four_corners &

# A00_both: latest_prob=0.0, with seeker zoo
pixi run python trainer/train_zoo.py \
    --timesteps 10000000 \
    --latest-prob 0.0 \
    --output-dir "${OUTPUT_BASE}/A00_both" \
    --layout four_corners \
    --use-seeker-zoo &

wait
echo "Both A=0 experiments finished."
