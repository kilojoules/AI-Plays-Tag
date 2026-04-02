#!/bin/bash
#
# Submit all AAMAS paper experiments to LUMI.
#
# Usage (from LUMI):
#   cd /scratch/project_465002609/julian/AI-Plays-Tag
#   bash slurm/submit_all.sh [--dry-run]
#
# Or from local machine:
#   git push && ssh quickjul@lumi.csc.fi \
#     "cd /scratch/project_465002609/julian/AI-Plays-Tag && git pull && bash slurm/submit_all.sh"
#
set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
fi

cd "$(dirname "$0")/.."
mkdir -p experiments/results/paper_ablations/logs

echo "======================================"
echo "AAMAS Paper Experiments — LUMI"
echo "======================================"
echo ""

echo "--- Experiment 1A: Alpha counterfactual (9 tasks) ---"
pixi run python experiments/paper_ablation_tasks.py --list --experiment 1A
echo ""

echo "--- Experiment 1B: Alpha dynamics (24 tasks) ---"
pixi run python experiments/paper_ablation_tasks.py --list --experiment 1B
echo ""

echo "Total: 33 tasks @ ~1h each on small partition"
echo ""

if [[ $DRY_RUN -eq 1 ]]; then
    echo "[DRY RUN] Would submit:"
    echo "  sbatch slurm/submit_1A_counterfactual.sh"
    echo "  sbatch slurm/submit_1B_alpha_dynamics.sh"
    exit 0
fi

JOB_1A=$(sbatch --parsable slurm/submit_1A_counterfactual.sh)
echo "Submitted 1A: job $JOB_1A (array 0-8)"

JOB_1B=$(sbatch --parsable slurm/submit_1B_alpha_dynamics.sh)
echo "Submitted 1B: job $JOB_1B (array 9-32)"

echo ""
echo "Monitor:"
echo "  squeue -u quickjul"
echo "  tail -f experiments/results/paper_ablations/logs/1A_${JOB_1A}_*.out"
