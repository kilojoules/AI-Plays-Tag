#!/bin/bash
# One-shot gbar setup for the /tmp job pattern:
# pack the pixi env into a single tarball in home, verify it works when
# relocated to /tmp (jobs unpack to per-job /tmp dirs), purge the
# unpacked env + pixi cache from home, then submit both LSF arrays.
set -euo pipefail
cd "$HOME/AI-Plays-Tag"

echo "=== leftover dc jobs ==="
bjobs -w 2>/dev/null | grep -E "dc_urg|dc_hsm" || echo "none"

echo "=== clean partial outputs from killed home-run tasks ==="
rm -rf experiments/results logs
mkdir -p logs

echo "=== direct-python smoke (no pixi run) ==="
./.pixi/envs/default/bin/python -c 'import torch; print("torch", torch.__version__)'

echo "=== packing env ($(du -sh .pixi | cut -f1)) ==="
tar -czf pixi_env.tar.gz .pixi
ls -lh pixi_env.tar.gz

echo "=== relocation test on /tmp ==="
RT="/tmp/${USER}_reloc_test"
rm -rf "$RT"; mkdir -p "$RT"
tar -C "$RT" -xzf pixi_env.tar.gz
"$RT/.pixi/envs/default/bin/python" -c 'import torch, numpy, pandas; print("reloc OK", torch.__version__)'
rm -rf "$RT"

echo "=== purge unpacked env + pixi cache from home ==="
rm -rf .pixi "$HOME/.cache/rattler"

echo "=== submit ==="
bsub < lsf/submit_design_c_urgency_only.lsf
bsub < lsf/submit_design_c_hsm_flank.lsf
sleep 3
bjobs -w 2>/dev/null | head -10

echo "=== home usage after ==="
du -sh "$HOME/AI-Plays-Tag"
getquota_zhome.sh 2>/dev/null | head -2 || true
echo "SETUP DONE"
