#!/usr/bin/env bash
# v0.1 exploratory series. Per-run rows go to results/v01/*.csv (gitignored), summaries to stdout.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
OUT=results/v01
T=ticks=10000

echo "### A: noise sweep (main hypothesis)"
$PY -m emergent_codes sweep noise 0,0.01,0.02,0.04,0.06,0.08,0.1,0.15 --seeds 30 --set $T --out $OUT/A_noise.csv

echo "### B: merge_radius x decay at noise=0.05"
for decay in 0.01 0.03 0.1; do
  echo "# decay=$decay"
  $PY -m emergent_codes sweep merge_radius 1,2,3,4 --seeds 10 --set $T decay=$decay --out $OUT/B_theta_decay$decay.csv
done

echo "### C: number of roles at n=12"
$PY -m emergent_codes sweep n_roles 1,2,4,8,16,32 --seeds 10 --set $T --out $OUT/C_roles.csv

echo "### D: control, n=7 R=4, seeded with Hamming [7,4] vs random"
for init in hamming74 random; do
  echo "# init=$init"
  $PY -m emergent_codes sweep noise 0,0.05,0.1 --seeds 10 --set $T n=7 init=$init --out $OUT/D_$init.csv
done

echo "### E: noise sweep at larger merge radius (is the breakdown tied to theta?)"
for theta in 3 4; do
  echo "# merge_radius=$theta"
  $PY -m emergent_codes sweep noise 0,0.02,0.04,0.06,0.08,0.1,0.15 --seeds 10 --set $T merge_radius=$theta --out $OUT/E_noise_theta$theta.csv
done
