#!/usr/bin/env bash
# Long-running full Occ3D job; train.py resumes from latest.pth if restarted.
set -euo pipefail
cd "$(dirname "$0")/.."

run_dir=work_dirs/full_occ3d_5e/opusv1_gaussian_softplus_learned_bias
status_file=work_dirs/full_occ3d_5e/status.txt
mkdir -p "$run_dir"
echo 'training opusv1_gaussian_softplus_learned_bias' > "$status_file"
trap 'code=$?; if ((code != 0)); then echo "failed (exit $code)" > "$status_file"; fi' EXIT

python3.10 train.py \
    --py-config config/full_occ3d_5e/opusv1_gaussian_softplus_learned_bias.py \
    --work-dir "$run_dir" --gradient-accumulation 4 --seed 928 \
    > "$run_dir/console.log" 2>&1
echo 'training complete; awaiting result aggregation' > "$status_file"
