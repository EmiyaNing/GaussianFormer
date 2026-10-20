#!/usr/bin/env bash
# Independent ten-epoch run; restarts resume only this run's latest checkpoint.
set -euo pipefail
cd "$(dirname "$0")/.."

run_root=work_dirs/full_occ3d_10e_no_benchmark
run_dir="$run_root/opusv1_gaussian_softplus_learned_bias"
mkdir -p "$run_dir"
exec 9> "$run_root/run.lock"
flock -n 9 || { echo 'This ten-epoch job is already running.' >&2; exit 1; }
echo "$$" > "$run_root/launcher.pid"
status_file="$run_root/status.txt"
trap 'code=$?; if ((code != 0)); then echo "failed (exit $code); inspect console.log" > "$status_file"; fi' EXIT
echo 'training: 10 epochs, full Occ3D, cudnn.benchmark=False' > "$status_file"

python3.10 -u train.py \
    --py-config config/full_occ3d_10e/opusv1_gaussian_softplus_learned_bias_no_benchmark.py \
    --work-dir "$run_dir" --gradient-accumulation 4 --seed 928 \
    >> "$run_dir/console.log" 2>&1
echo 'training complete; aggregating results' > "$status_file"
python3.10 tools/summarize_full_occ3d_10e.py >> "$run_root/summary.log" 2>&1
echo 'complete: 10 evaluated epochs; best checkpoints and per-class results summarized' > "$status_file"
