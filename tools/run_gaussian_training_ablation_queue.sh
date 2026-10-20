#!/usr/bin/env bash
# Resume/execute the three controlled ablations serially, then write the report.
set -euo pipefail
cd "$(dirname "$0")/.."

status_file=work_dirs/mini_occ3d_4096_1024/gaussian_training_ablation_status.txt
echo 'starting ablations' > "$status_file"
trap 'code=$?; if ((code != 0)); then echo "failed (exit $code)" > "$status_file"; fi' EXIT

for variant in opusv1_gaussian_margin opusv1_gaussian_softplus opusv1_gaussian_margin_softplus; do
    echo "training $variant" > "$status_file"
    run_dir="work_dirs/mini_occ3d_4096_1024/$variant"
    mkdir -p "$run_dir"
    python3.10 train.py \
        --py-config "config/mini_occ3d_4096_1024/$variant.py" \
        --work-dir "$run_dir" --gradient-accumulation 4 --seed 928 \
        > "$run_dir/console.log" 2>&1
done

for variant in opusv1_gaussian_margin opusv1_gaussian_softplus opusv1_gaussian_margin_softplus; do
    echo "diagnosing $variant" > "$status_file"
    run_dir="work_dirs/mini_occ3d_4096_1024/$variant"
    best_epoch="$(python3.10 -c 'import sys; from pathlib import Path; sys.path.insert(0,"tools"); from summarize_mini_occ3d import parse_log; p=Path(sys.argv[1]); rows=parse_log(sorted(p.glob("*.log"))); print(max(rows,key=lambda x:(x["mIoU"],-x["epoch"]))["epoch"])' "$run_dir")"
    python3.10 tools/diagnose_opus_v1_gaussian.py \
        --split val --limit 1024 \
        --py-config "config/mini_occ3d_4096_1024/$variant.py" \
        --checkpoint "$run_dir/epoch_$best_epoch.pth" \
        --out "$run_dir/diagnostic_val1024.json" \
        > "$run_dir/diagnostic_console.log" 2>&1
done

python3.10 tools/summarize_gaussian_training_ablation.py
echo 'complete' > "$status_file"
