#!/usr/bin/env bash
# Run the camera mask, learned-bias, and class-aware margin ablations serially.
set -euo pipefail
cd "$(dirname "$0")/.."

status_file=work_dirs/mini_occ3d_4096_1024/gaussian_followup_status.txt
mkdir -p work_dirs/mini_occ3d_4096_1024
echo 'starting' > "$status_file"
trap 'code=$?; if ((code != 0)); then echo "failed (exit $code)" > "$status_file"; fi' EXIT

# The first run may already be training in an interactive exec session.  The
# takeover file is written only after that process has been stopped at a saved
# epoch; train.py then resumes from latest.pth without duplicating GPU work.
takeover_file=work_dirs/mini_occ3d_4096_1024/gaussian_followup_takeover.signal
echo 'waiting for first-run takeover' > "$status_file"
for ((poll=0; poll<1440; poll++)); do
    if [[ -f "$takeover_file" ]]; then
        break
    fi
    sleep 30
done
if ((poll == 1440)); then
    echo 'Takeover signal was not received within 12 hours.' >&2
    exit 1
fi

for variant in \
    opusv1_gaussian_softplus_camera_mask \
    opusv1_gaussian_softplus_learned_bias \
    opusv1_gaussian_softplus_class_margin; do
    run_dir="work_dirs/mini_occ3d_4096_1024/$variant"
    mkdir -p "$run_dir"
    echo "training $variant" > "$status_file"
    python3.10 train.py \
        --py-config "config/mini_occ3d_4096_1024/$variant.py" \
        --work-dir "$run_dir" --gradient-accumulation 4 --seed 928 \
        > "$run_dir/console.log" 2>&1

    best_epoch="$(python3.10 -c 'import sys; from pathlib import Path; sys.path.insert(0,"tools"); from summarize_mini_occ3d import parse_log; p=Path(sys.argv[1]); rows=parse_log(sorted(x for x in p.glob("*.log") if x.name not in ("console.log","diagnostic_console.log"))); assert [r["epoch"] for r in rows] == [1,2,3,4,5]; print(max(rows,key=lambda r:(r["mIoU"],-r["epoch"]))["epoch"])' "$run_dir")"
    echo "diagnosing $variant epoch $best_epoch" > "$status_file"
    python3.10 tools/diagnose_opus_v1_gaussian.py \
        --split val --limit 1024 \
        --py-config "config/mini_occ3d_4096_1024/$variant.py" \
        --checkpoint "$run_dir/epoch_$best_epoch.pth" \
        --out "$run_dir/diagnostic_val1024.json" \
        > "$run_dir/diagnostic_console.log" 2>&1
    python3.10 tools/summarize_gaussian_followup.py --allow-partial
done
python3.10 tools/summarize_gaussian_followup.py
echo 'complete' > "$status_file"
