#!/usr/bin/env bash
# Run the OPUSv1 P1 core matrix serially and leave a reproducible eval log per arm.
#
# The P1 gate is decided by A5 versus A1/A2/A3.  A4 is kept as the fixed-geometry
# diagnostic.  A6/A7 are intentionally absent: they are not implemented in this
# repository and this queue must not claim otherwise.
#
# Safe to re-run: a finished epoch_20 checkpoint skips training, but its final
# standalone eval is re-run so every queued arm has a fresh eval.log.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

jobs=(
  "a5 42 config/opus/ablations_opusv1/p1_a5_gaussian.py out/opusv1_p1_a5_s42_full"
  "a5 43 config/opus/ablations_opusv1/p1_a5_gaussian.py out/opusv1_p1_a5_s43"
  "a5 44 config/opus/ablations_opusv1/p1_a5_gaussian.py out/opusv1_p1_a5_s44"
  "a1 42 config/opus/ablations_opusv1/p1_a1_pointwise.py out/opusv1_p1_a1_s42"
  "a1 43 config/opus/ablations_opusv1/p1_a1_pointwise.py out/opusv1_p1_a1_s43"
  "a1 44 config/opus/ablations_opusv1/p1_a1_pointwise.py out/opusv1_p1_a1_s44"
  "a2 42 config/opus/ablations_opusv1/p1_a2_knn_mean.py out/opusv1_p1_a2_s42"
  "a2 43 config/opus/ablations_opusv1/p1_a2_knn_mean.py out/opusv1_p1_a2_s43"
  "a2 44 config/opus/ablations_opusv1/p1_a2_knn_mean.py out/opusv1_p1_a2_s44"
  "a3 42 config/opus/ablations_opusv1/p1_a3_graph_attention.py out/opusv1_p1_a3_s42"
  "a3 43 config/opus/ablations_opusv1/p1_a3_graph_attention.py out/opusv1_p1_a3_s43"
  "a3 44 config/opus/ablations_opusv1/p1_a3_graph_attention.py out/opusv1_p1_a3_s44"
  "a4 42 config/opus/ablations_opusv1/p1_a4_fixed_gaussian.py out/opusv1_p1_a4_s42"
  "a4 43 config/opus/ablations_opusv1/p1_a4_fixed_gaussian.py out/opusv1_p1_a4_s43"
  "a4 44 config/opus/ablations_opusv1/p1_a4_fixed_gaussian.py out/opusv1_p1_a4_s44"
)

for job in "${jobs[@]}"; do
  read -r variant seed config work_dir <<<"$job"
  checkpoint="$work_dir/epoch_20.pth"
  mkdir -p "$work_dir"
  printf '\n=== %s seed=%s started=%s ===\n' "$variant" "$seed" "$(date -Is)"

  if [[ ! -s "$checkpoint" ]]; then
    python3.10 train.py --py-config "$config" --work-dir "$work_dir" --seed "$seed" \
      >"$work_dir/train.log" 2>&1
  else
    printf 'reusing completed checkpoint: %s\n' "$checkpoint"
  fi

  if [[ ! -s "$checkpoint" ]]; then
    printf 'ERROR: expected checkpoint missing after training: %s\n' "$checkpoint" >&2
    exit 1
  fi

  python3.10 eval.py --py-config "$config" --resume-from "$checkpoint" \
    --work-dir "$work_dir/eval" --seed "$seed" >"$work_dir/eval.log" 2>&1
  # systemd user services use a deliberately minimal PATH; rely on POSIX grep
  # rather than the developer-only ``rg`` binary for this completion check.
  if ! grep -Eq "occ/mIoU|mIoU of" "$work_dir/eval.log"; then
    printf 'ERROR: no Occ3D mIoU in %s\n' "$work_dir/eval.log" >&2
    exit 1
  fi
  printf '=== %s seed=%s completed=%s ===\n' "$variant" "$seed" "$(date -Is)"
done
