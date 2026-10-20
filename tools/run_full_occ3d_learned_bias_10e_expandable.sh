#!/usr/bin/env bash
# Uses the original run directory and lock; an existing live job is never killed.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_ALLOC_CONF=expandable_segments:True
unset PYTORCH_CUDA_ALLOC_CONF
exec bash tools/run_full_occ3d_learned_bias_10e.sh
