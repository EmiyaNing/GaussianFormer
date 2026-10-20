#!/usr/bin/env bash
# Fourth-stage GPU queue: re-measure the horizon-curriculum cost lever PAIRED.
#
# The 5.4x training-step ratio (124.7 ms at horizon 1 vs 669.1 ms at horizon 6)
# is the project's headline training-cost finding, but it came from two separate
# invocations 2m15s apart.  P4-6 showed this environment can report the same
# configuration 4.3x apart across groups, so the ratio must be measured with
# both epoch settings interleaved inside ONE process.  The probe now supports
# that via --epochs.
set -uo pipefail

cd "$(dirname "$0")/.."

echo "=== waiting for the degree sweep ==="
while pgrep -f "[r]un_p4_degree_sweep\.sh" > /dev/null; do sleep 60; done
while pgrep -f "[t]rain\.py --py-config" > /dev/null; do sleep 30; done
while pgrep -f "[e]val\.py" > /dev/null; do sleep 30; done
sleep 15
echo "=== GPU handed over at $(date -Is) ==="

# epoch 0 -> 1 horizon, epoch 8 -> 4 horizons: the model computes
#   max(1, min(curr_epoch - finetune_epoch + 1, num_fu_frames)) = min(8-5+1, 6) = 4
# in training mode.  The earlier comment said 6, which was wrong (P3-2 correction 2).
#
# ``--iters 40``, not 12.  The recorded per-step spread (p95/p50 = 1.43, i.e.
# log-sigma 0.22) gives a 95% margin of 13.2% at n = 12, while the density-cap
# effect this probe must resolve is only 12.7% -- at n = 12 the probe could not
# have settled the very quantity it exists to measure.  At n = 40 the margin is
# 7.0%, which is resolvable.  The reduced val split has 161 keyframes, so 40+3
# batches per (epoch, cap) combination fit.  NOTE: keep comment lines OUT of the
# continued command below -- a ``#`` inside a ``\``-continued command terminates
# it, silently splitting off the remaining arguments as a separate command.
python3.10 benchmarks/p3/profile_training.py \
  --config config/opus/ablations/p1_a5_gaussian.py \
  --checkpoint out/p1_a5_gaussian/epoch_40.pth \
  --epochs 0 8 --caps 0 32 --iters 40 \
  --json benchmarks/results/p3_train_mem_horizon_paired.json || exit 1
echo "=== paired horizon probe complete at $(date -Is) ==="
