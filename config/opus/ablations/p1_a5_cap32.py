"""P1/A5 with the density cap enabled at evaluation time (K = 32).

Same trained adapter weights as ``p1_a5_gaussian.py``; only the neighbour cap
differs.  Evaluating one checkpoint under several caps measures the *inference
time* sensitivity of an already-trained adapter to neighbour truncation, which
is the condition P0 left open.

Evaluated with:

    python3.10 eval.py --py-config config/opus/ablations/p1_a5_cap32.py \
      --resume-from out/p1_a5_gaussian/epoch_100.pth \
      --work-dir out/p1_a5_cap32
"""
_base_ = ['./p1_a5_gaussian.py']

model = dict(
    semantic_gaussian=dict(max_neighbours=32))
