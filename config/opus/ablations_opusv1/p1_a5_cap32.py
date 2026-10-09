"""P1/P4 cap sweep: same A5 weights, maximum 32 accepted neighbours/query."""
_base_ = ['./p1_a5_gaussian.py']

model = dict(head=dict(gaussian_max_neighbours=32))
