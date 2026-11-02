"""Explicit 1.6N cap/shape: construct only K=0.6N children after selection."""

_base_ = ["./adaptive_allocationv7_occ3d.py"]
model = dict(encoder=dict(densify_layer=dict(allocation_ratio=0.6)))
