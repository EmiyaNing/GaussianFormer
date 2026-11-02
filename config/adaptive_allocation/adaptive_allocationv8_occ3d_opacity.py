"""Baseline-equivalent selection control; still trains the detached scorer."""

_base_ = ["./adaptive_allocationv8_occ3d.py"]
model = dict(encoder=dict(densify_layer=dict(selection_mode="opacity")))
