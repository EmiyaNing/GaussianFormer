"""Same generator, growth quota and loss; parent selection uses opacity."""

_base_ = ["./adaptive_allocationv7_occ3d.py"]
model = dict(encoder=dict(densify_layer=dict(selection_mode="opacity")))
