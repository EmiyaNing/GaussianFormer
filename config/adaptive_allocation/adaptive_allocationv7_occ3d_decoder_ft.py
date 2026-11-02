"""Optional second stage: fine-tune decoder + allocator, keep sensor trunks fixed.

Set load_from to an exported full-model V7 checkpoint from the first stage.
"""

_base_ = ["./adaptive_allocationv7_occ3d.py"]
load_from = "work_dirs/adaptive_allocation_v7_pilot/learned_s928_full.pth"
model = dict(freeze_encoder_except_densify=False, freeze_encoder_anchor_encoder=False)
