"""Independent 19,200-Gaussian mini baseline; only final child count changes."""
_base_ = ['./a1_align_final.py']

model = dict(encoder=dict(num_refines=[1, 4, 16, 32, 64, 32]))
