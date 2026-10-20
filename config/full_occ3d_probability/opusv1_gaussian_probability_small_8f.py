"""OPUS-S probability baseline: 1,200 queries, 76,800 Gaussians, 30 epochs."""
_base_ = ['./_base_8f.py']

model = dict(lifter=dict(num_queries=1200),
             encoder=dict(num_points=2, num_refines=[1, 4, 8, 16, 32, 64]))
