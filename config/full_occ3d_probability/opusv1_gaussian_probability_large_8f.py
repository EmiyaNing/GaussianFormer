"""OPUS-L probability baseline: 4,800 queries, 76,800 Gaussians, 30 epochs."""
_base_ = ['./_base_8f.py']

model = dict(lifter=dict(num_queries=4800),
             encoder=dict(num_points=2, num_refines=[1, 2, 4, 8, 16, 16]))
