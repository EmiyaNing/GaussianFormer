"""OPUS-M probability baseline: 2,400 queries, 76,800 Gaussians, 30 epochs."""
_base_ = ['./_base_8f.py']

model = dict(lifter=dict(num_queries=2400),
             encoder=dict(num_points=2, num_refines=[1, 2, 4, 8, 16, 32]))
