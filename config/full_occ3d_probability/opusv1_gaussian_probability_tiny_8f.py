"""OPUS-T probability baseline: 600 queries, 76,800 Gaussians, 30 epochs."""
_base_ = ['./_base_8f.py']

model = dict(lifter=dict(num_queries=600),
             encoder=dict(num_points=4, num_refines=[1, 4, 16, 32, 64, 128]))
