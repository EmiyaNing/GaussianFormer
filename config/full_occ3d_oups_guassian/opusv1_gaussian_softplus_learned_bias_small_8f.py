"""OPUS-S: Q=1200, S=2, 64 Gaussian children/query, 30 epochs."""
_base_ = ['./_base_8f.py']
model = dict(lifter=dict(num_queries=1200),
             encoder=dict(num_points=2, num_refines=[1, 4, 8, 16, 32, 64]))
