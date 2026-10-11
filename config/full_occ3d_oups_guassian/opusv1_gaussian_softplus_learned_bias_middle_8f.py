"""OPUS-M: Q=2400, S=2, 32 Gaussian children/query, 30 epochs."""
_base_ = ['./_base_8f.py']
model = dict(lifter=dict(num_queries=2400),
             encoder=dict(num_points=2, num_refines=[1, 2, 4, 8, 16, 32]))
