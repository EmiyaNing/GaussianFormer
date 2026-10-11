"""OPUS-T: Q=600, S=4, 128 Gaussian children/query, 30 epochs."""
_base_ = ['./_base_8f.py']
model = dict(lifter=dict(num_queries=600),
             encoder=dict(num_points=4, num_refines=[1, 4, 16, 32, 64, 128]))
