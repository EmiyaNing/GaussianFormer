"""P1/A4: OPUS-v1 CUDA Gaussian with global learned geometry."""
_base_ = ['./p1_adapter_base.py']

model = dict(head=dict(gaussian_freeze_geometry=True))
