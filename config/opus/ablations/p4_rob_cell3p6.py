"""P4.3 robustness: frozen A5 adapter evaluated with a coarser hash cell (更少 cell、每个 cell 更多 Gaussian).

Every weight is frozen; only this one inference-time knob of the sparse G2Q
operator changes.  ``cell_size`` was 1.8 during
training, so a change in mIoU here is pure domain shift of the neighbourhood
operator, not a retrained adaptation.

P0 measured the *cost* side of these knobs; this measures the *accuracy* side,
which the plan's P4.3 asks for.
"""
_base_ = ['./p1_a5_gaussian.py']

model = dict(semantic_gaussian=dict(cell_size=3.6))
