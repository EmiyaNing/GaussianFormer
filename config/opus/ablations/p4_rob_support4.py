"""P4.3 robustness: frozen A5 adapter evaluated with a larger support reach than the trained 3.0.

Every weight is frozen; only this one inference-time knob of the sparse G2Q
operator changes.  ``support_sigma`` was 3.0 during
training, so a change in mIoU here is pure domain shift of the neighbourhood
operator, not a retrained adaptation.

P0 measured the *cost* side of these knobs; this measures the *accuracy* side,
which the plan's P4.3 asks for.
"""
_base_ = ['./p1_a5_gaussian.py']

model = dict(semantic_gaussian=dict(support_sigma=4.0))
