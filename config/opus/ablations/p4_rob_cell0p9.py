"""P4.3 robustness: evaluate the frozen A5 adapter with a finer hash cell.

``cell_size`` is the world-space bin edge the sparse G2Q operator uses for its
candidate lookup (a query scans the ``(2k+1)^3`` cells around it, with
``k = ceil(support_sigma * max_scale / cell_size)``).  Training used 1.8 m.  This
variant keeps every weight frozen and only changes the bin edge, so any change
in mIoU is pure inference-time domain shift of the neighbourhood operator.

P0's cell sweep measured the *cost* side of this knob; this measures the
*accuracy* side, which the plan's P4.3 asks for.
"""
_base_ = ['./p1_a5_gaussian.py']

model = dict(semantic_gaussian=dict(cell_size=0.9))
