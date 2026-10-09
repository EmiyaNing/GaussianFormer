"""P4.3 follow-up: density cap K=8 evaluated at a *finer* hash cell (0.9 m).

Why this exists
---------------
The density cap does not select "the nearest K neighbours"; it keeps the first K
neighbours found while visiting cell shells in increasing Chebyshev radius.  That
order is defined in terms of ``cell_size``, so the selected subset changes with
the bin edge: at K=8, moving from cell 1.8 to cell 0.6 leaves only **0.1%** of
queries with the same neighbour set (mean Jaccard 0.100; measured by
``benchmarks/p4/cell_invariance.py --cap-scan``).

P1-1 and P0-4 report "cap=8 costs -0.26 mIoU" measured at cell 1.8.  If that
cost is a property of the *cap* it should survive an almost-total change of the
selected neighbours; if it is an artifact of the shell heuristic at that one bin
size, it should not.  This variant keeps every weight frozen and changes only the
bin edge, exactly like ``p4_rob_cell0p9.py``, but with the cap enabled.
"""
_base_ = ['./p1_a5_gaussian.py']

model = dict(semantic_gaussian=dict(max_neighbours=8, cell_size=0.9))
