"""P4.3 follow-up: density cap K=8 evaluated at a *coarser* hash cell (3.6 m).

Coarser bin => larger cells => the shell visit order changes again, and the K=8
subset overlaps cell 1.8's by only ~7% of rows exactly / mean Jaccard 0.407
(``benchmarks/p4/cell_invariance.py --cap-scan``).  Paired with
``p4_cap8_cell0p9.py`` this brackets the bin edge from both sides, so the
question "is the cap's accuracy cost a property of the cap or of the bin size?"
is answered by a spread rather than a single re-run.

Same frozen A5 weights as every other cap variant; inference-time only.
"""
_base_ = ['./p1_a5_gaussian.py']

model = dict(semantic_gaussian=dict(max_neighbours=8, cell_size=3.6))
