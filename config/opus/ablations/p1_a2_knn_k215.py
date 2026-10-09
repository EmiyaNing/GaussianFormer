"""P1/A2' : KNN with a receptive field MATCHED to A5's learned one.

Why this exists
---------------
The plan lists A2 as "KNN-8 mean/max".  A2-k8 aggregates **8** uniformly
weighted nearest points; A5 aggregates every point inside its support box, which
at its learned scale (0.53) is **215.2 accepted neighbours on average** (p95 661;
measured on the real workload, see P4-1 and P4-8 section 3).  So A5 vs A2-k8
differs in two things at once:

* the **weighting** (Gaussian distance weighting vs uniform mean), and
* the **receptive field / aggregation arithmetic** (215 neighbours and ~11,600
  flops per query vs 8 neighbours and ~272).

P1's verdict ("the gain comes from residual capacity, not from aggregation")
rests on that comparison, so the two must be separated.  This variant changes
**only** the neighbour count, to A2's receptive field at A5's *final* trained
state -- which is the state the comparison is actually evaluated in.  A5 grows
its scale during training (0.35 -> 0.53), so its receptive field grows from ~95
to ~215; matching the final state is the right choice for an eval-time
comparison, and k = 95 is the corresponding alternative if the initial state is
of interest instead.

Four outcomes, and one of them would partially EXONERATE the Gaussian:

* A2'-215 ~ A2-k8  -> neighbour count is irrelevant, so the model's learned
  growth of the receptive field was wasted, strengthening "the aggregation
  mechanism does not matter".
* A2'-215 >> A5    -> uniform weighting over a large neighbourhood beats
  Gaussian weighting: the kernel is actively harmful.
* A2'-215 << A2-k8, while A5 stays near A2-k8 -> **uniform weighting collapses
  at a large receptive field but Gaussian weighting does not.**  A5 would then
  be delivering A2-k8-level accuracy while aggregating a 27x larger field,
  i.e. the Gaussian kernel acts as a *stabiliser for large-field aggregation*.
  Gate 1 of the plan (must beat KNN on accuracy) would still FAIL, but the
  interpretation would change from "Gaussian is useless" to "Gaussian buys
  field size rather than accuracy" -- a mechanism claim worth reporting rather
  than a null result.
* A2'-215 between the two -> partial; report the monotone trend in k.

This also completes T1.2's compute-alignment requirement in the aggregation
dimension, where only an analytic estimate existed before (P1-4 section 7.5).
"""
_base_ = ['./p1_a5_gaussian.py']

model = dict(
    semantic_gaussian=dict(
        _delete_=True,
        type='KNNResidualAggregator',
        embed_dims=256, num_children=48, num_classes=17,
        pc_range=[-40., -40., -1., 40., 40., 5.4],
        num_neighbors=215, include_self=True, mode='mean',
        query_chunk_size=256))
