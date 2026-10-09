"""P1/A2'': KNN whose EFFECTIVE field matches A5's, not its nominal one.

Companion to ``p1_a2_knn_k215.py``.  The Gaussian weighting is not uniform, so
A5's nominal 215.2 accepted neighbours are not 215.2 equally-weighted votes.  The
effective sample size

    ESS = (sum_h w_h)^2 / sum_h w_h^2

measured over the real workload at A5's learned scale (0.53) is **59.3 on
average, median 38.6** (``benchmarks/p4/effective_sample_size.py``), i.e.
ESS/accepted = 0.278.  A uniform mean over k neighbours has ESS = k exactly.

So the two controls answer different questions:

* **k = 215** (``p1_a2_knn_k215``): matches A5's *nominal* field.  Tests whether
  uniform weighting survives a 3.6x larger effective field than A5's, i.e. the
  "Gaussian is a stabiliser for large-field aggregation" hypothesis.
* **k = 59** (this file): matches A5's *effective* field.  This is the clean
  test of P1-3's confound -- at the same effective field, does Gaussian
  weighting beat a uniform mean?  If A2''-59 ~ A5, the weighting form is
  irrelevant; if A2''-59 > A5, uniform wins outright; if A2''-59 < A5, the
  kernel is doing real work at matched field size.

Per-query ESS varies widely (p50 38.6, p95 199), so no single k matches every
query; the mean is used here and the spread is recorded in
``benchmarks/results/p4_ess.json``.
"""
_base_ = ['./p1_a5_gaussian.py']

model = dict(
    semantic_gaussian=dict(
        _delete_=True,
        type='KNNResidualAggregator',
        embed_dims=256, num_children=48, num_classes=17,
        pc_range=[-40., -40., -1., 40., 40., 5.4],
        num_neighbors=59, include_self=True, mode='mean',
        query_chunk_size=256))
