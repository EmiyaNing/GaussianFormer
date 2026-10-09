"""P1/A2-gauss: A2's exact k-NN selection with Gaussian instead of uniform weights.

Why this exists
---------------
P1-6 section 4.2.1 left one mechanism question open.  A5 (Gaussian, support-box
shell selection) is 0.41 worse on thin classes than A2 (uniform, 8 Euclidean
nearest) at a matched neighbour count of 8 -- but that comparison confounds two
things, because the two rules pick largely different neighbours: measured
overlap 0.177, mean neighbour distance 0.399 m (k-NN) vs 0.978 m (shell).

This variant separates them by changing **only the weight profile**:

* **A2 (uniform, k=8) vs this variant (Gaussian, k=8)** -> isolates the weighting,
  with the selection held fixed;
* **A5 cap=8 (Gaussian, shell selection) vs this variant (Gaussian, k-NN
  selection)** -> isolates the selection, with the weighting held fixed.

The scale is a fixed constant (not learned), so the parameter count stays
identical to A2's; the plan's T1.2 protocol requires exactly that.  Note there is
no opacity factor here, unlike A5's ``opacity * kernel``: this isolates the
*distance decay itself*, and adding the trained opacity would reintroduce a
confound with A5's learned geometry.

**Why 0.09 and not 0.35.** The first draft used 0.35, A5's *initialisation*
scale, on the reasoning that a value inherited from the architecture is not tuned
on the outcome.  Measuring the manipulation (``selection_overlap.py
--scale-sweep``) showed that reasoning produced a nearly vacuous control: the
8 nearest neighbours sit 0.399 m away on average, so at scale 0.35 the Gaussian
weights have ESS **6.97 of 8** -- only 13% more concentrated than uniform, and a
null result would have shown nothing.  The scale is therefore chosen by
**relative concentration** instead: A5's kernel has ESS/accepted = **0.278** over
its own accepted neighbours, and **0.09** reproduces that ratio over the k-NN set
(ESS 2.31 of 8, ESS/k **0.288**, nearest neighbour carrying 68.5% of the weight).

That criterion is fixed before the run and depends on no accuracy outcome, so it
is a design choice rather than post-hoc tuning; the run had not started when it
was changed.
"""
_base_ = ['./p1_a5_gaussian.py']

model = dict(
    semantic_gaussian=dict(
        _delete_=True,
        type='KNNResidualAggregator',
        embed_dims=256, num_children=48, num_classes=17,
        pc_range=[-40., -40., -1., 40., 40., 5.4],
        num_neighbors=8, include_self=True, mode='gaussian',
        gaussian_scale=0.09, query_chunk_size=256))
