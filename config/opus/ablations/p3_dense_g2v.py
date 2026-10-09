"""P3 control: dense Gaussian-to-voxel rendering, identical in every other way.

Derived from the sparse A5 recipe and swapping **only** the aggregation module,
so a model-level comparison isolates the operator:

* same frozen SparseWorld backbone, same warm start, same data, same budget;
* parameter count is identical (180,501) because attribute prediction is
  inherited unchanged;
* the field is materialised on the full ``200x200x16`` voxel grid with the
  repository's dense Gaussian-to-voxel operator, then sampled back at the
  predicted points.

This is the missing dense reference for the premise measured in P3-1/P3-2.  It
is *not* a faithful S2GO reimplementation: S2GO's ``4x4x4`` block kernel is
still absent, so results must be labelled ``ours/dense-G2V`` and must never be
compared against S2GO's paper numbers.

    python3.10 benchmarks/p3/profile_model.py \
      --config config/opus/ablations/p3_dense_g2v.py \
      --checkpoint <a trained dense-G2V checkpoint>
"""
_base_ = ['./p1_a5_gaussian.py']

model = dict(
    semantic_gaussian=dict(
        _delete_=True,
        type='DenseG2VResidualAggregator',
        embed_dims=256, num_children=48, num_classes=17,
        pc_range=[-40., -40., -1., 40., 40., 5.4],
        scale_range=[[.1, .1, .1], [.6, .6, .6]],
        initial_scale=.35, initial_opacity=.02,
        num_neighbors=8, include_self=True, query_chunk_size=256,
        grid_shape=[200, 200, 16], grid_size=0.4, scale_multiplier=3.0))
