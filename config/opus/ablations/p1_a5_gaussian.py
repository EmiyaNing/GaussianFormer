"""P1/A5: frozen SparseWorld + point-anchored Semantic Gaussian adapter.

Bounded-budget variant of ``config/opus/sparse_gaussian_world_5e.py``:

* warm-starts from the converted released epoch-56 checkpoint (the upstream
  ``train_load_from`` is commented out, so it must be set here);
* trains on the reduced split from ``tools/make_reduced_split.py`` so that one
  epoch costs minutes instead of the full split's ~1.6 h;
* evaluates every epoch, because ``train.py`` only writes a checkpoint at an
  epoch boundary.

The budget reduction is a hardware compromise and is reported as such; it is
not the full P1 protocol.
"""
_base_ = ['../sparse_gaussian_world_5e.py']

# Warm start: the converted released SparseWorldStrict epoch-56 weights.
train_load_from = 'ckpts/sparseworld_strict_epoch_56.pth'
sparse_gaussian_warmstart = True

# Reduced split (see tools/make_reduced_split.py).
train_dataset_config = dict(imageset='data/reduced_splits/train_reduced.pkl')
val_dataset_config = dict(imageset='data/reduced_splits/val_reduced.pkl')

# batch 4 (not the upstream 8) leaves head-room on a 24-GiB RTX 4090 and avoids
# the allocator's NVML-backed OOM path, which is broken in this environment
# (driver/library mismatch 595.84 vs 595.91). Every variant uses the same
# batch size, so the comparison stays controlled.
train_loader = dict(batch_size=4, num_workers=4, shuffle=True)
val_loader = dict(batch_size=1, num_workers=4)
# ~85 iterations/epoch on the reduced split -> 40 epochs is ~3,400 iterations.
# That clears the loss warm-up ramp at iteration 1,000 with ~2,400 annealing
# iterations left.  The cosine schedule's t_initial scales with max_epochs, so
# every variant must use the SAME budget to be comparable (and to anneal at
# all); this is the matched budget for all P1 variants.
max_epochs = 40
eval_every_epochs = 40
print_freq = 10
