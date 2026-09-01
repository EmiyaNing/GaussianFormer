"""Five-epoch Gaussian-adapter validation on the frozen epoch-56 baseline."""
_base_ = ['./sparse_gaussian_world.py']

model = dict(freeze_sparseworld=True)

# The released SparseWorld graph is frozen in this validation recipe, so its
# set/trajectory loss would add compute but no gradient. The full joint recipe
# remains available in sparse_gaussian_world.py.
loss = dict(
    _delete_=True,
    type='MultiLoss',
    loss_cfgs=[dict(
        type='SparseWorldGaussianPointOccupancyLoss', weight=1.,
        empty_label=17, focal_gamma=2., focal_alpha=.25,
        warmup_iters=1000, chunk_size=1024,
        horizon_weights=[1., .5, .5, .5, .5, .5, .5],
        enable_during_pretrain=False,
        class_weights=[3, 2, 3, 2, 2, 3, 3, 2, 3,
                       2, 2, 1, 2, 1, 1, 1, 1])])

# Measured at epoch-5's largest active training span: 15.1 GiB on RTX 4090.
train_loader = dict(batch_size=8, num_workers=4, shuffle=True)
val_loader = dict(batch_size=1, num_workers=4)
max_epochs = 5
eval_every_epochs = 5
