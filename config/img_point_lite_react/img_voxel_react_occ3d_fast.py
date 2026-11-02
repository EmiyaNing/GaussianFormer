"""Execution-only React optimization; model tensors/checkpoint keys unchanged.

Same data, losses, sampling points, precision and Gaussian budget as
img_voxel_react_occ3d.py, but images are fed at 256 x 704 instead of the
parent's 1600 x 864. The inherited dataset configs bake in their own
``data_aug_conf``, so the child has to override ``train_dataset_config`` and
``val_dataset_config`` explicitly; redefining the top-level ``data_aug_conf``
alone never reaches them. The regular render-grid cache requires unperturbed
ego-frame Occ3D grid coordinates, as provided by the inherited pipeline.
"""

_base_ = ['./img_voxel_react_occ3d.py']

input_shape = (704, 256)
data_aug_conf = {
    "resize_lim": (0.44, 0.44),
    "final_dim": input_shape[::-1],
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 900,
    "W": 1600,
    "rand_flip": True,
}

# The parent expands its dataset configs with the old ``data_aug_conf``, so
# the override has to be pushed into both dataset configs explicitly.
train_dataset_config = dict(data_aug_conf=data_aug_conf)
val_dataset_config = dict(data_aug_conf=data_aug_conf)

# The inherited loaders default to ``num_workers=0`` (train) / ``2`` (val);
# both are overridden to 4 for this recipe.
train_loader = dict(num_workers=4)
val_loader = dict(num_workers=4)

model = dict(
    encoder=dict(cache_image_features=True, skip_unused_anchor_embed=True,
                 terminal_densify=True),
    head=dict(analytic_covariance=True, separate_empty=True,
              incremental_densify=True, cache_render_grid=True,
              cuda_kwargs=dict(validate_inputs=False)),
)
