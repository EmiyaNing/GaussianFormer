"""Execution-only React optimization; model tensors/checkpoint keys unchanged.

Same data, losses, sampling points, precision, image size and Gaussian budget
as img_voxel_react_occ3d.py. The regular render-grid cache requires unperturbed
ego-frame Occ3D grid coordinates, as provided by the inherited pipeline.
"""

_base_ = ['./img_voxel_react_occ3d.py']

model = dict(
    encoder=dict(cache_image_features=True, skip_unused_anchor_embed=True,
                 terminal_densify=True),
    head=dict(analytic_covariance=True, separate_empty=True,
              incremental_densify=True, cache_render_grid=True,
              cuda_kwargs=dict(validate_inputs=False)),
)
