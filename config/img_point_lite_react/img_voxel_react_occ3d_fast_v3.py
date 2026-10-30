"""Third execution round: tiled exact-support React renderer and warp backward.

Requires building model/head/react_render_ops. Same model, data and objective
as fast_v2.py. The radius hint sizes workspace; overflow takes an exact GPU
fallback and never truncates contributions.
"""
_base_ = ['./img_voxel_react_occ3d_fast_v2.py']

model = dict(head=dict(react_renderer='tiled',
                       cuda_kwargs=dict(tile_size=4, radius_hint=5, cache_query_map=True)))
