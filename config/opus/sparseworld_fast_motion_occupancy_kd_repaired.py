"""Matched repair experiment; original weights, split, seed and budget."""
_base_=['./sparseworld_fast_motion_occupancy_kd.py']
fast_root='out/sparseworld_fast_motion_distillation/20261009_repaired_2048x512'
fast_target_cache='out/sparseworld_fast_motion_distillation/20261008_2048x512'
# Match strict SparseWorld task gradients, not merely its forward values.
fast_position_gradient=False
fast_kd_policy='positive_supported_native'
