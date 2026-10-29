"""Pure NumPy grouping and aggregation, also used by independent tests."""
import numpy as np

SUPPORTS = ('core_1sigma', 'strict_3sigma', 'native', 'same_class_core_1sigma')
GROUPS = ('all', '0-20%', '20-40%', '40-60%', '60-80%', '80-100%')
DISTANCES = ('query_reference_center_to_voxel_center_grid',
             'query_reference_center_to_foreground_region_grid',
             'mean_child_GS_center_to_voxel_center_grid',
             'mean_child_GS_center_to_foreground_region_grid')
COUNTS = ('foreground_core_hit_GS', 'same_class_core_hit_GS',
          'foreground_core_GS_voxel_pairs', 'same_class_core_GS_voxel_pairs')


def quintiles(load):
    """High to low; equal load uses higher query ID first, matching old top20."""
    load = np.asarray(load)
    assert load.ndim == 1 and len(load) % 5 == 0
    order = np.argsort(load, kind='stable')[::-1]
    group = np.empty(len(load), np.int32)
    group[order] = np.repeat(np.arange(5, dtype=np.int32), len(load) // 5)
    return group


def summarize_frame(distances, counts, load, groups, unions, foreground, children):
    """Keep FP64 sums and integer denominators; do not average percentages."""
    result = []
    for g, name in enumerate(GROUPS):
        keep = np.ones(len(load), bool) if g == 0 else groups == g - 1
        n = int(keep.sum())
        row = dict(group=name, queries=n, GS=n * children,
                   foreground=int(foreground), assigned_foreground=int(load[keep].sum()))
        for i, key in enumerate(DISTANCES):
            row[key + '_sum'] = float(distances[keep, i].sum(dtype=np.float64))
        for i, key in enumerate(COUNTS):
            row[key] = int(counts[keep, i].sum(dtype=np.int64))
        for i, key in enumerate(SUPPORTS):
            row[key + '_union'] = int(unions[g, i])
        result.append(row)
    return result
