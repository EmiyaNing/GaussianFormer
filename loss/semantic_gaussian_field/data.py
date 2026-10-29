"""Camera-valid component labels and exact voxel-solid distances on grid centers."""
from pathlib import Path
import hashlib
import numpy as np
from scipy import ndimage as ndi

VERSION = 'camera_semantic_components_half_lattice_distance_v1'


def construct(labels, camera, h=.4):
    foreground = camera.astype(bool) & (labels >= 0) & (labels < 17)
    components = np.zeros(labels.shape, np.int32)
    next_id = 0
    structure = ndi.generate_binary_structure(3, 1)
    for c in range(17):
        group, n = ndi.label(foreground & (labels == c), structure)
        take = group > 0
        components[take] = group[take] + next_id
        next_id += n
    if not foreground.any():
        return np.zeros(labels.shape, np.float32), components
    # Every closest point from a voxel center onto an axis-aligned voxel cube
    # lies on the half-voxel lattice. Filling all 27 lattice sites of each
    # foreground cube makes these distances exact at original grid centers.
    fine = np.zeros(tuple(2 * n + 1 for n in labels.shape), dtype=bool)
    fine[1::2, 1::2, 1::2] = foreground
    fine = ndi.binary_dilation(fine, np.ones((3, 3, 3), dtype=bool))
    distance = ndi.distance_transform_edt(~fine, sampling=h / 2)
    return np.ascontiguousarray(distance[1::2, 1::2, 1::2], dtype=np.float32), components


def cached(labels, camera, folder, h=.4):
    labels, camera = np.asarray(labels), np.asarray(camera, dtype=bool)
    key = hashlib.sha256(VERSION.encode() + labels.tobytes() + camera.tobytes()).hexdigest()
    path = Path(folder) / (key + '.npz')
    if path.exists():
        with np.load(path) as z:
            assert str(z['version']) == VERSION
            return z['distance'].copy(), z['components'].copy()
    distance, components = construct(labels, camera, h)
    path.parent.mkdir(parents=True, exist_ok=True)
    import os
    temp = path.with_suffix(f'.{os.getpid()}.tmp')
    with temp.open('wb') as stream:
        np.savez(stream, version=np.asarray(VERSION), distance=distance, components=components)
    temp.replace(path)
    return distance, components
