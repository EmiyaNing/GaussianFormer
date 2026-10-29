"""Exact voxel-face distances at voxel centers, without meshing each frame.

Each reliable square face is sampled on a half-voxel lattice (nine points).
For a voxel-center query, its closest point on any axis-aligned voxel face
lies on that lattice. Exact nearest-site queries at original voxel centers
therefore give face distances, rather than opposite-class center distances.
"""
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

VERSION = 'known_voxel_faces_halfgrid_normals_v2'


def face_seeds(labels, camera):
    shape = labels.shape
    known = camera.astype(bool) & (labels >= 0) & (labels <= 17)
    fg = known & (labels != 17)
    fine = np.zeros(tuple(2*n+1 for n in shape), dtype=bool)
    faces = []
    for axis in range(3):
        lo = [slice(None)]*3; hi = lo.copy()
        lo[axis] = slice(None, -1); hi[axis] = slice(1, None)
        lo, hi = tuple(lo), tuple(hi)
        active = known[lo] & known[hi] & (fg[lo] != fg[hi])
        faces.append(active)
        tangents = [k for k in range(3) if k != axis]
        for a in range(3):
            for b in range(3):
                sl = [None]*3
                sl[axis] = slice(2, 2*shape[axis], 2)
                sl[tangents[0]] = slice(a, 2*shape[tangents[0]]+a, 2)
                sl[tangents[1]] = slice(b, 2*shape[tangents[1]]+b, 2)
                fine[tuple(sl)] |= active
    return fine, known, fg, faces


def construct(labels, camera, spacing=0.4, return_normals=False):
    seeds, known, fg, faces = face_seeds(labels, camera)
    if not seeds.any():
        result = (np.zeros(labels.shape, np.float32), np.zeros(labels.shape, bool))
        return (*result, np.zeros((3, *labels.shape), np.float32)) if return_normals else result
    sites = np.argwhere(seeds)
    indices = np.argwhere(known); queries = 2*indices+1
    distances, nearest = cKDTree(sites).query(queries, workers=1)
    normal = (queries-sites[nearest])/np.maximum(distances[:, None], 1e-10)
    normal[fg[known]] *= -1
    distance = np.zeros(labels.shape, np.float32)
    distance[known] = distances*(spacing/2)
    # Distances to unknown cubes are determined by known/unknown interfaces.
    seeds.fill(False)
    for axis in range(3):
        lo = [slice(None)]*3; hi = lo.copy()
        lo[axis] = slice(None, -1); hi[axis] = slice(1, None)
        active = known[tuple(lo)] != known[tuple(hi)]
        tangents = [k for k in range(3) if k != axis]
        for a in range(3):
            for b in range(3):
                sl = [None]*3; sl[axis] = slice(2, 2*labels.shape[axis], 2)
                sl[tangents[0]] = slice(a, 2*labels.shape[tangents[0]]+a, 2)
                sl[tangents[1]] = slice(b, 2*labels.shape[tangents[1]]+b, 2)
                seeds[tuple(sl)] |= active
    position = (indices+.5)*spacing
    to_unknown = np.minimum(position, np.asarray(labels.shape)*spacing-position).min(axis=1)
    if seeds.any():
        internal = cKDTree(np.argwhere(seeds)).query(queries, workers=1)[0]*(spacing/2)
        to_unknown = np.minimum(to_unknown, internal)
    valid = np.zeros(labels.shape, bool)
    valid[known] = distance[known]+1e-6 < to_unknown
    distance[fg] *= -1
    if not return_normals: return distance, valid
    normals = np.zeros((3, *labels.shape), np.float32)
    normals[:, known] = normal.T
    return distance, valid, normals


def volume(sdf, valid, spacing=0.4, normals=None):
    """[sdf, nx, ny, nz, center_valid, shape_valid] in [C,Z,Y,X]."""
    if normals is None:
        gradient = np.stack(np.gradient(sdf, spacing, edge_order=2))
        magnitude = np.linalg.norm(gradient, axis=0)
        neighbors_valid = ndi.binary_erosion(valid, structure=ndi.generate_binary_structure(3, 1), border_value=0)
        shape_valid = neighbors_valid & (magnitude >= 0.8) & (magnitude <= 1.2)
        normal = gradient / np.maximum(magnitude[None], 1e-6)
    else:
        magnitude = np.linalg.norm(normals.astype(np.float32), axis=0)
        normal = normals.astype(np.float32)/np.maximum(magnitude[None], 1e-6)
        shape_valid = valid & (magnitude > .9)
    out = np.concatenate((sdf[None], normal, valid[None], shape_valid[None]), axis=0).astype(np.float32)
    return np.ascontiguousarray(out.transpose(0, 3, 2, 1))


def save(path, sdf, valid, gt_sha256, normals=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as stream:
        arrays = dict(sdf=sdf, valid=np.packbits(valid.reshape(-1)),
                      gt_sha256=np.asarray(gt_sha256), version=np.asarray(VERSION))
        if normals is not None: arrays['normals'] = normals.astype(np.float16)
        np.savez(stream, **arrays)
    temp.replace(path)


def load(path, return_normals=False):
    with np.load(path, allow_pickle=False) as data:
        if str(data['version']) != VERSION:
            raise ValueError(f'Unsupported SDF cache: {path}')
        sdf = data['sdf']
        valid = np.unpackbits(data['valid'], count=sdf.size).reshape(sdf.shape).astype(bool)
        normals = data['normals'].astype(np.float32) if return_normals and 'normals' in data else None
    return (sdf, valid, normals) if return_normals else (sdf, valid)
