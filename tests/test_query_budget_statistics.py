"""Independent tiny-scene enumeration for the query census CUDA diagnostic."""
import numpy as np
import pytest
import torch
from tools.query_budget_stats import extension


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA diagnostic')
@pytest.mark.parametrize('nq,children', [(2, 2), (66, 1)])
def test_query_census_matches_independent_enumeration(nq, children):
    rng = np.random.default_rng(928)
    shape = (6, 5, 4)
    xyz = np.stack(np.meshgrid(*[np.arange(n, dtype=np.float32) + .5 for n in shape], indexing='ij'), -1).reshape(-1, 3)
    ng, nv = nq * children, len(xyz)
    means = rng.uniform([.1, .1, .1], np.array(shape) - .1, size=(ng, 3)).astype('float32')
    scale = rng.uniform(.45, .95, size=(ng, 3)).astype('float32')
    inverse = np.concatenate((1 / scale**2, np.zeros_like(scale)), 1).astype('float32')
    alpha = rng.uniform(.1, .9, ng).astype('float32')
    alpha[0] = 0
    sem = rng.uniform(.01, 2., (ng, 17)).astype('float32')
    centers = means.astype('int32')
    radii = np.ceil(scale.max(1) * 3).astype('int32')
    labels = rng.integers(0, 18, nv, dtype='int32')
    labels[0] = 255
    camera = rng.uniform(size=nv) > .15
    target = np.where(camera & (labels < 17))[0]
    lookup = np.full(nv, -1, dtype='int32')
    lookup[target] = np.arange(len(target))
    assigned = rng.integers(0, nq, len(target), dtype='int32')
    winner = rng.integers(0, 17, nv, dtype='int32')
    counts = np.zeros((len(target), 3), dtype='int32')
    covered = np.zeros((len(target), nq), dtype=bool)
    gs = np.zeros((ng, 5), dtype='int32')
    best, owner = np.zeros(nv, dtype='float32'), np.full(nv, -1, dtype='int64')
    cells = xyz.astype('int32')
    for g in range(ng):
        if alpha[g] <= 0:
            continue
        for v in range(nv):
            if not camera[v] or not 0 <= labels[v] < 18:
                continue
            if np.any(cells[v] < centers[g] - radii[g]) or np.any(cells[v] > centers[g] + radii[g]):
                continue
            d2 = float(np.sum((means[g] - xyz[v])**2 * inverse[g, :3]))
            weight = float(alpha[g] * np.exp(-.5 * d2))
            if weight <= 0:
                continue
            core, three = d2 <= 1., d2 <= 9.
            gs[g, 0] += 1
            gs[g, 3] += core
            t = lookup[v]
            if t >= 0:
                counts[t] += [1, core, three]
                covered[t, g // children] |= core
                gs[g, 1] += 1
                gs[g, 2] += core
                gs[g, 4] += three
            w = weight * sem[g, winner[v]]
            if w > best[v]:
                best[v], owner[v] = w, g
    inputs = [means, inverse, alpha, sem, centers, radii, xyz, labels, camera, lookup, winner, assigned]
    result = extension().census(*[torch.as_tensor(a, device='cuda').contiguous() for a in inputs],
                                nq, children, *shape)
    actual = [a.cpu().numpy() for a in result]
    np.testing.assert_array_equal(actual[0], counts)
    np.testing.assert_array_equal(actual[3], gs)
    np.testing.assert_array_equal(actual[4], covered[np.arange(len(target)), assigned])
    np.testing.assert_array_equal(actual[5], covered.sum(1))
    qc = np.stack((covered.sum(0), covered[covered.sum(1) > 1].sum(0),
                   covered[covered.sum(1) == 1].sum(0)), 1)
    np.testing.assert_array_equal(actual[6], qc)
    decoded = 0xffffffff - (actual[2] & 0xffffffff)
    decoded[actual[2] == 0] = -1
    np.testing.assert_array_equal(decoded, owner)
    for q in range(nq):
        np.testing.assert_array_equal((actual[1][:, q // 64] >> (q % 64)) & 1, covered[:, q])
