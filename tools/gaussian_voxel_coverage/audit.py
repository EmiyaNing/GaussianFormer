"""Independent NumPy pair enumeration for the CUDA coverage diagnostic."""
import numpy as np
import torch
from . import extension


def reference(info, shape):
    keys = ('means', 'inverse', 'alpha', 'semantic', 'centers', 'radii',
            'xyz', 'targets', 'winner', 'maximum')
    a = {k:info[k].detach().cpu().numpy() for k in keys}
    nt = len(a['targets'])
    lookup = np.full(len(a['xyz']), -1, dtype=np.int64)
    lookup[a['targets']] = np.arange(nt)
    counts = np.zeros((nt, 7), np.int32)
    mass = np.zeros((nt, 11), np.float64)
    nearest = np.full((nt, 2), np.inf)
    best = np.zeros(nt, np.float32)
    owner = np.full(nt, -1, np.int64)
    for g in range(len(a['means'])):
        if not a['alpha'][g] > 0:
            continue
        lo = np.maximum(a['centers'][g] - a['radii'][g], 0)
        hi = np.minimum(a['centers'][g] + a['radii'][g] + 1, shape)
        cells = np.stack(np.meshgrid(*[np.arange(l, h) for l, h in zip(lo, hi)], indexing='ij'), -1).reshape(-1, 3)
        v = (cells[:, 0] * shape[1] + cells[:, 1]) * shape[2] + cells[:, 2]
        t = lookup[v]
        v, t = v[t >= 0], t[t >= 0]
        dx = a['means'][g] - a['xyz'][v]
        inv = a['inverse'][g]
        power = -.5 * (inv[:3] * dx**2).sum(1) - (
            inv[3:] * dx[:, [0, 1, 0]] * dx[:, [1, 2, 2]]).sum(1)
        kernel = np.exp(power).astype(np.float32)
        weight = a['alpha'][g] * kernel
        valid = weight > 0
        t, dx, power, kernel, weight = [x[valid] for x in (t, dx, power, kernel, weight)]
        d = np.sqrt(np.maximum(-2 * power, 0))
        e = np.sqrt((dx**2).sum(1))
        counts[t, 0] += 1
        counts[t, 1] += d <= 1
        counts[t, 2] += d <= 2
        counts[t, 3] += d <= 3
        nearest[t, 0] = np.minimum(nearest[t, 0], d)
        nearest[t, 1] = np.minimum(nearest[t, 1], e)
        mass[t, 0] += kernel
        mass[t, 1] += weight
        mass[t, 4] += kernel * d
        w = weight * a['semantic'][g, a['winner'][t]]
        valid = w > 0
        t, d, e, w = [x[valid] for x in (t, d, e, w)]
        q = np.divide(w, a['maximum'][t], out=np.zeros_like(w), where=a['maximum'][t] > 0)
        counts[t, 4] += 1
        counts[t, 5] += q >= .01
        counts[t, 6] += q >= .10
        mass[t, 2] += w
        mass[t, 3] += q*q
        mass[t, 5] += w*d
        mass[t, 6] += w*e
        bands = np.where(d <= 1, 0, np.where(d <= 2, 1, np.where(d <= 3, 2, 3)))
        mass[t, 7+bands] += w
        update = w > best[t]
        best[t[update]] = w[update]
        owner[t[update]] = g
    return counts, mass, nearest, owner


def reference_audit(result, info, shape):
    expected = reference(info, shape)
    actual = [x.detach().cpu().numpy() for x in result]
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_allclose(actual[1], expected[1], rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(actual[2], expected[2], rtol=3e-5, atol=3e-6)
    ids = 0xffffffff - (actual[3] & 0xffffffff)
    ids[actual[3] == 0] = -1
    np.testing.assert_array_equal(ids, expected[3])


def synthetic_audit():
    rng = np.random.default_rng(928)
    shape = (10, 8, 6)
    cells = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing='ij'), -1).reshape(-1, 3)
    xyz = ((cells+.5)*.4).astype(np.float32)
    means = np.array([[1.131, .713, .619], [.617, 1.277, .477],
                      [2.321, .925, .901], [2.113, .621, 1.003]], np.float32)
    matrices = []
    for _ in means:
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        matrices.append(q @ np.diag(1/rng.uniform(.15, .4, 3)**2) @ q.T)
    matrices = np.array(matrices, np.float32)
    inv = matrices.reshape(-1, 9)[:, [0, 4, 8, 1, 5, 2]].copy()
    centers = (means/.4).astype(np.int32)
    radii = np.array([3, 2, 3, 3], np.int32)
    alpha = np.array([.83, .97, 0., .51], np.float32)
    sem = rng.uniform(.05, 5, size=(4, 17)).astype(np.float32)
    sem[3, :9] = 0
    winner = rng.integers(0, 17, len(xyz), dtype=np.int32)
    maximum = np.full(len(xyz), 5, np.float32)
    gpu = lambda x:torch.from_numpy(x).cuda().contiguous()
    lookup = np.arange(len(xyz), dtype=np.int32)
    info = dict(means=gpu(means), inverse=gpu(inv), alpha=gpu(alpha), semantic=gpu(sem),
        centers=gpu(centers), radii=gpu(radii), xyz=gpu(xyz), targets=gpu(lookup).long(),
        winner=gpu(winner).long(), maximum=gpu(maximum))
    result = extension().coverage(info['means'], info['inverse'], info['alpha'], info['semantic'],
        info['centers'], info['radii'], info['xyz'], gpu(lookup), gpu(winner), info['maximum'], *shape)
    reference_audit(result, info, shape)
    assert (result[0][:, 0] > 0).any() and (result[0][:, 0] == 0).any()
    return dict(status='passed', gaussians=4, voxels=len(xyz),
        checks=['all counts', 'normalized/euclidean positions', 'weighted radial sums',
                'positive-alpha exclusion', 'zero semantic class exclusion', 'dominant contributor'])
