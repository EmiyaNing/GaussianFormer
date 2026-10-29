"""Full 6,019-frame validation: all six renders and fixed initial load groups."""
from pathlib import Path
import gc
import json
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.all_layer_render_experiment import *
from tools.boundary_gaussian_stats.data import EvaluationFrames, FeatureCache
from tools.probe_gaussian_scale_supervision import layout
from tools.decoder_layer_stats import extension
from tools.decoder_layer_stats.common import quintiles
from tools.query_foreground_distance import extension as distance_extension
from tools.stat_decoder_layers import capture
from loss.opus_set_loss import _nearest_indices

HISTORY = ROOT / 'work_dirs/gaussian_sizes_full_val/small/scenes'


@torch.inference_mode()
def measure(net, out, batch, initial_load):
    q, children = 1200, [1, 4, 8, 16, 32, 64]
    xyz = out['sampled_xyz'][0].contiguous()
    y = batch['occ_label'].flatten().int().contiguous()
    valid = batch['occ_cam_mask'].flatten().bool() & (y >= 0) & (y < 18)
    targets = torch.where(valid & (y < 17))[0]
    gt = xyz[targets].contiguous()
    lookup = torch.full_like(y, -1)
    lookup[targets] = torch.arange(len(gt), device='cuda', dtype=torch.int32)
    groups = quintiles(initial_load)
    group_cuda = torch.tensor(groups, device='cuda', dtype=torch.int32)
    agg = net.head.aggregator
    ds, cs, us, qs, cms, refs, attrs, class_cc = [], [], [], [], [], [], [], []
    assert len(out['pred_occ']) == len(out['gaussians']) == 6
    for i, (gaussian, k, logits, raw) in enumerate(zip(out['gaussians'], children, out['pred_occ'], net._decoder_raw_points)):
        _, values = layout(net.head, gaussian)
        means, scales, inverse, alpha, sem, centers, radii = values
        assert len(means) == q*k
        pc = raw.new_tensor(net.encoder.pc_range)
        ref = (pc[:3] + raw[0]*(pc[3:]-pc[:3])).mean(1).contiguous()
        d = distance_extension().distances(torch.cat((ref, means)).contiguous(), gt, agg.grid_size)
        assert bool(torch.isfinite(d).all())
        query_d = torch.cat((d[:q].double(), d[q:].reshape(q, k, 2).double().mean(1)), 1)
        classes = sem.argmax(-1).int().contiguous()
        bits, coverage, unions, gs = extension().support(means, inverse, alpha, centers, radii,
            xyz, lookup, y, classes, group_cuda, len(gt), q, k, agg.H, agg.W, agg.D)
        hits = (bits != 0).any(2)
        cc = torch.zeros((17, 5), device='cuda', dtype=torch.int64)
        cc[:, 0] = torch.bincount(y[targets].long(), minlength=17)
        for j in range(4):
            cc[:, j+1] = torch.bincount(y[targets][hits[j]].long(), minlength=17)
        pred = logits[0].argmax(0).flatten()
        cm = torch.bincount((y[valid]*18+pred[valid]).long(), minlength=324).reshape(18, 18)
        assert bool((coverage[3] <= coverage[0]).all()) and bool((gs[:, 1] <= gs[:, 0]).all())
        assert bool((unions[:, 0] <= unions[:, 1]).all()) and bool((unions[:, 1] <= unions[:, 2]).all())
        ds.append(query_d.cpu().numpy())
        cs.append(torch.cat(((gs > 0).reshape(q, k, 2).sum(1), gs.reshape(q, k, 2).long().sum(1)), 1).cpu().numpy())
        us.append(unions.cpu().numpy())
        qs.append(coverage.T.cpu().numpy())
        cms.append(cm.cpu().numpy())
        refs.append(ref.cpu().numpy())
        attrs.append(np.array([len(means), float(scales.double().sum()), float(scales.amax(1).double().sum()),
            float(alpha.double().sum()), float(sem.double().sum()), int((scales.amax(1) >= .3999).sum())]))
        class_cc.append(cc.cpu().numpy())
        if i == 5:
            load = torch.bincount(_nearest_indices(gt, means, 1024)//k, minlength=q).cpu().numpy().astype(np.int32)
        del bits, coverage, gs, unions
    return dict(query_distances=np.stack(ds), query_GS_counts=np.stack(cs),
        group_unions=np.stack(us), query_coverage=np.stack(qs), reference_centers=np.stack(refs),
        confusion=np.stack(cms), class_coverage=np.stack(class_cc), gaussian_attributes=np.stack(attrs),
        load_sixth=load, initial_load=initial_load, load_groups=groups, foreground=np.array(len(gt), np.int64))


def evaluate(arm):
    assert_protocol()
    from dataset import OPENOCC_DATASET
    from dataset.utils import custom_collate_fn_temporal
    from torch.utils.data import DataLoader, Subset
    checkpoint = INITIAL if arm == 'initial' else RUN / 'train' / arm / 'epoch_1.pth'
    checkpoint_sha = sha(checkpoint)
    folder = RUN / 'eval' / arm
    if (folder / 'complete.json').exists():
        assert json.loads((folder / 'complete.json').read_text())['checkpoint_sha256'] == checkpoint_sha
        return
    cfg, net = build('baseline' if arm == 'initial' else arm, None if arm == 'initial' else checkpoint)
    hook = capture(net)
    ds = EvaluationFrames(cfg, 'val')
    assert len(ds) == 6019
    scenes = {}
    for i, (scene, _) in enumerate(ds.keyframes):
        scenes.setdefault(scene, []).append(i)
    assert len(scenes) == 150
    pending = []
    for scene, ids in scenes.items():
        path = folder / 'scenes' / f'{scene}.json'
        if path.exists():
            meta = json.loads(path.read_text())
            assert meta['indices'] == ids and meta['checkpoint_sha256'] == checkpoint_sha
            assert meta['npz_sha256'] == sha(path.with_suffix('.npz'))
        else:
            pending.extend(ids)
    # Check all six predictions against native 48-image input once per model.
    native = OPENOCC_DATASET.build(cfg.val_dataset_config)
    a = to_cuda(custom_collate_fn_temporal([native[0]]))
    with torch.inference_mode():
        original = forward(net, a)
        expected = [p.detach().cpu() for p in original['pred_occ']]
    del original, a, native
    cache = FeatureCache(net, ds)
    b = to_cuda(custom_collate_fn_temporal([ds[0]]))
    with torch.inference_mode():
        cached = forward(net, b)
        for reference, actual in zip(expected, cached['pred_occ']):
            torch.testing.assert_close(reference, actual.cpu(), rtol=0, atol=0)
    del expected, cached, b
    write(folder / 'pilot.json', dict(status='passed', native_and_cached_six_logits_exact=True))
    loader = DataLoader(Subset(ds, pending), batch_size=1, num_workers=3, shuffle=False,
        collate_fn=custom_collate_fn_temporal, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    start, done, active = time.monotonic(), 0, None
    try:
        for index, batch in zip(pending, loader):
            scene, f = ds.keyframes[index]
            if active is None:
                active, arrays, ids, tokens = scene, {}, [], []
                prior_meta = json.loads((HISTORY / f'{scene}.json').read_text())
                assert prior_meta['indices'] == scenes[scene]
                with np.load(HISTORY / f'{scene}.npz') as z:
                    prior = {key: z[key] for key in ('load_sixth', 'query_core', 'class_coverage', 'confusion')}
            assert active == scene
            batch = to_cuda(batch)
            with torch.inference_mode():
                out = forward(net, batch)
                row = measure(net, out, batch, prior['load_sixth'][len(ids)].astype(np.int32))
            if arm == 'initial':
                np.testing.assert_array_equal(row['load_sixth'], prior['load_sixth'][len(ids)])
                np.testing.assert_array_equal(row['query_coverage'][-1, :, 0], prior['query_core'][len(ids)])
                np.testing.assert_array_equal(row['confusion'][-1], prior['confusion'][len(ids)])
            token = ds.base.scene_infos[scene][f]['token']
            assert token == prior_meta['tokens'][len(ids)]
            for key, value in row.items():
                arrays.setdefault(key, []).append(value)
            ids.append(index)
            tokens.append(token)
            done += 1
            if index == scenes[scene][-1]:
                path = folder / 'scenes' / f'{scene}.npz'
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.with_suffix('.tmp').open('wb') as stream:
                    np.savez_compressed(stream, **{key: np.stack(value) for key, value in arrays.items()})
                path.with_suffix('.tmp').replace(path)
                write(path.with_suffix('.json'), dict(status='complete', indices=ids, tokens=tokens,
                    scene=scene, frames=len(ids), checkpoint_sha256=checkpoint_sha, npz_sha256=sha(path)))
                progress = dict(status='running', phase='evaluation', arm=arm,
                    frames=6019-len(pending)+done, expected_frames=6019,
                    scenes=len(list((folder / 'scenes').glob('*.json'))), expected_scenes=150,
                    frames_per_second=done/(time.monotonic()-start))
                write(folder / 'progress.json', progress)
                write(RUN / 'status.json', progress)
                print(json.dumps(progress), flush=True)
                active = None
            del out, batch, row
        paths = sorted((folder / 'scenes').glob('*.json'))
        all_ids = [i for p in paths for i in json.loads(p.read_text())['indices']]
        tokens = [t for p in paths for t in json.loads(p.read_text())['tokens']]
        assert len(paths) == 150 and sorted(all_ids) == list(range(6019)) and len(set(tokens)) == 6019
        assert_protocol()
        status = dict(status='complete', frames=6019, scenes=150, checkpoint_sha256=checkpoint_sha,
                      six_layers=True, seconds_this_run=time.monotonic()-start)
        write(folder / 'complete.json', status)
        write(folder / 'progress.json', status)
    finally:
        hook.remove()
        cache.close()
        release(net)
        del net, loader
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('arm', choices=('initial', *ARMS))
    evaluate(p.parse_args().arm)
