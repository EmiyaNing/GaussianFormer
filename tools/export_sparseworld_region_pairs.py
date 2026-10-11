#!/usr/bin/env python3
"""Export paired native forecasts/perception; enforce exact legacy metric replay."""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from mmengine import Config
from torch.utils.data import DataLoader, Subset

from dataset.sparseworld_trajectory import _pose
from dataset.utils import custom_collate_fn_temporal
from tools.diagnose_teacher_alignment import loader_for, tree_to
from tools.experiment_future2current import _future_window, baseline
from tools.experiment_future_response_distillation import build_student, paths_for
from tools.sparseworld_region_utils import HORIZONS, ROOT, joint_counts, sha256, write_json


def metadata(ds, position, source_index, scene):
    actual_scene, index = ds.keyframes[source_index]
    if actual_scene != scene:
        raise RuntimeError('Scene identity mismatch')
    frames = ds.scene_infos[scene]
    current = frames[index]
    history = [frames[i] for i in ds._history_indices(frames, index, 6, True)]
    future = ds._future_infos(current['token'])
    relative = {0: current, **{-i-1: f for i, f in enumerate(history)},
                **{i+1: f for i, f in enumerate(future)}}
    record = dict(position=position, index=source_index, scene=scene,
                  token=current['token'], timestamp=current['timestamp'],
                  current_pose=_pose(current).tolist(), current_occ_path=current['occ_path'],
                  camera_names=list(ds.sensor_types), targets=[])
    for h in HORIZONS:
        target = future[h-1]
        def frame_record(r):
            frame = relative[r]
            return dict(relative=r, token=frame['token'], timestamp=frame['timestamp'],
                        images=[str(Path(ds.data_path) / frame['data'][cam]['filename'])
                                for cam in ds.sensor_types])
        f_frames = [frame_record(r) for r in range(0, -5, -1)]
        p_frames = [frame_record(r) for r in range(h, h-5, -1)]
        assert all(f['timestamp'] <= current['timestamp'] for f in f_frames)
        assert all(f['timestamp'] <= target['timestamp'] for f in p_frames)
        record['targets'].append(dict(horizon=h, token=target['token'],
                                      timestamp=target['timestamp'],
                                      actual_seconds=(target['timestamp']-current['timestamp'])/1e6,
                                      pose=_pose(target).tolist(), annotations=target.get('anns', []),
                                      forecast_frames=f_frames, perception_frames=p_frames))
    return record


def check_legacy(gt, f, p, legacy_f, legacy_p, position):
    result = []
    for j, h in enumerate(HORIZONS):
        joint = joint_counts(gt[j], f[j], p[j])
        for name, hist, old in [('F', joint.sum(2), legacy_f),
                                 ('P', joint.sum(1), legacy_p)]:
            if not np.array_equal(hist, old['hist'][position, j]):
                diff = np.abs(hist - old['hist'][position, j])
                raise RuntimeError(f'Legacy replay mismatch {name}, row={position}, h={h}, '
                                   f'hist L1={diff.sum()}, max={diff.max()}')
            intersection = hist[:17, :17].sum()
            union = hist.sum() - hist[17, 17]
            assert intersection == old['intersection'][position, j]
            assert union == old['union'][position, j]
        result.append(joint)
    return np.stack(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT)
    parser.add_argument('--config', default='config/opus/future2current_feature_identity.py')
    parser.add_argument('--limit', type=int, default=512)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(4)
    cfg = Config.fromfile(args.config)
    source_manifest_path = Path('out/future2current/feature_identity_2048x512/manifest.json')
    source_manifest = json.loads(source_manifest_path.read_text())
    validation = source_manifest['val']
    checkpoint_hash = sha256(cfg.student_checkpoint)
    assert checkpoint_hash == cfg.student_checkpoint_sha256 == source_manifest['checkpoint_sha256']
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'pairs').mkdir(exist_ok=True)
    loader, selected = loader_for(cfg, 'val', 512, args.workers, cfg.identity_sample_seed)
    assert selected == validation
    ds = loader.dataset.dataset
    records = [metadata(ds, i, index, scene) for i, (index, scene) in
               enumerate(zip(validation['indices'], validation['scenes']))]
    scenes = sorted(set(validation['scenes']), key=lambda s:
                    __import__('hashlib').sha256(('20261008:' + s).encode()).hexdigest())
    write_json(args.output_dir / 'scene_split.json', dict(seed=20261008, discovery=scenes[:100],
               holdout=scenes[100:], caveat='Analytical holdout only; all 150 scenes used in earlier experiments'))
    manifest = dict(samples=records, source_manifest=str(source_manifest_path),
                    source_manifest_sha256=sha256(source_manifest_path), eligible_samples=len(ds))
    write_json(args.output_dir / 'manifest.json', manifest)
    protocol = dict(version=1, checkpoint=str(cfg.student_checkpoint), checkpoint_sha256=checkpoint_hash,
                    config=args.config, config_sha256=sha256(args.config),
                    strict_config=cfg.student_model_config,
                    strict_config_sha256=sha256(cfg.student_model_config),
                    dataset_annotation=cfg.val_dataset_config.imageset,
                    dataset_annotation_sha256=sha256(cfg.val_dataset_config.imageset),
                    horizons=list(HORIZONS), grid_shape=[200, 200, 16], grid_size=.4,
                    pc_range=[-40, -40, -1, 40, 40, 5.4], mask='none', free_label=17,
                    axis_order='xyz', coordinate='native_future', python=platform.python_version(),
                    torch=torch.__version__, cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(0), precision='fp32, tf32 disabled',
                    teacher_ego='Cached current plan_head ego; no target ego state input',
                    score_threshold=[.35]*15+[.25, .3], padding=True,
                    code_sha256={p: sha256(p) for p in [__file__,
                        'model/segmentor/sparseworld_strict_segmentor.py',
                        'model/head/sparseworld_strict_head.py']})
    protocol_path = args.output_dir / 'protocol.json'
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text()) == protocol, 'Output protocol has changed'
    else:
        write_json(protocol_path, protocol)
    protocol_hash = sha256(protocol_path)
    legacy_root = Path('out/future2current/native_2048x512')
    legacy_f = dict(np.load(legacy_root / 'baseline_replay.npz'))
    legacy_p = dict(np.load(legacy_root / 'teacher.npz'))
    sources = paths_for(Path(source_manifest['source_cache']), 'val', validation)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = build_student(cfg, Config.fromfile(cfg.student_model_config))
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    missing = [i for i in range(min(args.limit, 512)) if not
               (args.output_dir / 'pairs' / (records[i]['token'] + '.npz')).exists()]
    selected_dataset_indices = [validation['indices'][i] for i in missing]
    images = DataLoader(Subset(ds, selected_dataset_indices), batch_size=1, num_workers=args.workers,
                        collate_fn=custom_collate_fn_temporal, pin_memory=True)
    timing_path = args.output_dir / 'timing.json'
    timings = json.loads(timing_path.read_text()) if timing_path.exists() else []
    with torch.no_grad():
        for position, batch in zip(missing, images):
            start = time.monotonic()
            source = torch.load(sources[position], map_location='cpu', weights_only=False)
            record = records[position]
            assert (source['index'], source['scene']) == (record['index'], record['scene'])
            token = batch['sample_idx'][0]
            assert token == record['token'], (token, record['token'])
            data = tree_to(batch, 'cuda')
            imgs = data.pop('img')
            gt = source['metas']['future_occ_labels'][0, [h-1 for h in HORIZONS]].numpy()
            camera = source['metas']['future_occ_cam_masks'][0, [h-1 for h in HORIZONS]].numpy()
            lidar = source['metas']['future_occ_lidar_masks'][0, [h-1 for h in HORIZONS]].numpy()
            assert np.array_equal(gt, data['future_occ_labels'][0, [h-1 for h in HORIZONS]].cpu().numpy())
            predicted = baseline(model, source)
            f = np.stack([model.head.rasterizer(predicted['points'][h-1].float(),
                            predicted['logits'][h-1].float(), 48).view(200, 200, 16).cpu().numpy()
                          for h in HORIZONS]).astype(np.uint8)
            del predicted
            p, f_projection, p_projection, f_wh, p_wh = [], [], [], [], []
            for h in HORIZONS:
                selected_imgs, metas = _future_window(data, imgs, h)
                state = dict(imgs=selected_imgs, metas=metas, points=None)
                state.update(model.extract_img_feat(**state))
                state.update(model.lifter(**state))
                scale = (torch.tanh(model.points_scale(source['ego'].cuda())) + 1.) * .35 + .8
                state['query_points'] = state['query_points'] * scale
                state.update(model.encoder(**state))
                state.update(model.head(**state))
                p.append(state['final_occ_grid'][0].cpu().numpy().astype(np.uint8))
                transform = data['future_ego_to_current'][:, h-1]
                f_projection.append((data['projection_mat'][:, :30] @ transform[:, None])[0].cpu().numpy())
                p_projection.append(metas['projection_mat'][0].cpu().numpy())
                f_wh.append(data['image_wh'][0, :30].cpu().numpy())
                p_wh.append(metas['image_wh'][0].cpu().numpy())
                del state, selected_imgs, metas
            p = np.stack(p)
            check_legacy(gt, f, p, legacy_f, legacy_p, position)
            dest = args.output_dir / 'pairs' / (record['token'] + '.npz')
            tmp = dest.with_suffix('.tmp.npz')
            np.savez_compressed(tmp, F=f, P=p, GT=gt.astype(np.uint8), camera=camera, lidar=lidar,
                                F_projection=np.stack(f_projection), P_projection=np.stack(p_projection),
                                F_wh=np.stack(f_wh), P_wh=np.stack(p_wh),
                                protocol_sha256=np.asarray(protocol_hash), token=np.asarray(record['token']))
            tmp.replace(dest)
            torch.cuda.synchronize()
            elapsed = time.monotonic() - start
            timings.append(dict(position=position, seconds=elapsed, bytes=dest.stat().st_size,
                                peak_cuda_bytes=torch.cuda.max_memory_allocated()))
            write_json(timing_path, timings)
            print(f'[paired export] {position+1}/512 exact replay OK, {elapsed:.2f}s, '
                  f'{dest.stat().st_size/1e6:.2f} MB', flush=True)
            del source, batch, data, imgs, f, p
    verified = []
    for i, rec in enumerate(records[:args.limit]):
        path = args.output_dir / 'pairs' / (rec['token'] + '.npz')
        with np.load(path) as d:
            assert str(d['protocol_sha256']) == protocol_hash
            check_legacy(d['GT'], d['F'], d['P'], legacy_f, legacy_p, i)
        verified.append(i)
    write_json(args.output_dir / 'replay_audit.json', dict(status='complete' if len(verified)==512 else 'pilot_complete',
               samples=len(verified), exact_semantic_histogram_match=True,
               exact_occupancy_intersection_union_match=True, positions=verified,
               legacy_forecast=str(legacy_root/'baseline_replay.npz'),
               legacy_perception=str(legacy_root/'teacher.npz')))
    print('[paired export] verified', len(verified), 'samples', flush=True)


if __name__ == '__main__':
    main()
