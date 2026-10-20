"""Camera metrics and the official OPUS RayIoU implementation, streamed per frame.

The official ray directions, raycaster, class/depth matching and ego-origin
selection are imported from an existing OPUS checkout. Origins use the complete
validation scene, even when evaluating the mini split. No camera mask is applied
to RayIoU. Source hashes and a streaming/official numerical check are recorded.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from tools.center_support_common import write_json
from tools.diagnose_opus_v1_gaussian import metrics, calibrated_predictions, file_sha256


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RayCounts:
    def __init__(self):
        self.gt = np.zeros(18, dtype=np.int64)
        self.pred = np.zeros(18, dtype=np.int64)
        self.tp = np.zeros((3, 18), dtype=np.int64)

    def add(self, pred, gt):
        self.gt += np.bincount(gt[:, 0].astype(np.int64), minlength=18)
        self.pred += np.bincount(pred[:, 0].astype(np.int64), minlength=18)
        for j, radius in enumerate((1, 2, 4)):
            good = (pred[:, 0] == gt[:, 0]) & (np.abs(pred[:, 1] - gt[:, 1]) < radius)
            self.tp[j] += np.bincount(gt[good, 0].astype(np.int64), minlength=18)

    def iou(self):
        with np.errstate(invalid='ignore', divide='ignore'):
            return (self.tp / (self.gt[None] + self.pred[None] - self.tp))[:, :17]


def official_pose_infos(scene_infos, keyframes):
    """Lossless pose adapter for this repository's scene-keyed annotations."""
    output = []
    for scene, index in sorted(keyframes, key=lambda x: (x[0], int(x[1]))):
        frame = scene_infos[scene][index]
        lidar = frame['data']['LIDAR_TOP']
        output.append(dict(token=frame['token'], scene_token=scene,
            ego2global_translation=lidar['pose']['translation'],
            ego2global_rotation=lidar['pose']['rotation'],
            lidar2ego_translation=lidar['calib']['translation'],
            lidar2ego_rotation=lidar['calib']['rotation']))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--py-config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--limit', type=int, default=0, help='0 means every validation frame')
    parser.add_argument('--empty-logit-shift', type=float, default=0.,
                        help='Explicit diagnostic shift after learned bias; main contribution-1 protocol uses zero')
    parser.add_argument('--official-opus-root', default='/home/emiya/data_disk/OPUS')
    parser.add_argument('--ray-cache', default='work_dirs/center_support_contribution1/ray_gt_cache')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; no evaluation written')
    if args.limit < 0:
        parser.error('--limit must be nonnegative')
    if not np.isfinite(args.empty_logit_shift):
        parser.error('--empty-logit-shift must be finite')
    from mmengine import Config
    from mmseg.models import build_segmentor
    import model  # noqa: F401
    from dataset import get_dataloader
    import mmengine

    cfg = Config.fromfile(args.py_config)
    if cfg.model.head.get('empty_logit_offset', 0):
        raise ValueError('No post-hoc decision shifts allowed')
    cfg.val_loader.batch_size, cfg.val_loader.num_workers = 1, 4
    _, loader = get_dataloader(cfg.train_dataset_config, cfg.val_dataset_config,
                              cfg.train_loader, cfg.val_loader, val_only=True, seed=928)
    network = build_segmentor(cfg.model).cuda().eval()
    network.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=False)['state_dict'], strict=True)
    torch.backends.cudnn.benchmark = False
    official = Path(args.official_opus_root).resolve()
    source_files = ['loaders/ray_metrics.py', 'loaders/ego_pose_dataset.py', 'lib/dvr/dvr.cpp', 'lib/dvr/dvr.cu']
    source_hashes = {p: hashlib.sha256((official / p).read_bytes()).hexdigest() for p in source_files}
    # JIT compilation stays inside the workspace, including in sandboxed runs.
    os.environ.setdefault('TORCH_EXTENSIONS_DIR', str(ROOT / 'work_dirs/center_support_contribution1/torch_extensions'))
    os.environ.setdefault('MAX_JOBS', '2')
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.9')
    previous_cwd = os.getcwd()
    try:
        os.chdir(official)
        ray = import_file('official_center_support_ray', official / 'loaders/ray_metrics.py')
        poses = import_file('official_center_support_poses', official / 'loaders/ego_pose_dataset.py')
    finally:
        os.chdir(previous_cwd)
    # Full scene poses are essential: a mini split cannot define its own origins.
    full_path = ROOT / 'data/nuscenes_cam/nuscenes_infos_val_sweeps_occ.pkl'
    full = mmengine.load(full_path)
    pose_dataset = poses.EgoPoseDataset(official_pose_infos(full['infos'], full['metadata']))
    pose_index = {info['token']: i for i, info in enumerate(pose_dataset.data_infos)}
    cache_key = hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()
                               + str(full_path.stat().st_mtime_ns).encode()).hexdigest()[:16]
    cache = Path(args.ray_cache) / cache_key
    cache.mkdir(parents=True, exist_ok=True)
    lidar_rays = torch.from_numpy(ray.generate_lidar_rays())
    counts, first_counts = RayCounts(), RayCounts()
    first_pred, first_gt = [], []
    confusion = torch.zeros(18, 18, dtype=torch.float64)
    attributes = dict(gaussians=0, scale_saturated_axes=0, scale_saturated_gaussians=0,
                      opacity_sum=0., frames=0)
    started = time.monotonic()
    parity = False
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if args.limit and index >= args.limit:
                break
            scene, frame_index = loader.dataset.keyframes[index]
            frame = loader.dataset.scene_infos[scene][frame_index]
            token = frame['token']
            batch = {key: value.cuda(non_blocking=True) if isinstance(value, torch.Tensor) else value
                     for key, value in batch.items()}
            output = network(imgs=batch['img'], metas={k: v for k, v in batch.items() if k != 'img'})
            labels = output['sampled_label'][0].long().reshape(-1)
            logits = output['pred_occ'][-1][0]
            if args.empty_logit_shift:
                fg_max, fg_class = logits[:17].max(0)
                prediction = calibrated_predictions(fg_max, fg_class, logits[17], args.empty_logit_shift)
            else:
                prediction = logits.argmax(0)
            if labels.numel() != 200 * 200 * 16:
                raise ValueError('RayIoU requires the entire dense grid')
            valid = (labels >= 0) & (labels < 18)
            if not bool(valid.all()):
                raise ValueError('Unknown GT in official RayIoU input; explicit upstream protocol required')
            mask = batch['occ_cam_mask'][0].bool().reshape(-1) & valid
            confusion += torch.bincount(labels[mask] * 18 + prediction[mask], minlength=324).reshape(18, 18).cpu()
            pred_dense = prediction.reshape(200, 200, 16).cpu().numpy()
            gt_dense = labels.reshape(200, 200, 16).cpu().numpy()
            _, origins = pose_dataset[pose_index[token]]
            origins = origins[None]
            cached = cache / (token + '.npz')
            gt_sha256 = hashlib.sha256(gt_dense.tobytes()).hexdigest()
            if cached.exists():
                with np.load(cached) as saved:
                    gt_ray = saved['rays']
                    if not np.array_equal(saved['origins'], origins.numpy()):
                        raise ValueError('Ray cache origin mismatch')
                    if 'gt_sha256' in saved and str(saved['gt_sha256']) != gt_sha256:
                        raise ValueError('Ray cache GT mismatch')
            else:
                gt_ray = ray.process_one_sample(gt_dense, lidar_rays, origins)
                np.savez_compressed(cached, rays=gt_ray, origins=origins.numpy(), gt_sha256=gt_sha256)
            pred_ray = ray.process_one_sample(pred_dense, lidar_rays, origins)
            ray_mask = gt_ray[:, 0].astype(np.int32) != 17
            pred_ray, gt_ray = pred_ray[ray_mask], gt_ray[ray_mask]
            counts.add(pred_ray, gt_ray)
            if index < 8:
                first_pred.append(pred_ray)
                first_gt.append(gt_ray)
                first_counts.add(pred_ray, gt_ray)
                expected = np.asarray(ray.calc_metrics(first_pred, first_gt))
                if not np.allclose(expected, first_counts.iou(), rtol=0, atol=0, equal_nan=True):
                    raise RuntimeError('Streaming RayIoU differs from official calc_metrics')
                parity = True
            gaussian = output['gaussian']
            scales = gaussian.scales.reshape(-1, 3)
            attributes['gaussians'] += len(scales)
            attributes['scale_saturated_axes'] += int((scales >= .396).sum())
            attributes['scale_saturated_gaussians'] += int((scales >= .396).any(-1).sum())
            attributes['opacity_sum'] += float(gaussian.opacities.sum())
            attributes['frames'] += 1
            if (index + 1) % 100 == 0:
                print(f'evaluated {index + 1}/{len(loader)} frames', flush=True)
    if not attributes['frames'] or not parity:
        raise RuntimeError('No evaluation frames')
    iou = counts.iou()
    commit = subprocess.check_output(['git', '-C', str(official), 'rev-parse', 'HEAD'], text=True).strip()
    provenance_diff = subprocess.check_output(['git', '-C', str(official), 'diff', '--', *source_files], text=True)
    result = dict(status='complete', protocol=vars(args), samples=attributes['frames'],
        checkpoint_sha256=file_sha256(args.checkpoint),
        camera_metrics=metrics(confusion), camera_confusion=confusion.tolist(),
        per_class_intersection=confusion.diag()[:17].tolist(),
        per_class_union=(confusion.sum(0) + confusion.sum(1) - confusion.diag())[:17].tolist(),
        empty_false_positive_fraction=float(confusion[17, :17].sum() / confusion[17].sum().clamp_min(1)),
        gaussian_statistics=dict(scale_ge_0p396_axis_fraction=attributes['scale_saturated_axes'] / (3 * attributes['gaussians']),
            scale_ge_0p396_gaussian_fraction=attributes['scale_saturated_gaussians'] / attributes['gaussians'],
            opacity_mean=attributes['opacity_sum'] / attributes['gaussians']),
        ray_metrics={'RayIoU': float(np.nanmean(iou) * 100),
                     **{f'RayIoU@{r}': float(np.nanmean(iou[j]) * 100) for j, r in enumerate((1, 2, 4))}},
        ray_per_class=[{name: float(value * 100) if np.isfinite(value) else None
                       for name, value in zip(ray.occ_class_names[:17], row)} for row in iou],
        ray_counts=dict(gt=counts.gt.tolist(), predicted=counts.pred.tolist(), true_positive=counts.tp.tolist()),
        ray_protocol='Official OPUS rays and full-validation-scene ego origins; no camera mask; GT non-free rays; thresholds 1/2/4 m',
        official_source=dict(root=str(official), commit=commit, sha256=source_hashes, compatibility_diff=provenance_diff),
        streaming_official_parity_passed=parity, elapsed_seconds=time.monotonic() - started,
        peak_cuda_allocated_mib=torch.cuda.max_memory_allocated() / 2**20)
    write_json(args.out, result)
    print(json.dumps(dict(camera=result['camera_metrics'], ray=result['ray_metrics'], samples=result['samples']), indent=2))


if __name__ == '__main__':
    main()
