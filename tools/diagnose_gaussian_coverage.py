"""Nearest-center coverage of camera-visible Occ3D voxels by predicted Gaussians."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from mmengine import Config
from mmseg.models import build_segmentor
from scipy.spatial import cKDTree

import model  # noqa: F401
from dataset import get_dataloader
from summarize_mini_occ3d import CLASSES


def evenly_sample(indices, limit):
    if indices.size <= limit:
        return indices
    return indices[np.linspace(0, indices.size - 1, limit, dtype=np.int64)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--py-config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--limit', type=int, default=64)
    parser.add_argument('--voxels-per-class', type=int, default=2000)
    args = parser.parse_args()
    cfg = Config.fromfile(args.py_config)
    cfg.val_loader.batch_size = 1
    cfg.val_loader.num_workers = 4
    _, loader = get_dataloader(cfg.train_dataset_config, cfg.val_dataset_config,
                               cfg.train_loader, cfg.val_loader, val_only=True)
    network = build_segmentor(cfg.model).cuda().eval()
    state = torch.load(args.checkpoint, map_location='cpu')['state_dict']
    network.load_state_dict(state, strict=True)
    counts = {name: 0 for name in ('occupied', 'empty', 'active_gaussian', 'gaussian')}
    sums = {name: 0 for name in ('occupied_d04', 'occupied_d08', 'occupied_d12',
                                'occupied_active_d12', 'empty_d12')}
    per_class = {name: {'total': 0, 'same_class_d12': 0}
                 for name in CLASSES}
    with torch.inference_mode():
        for sample_index, batch in enumerate(loader):
            if sample_index >= args.limit:
                break
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.cuda(non_blocking=True)
            output = network(imgs=batch.pop('img'), metas=batch)
            gaussian = output['gaussian']
            means = gaussian.means[0].cpu().numpy()
            opacity = gaussian.opacities[0, :, 0].cpu().numpy()
            pred_class = gaussian.semantics[0].argmax(dim=-1).cpu().numpy()
            xyz = output['sampled_xyz'][0].cpu().numpy().reshape(-1, 3)
            labels = output['sampled_label'][0].cpu().numpy().reshape(-1)
            camera = batch['occ_cam_mask'][0].cpu().numpy().reshape(-1).astype(bool)
            occupied_ids = evenly_sample(np.flatnonzero(camera & (labels != 17)),
                                         args.voxels_per_class)
            empty_ids = evenly_sample(np.flatnonzero(camera & (labels == 17)),
                                      args.voxels_per_class)
            tree = cKDTree(means)
            counts['gaussian'] += means.shape[0]
            counts['active_gaussian'] += int((opacity > 0.5).sum())
            if occupied_ids.size:
                distances = tree.query(xyz[occupied_ids], workers=1)[0]
                counts['occupied'] += occupied_ids.size
                for threshold, key in ((0.4, 'occupied_d04'),
                                       (0.8, 'occupied_d08'),
                                       (1.2, 'occupied_d12')):
                    sums[key] += int((distances <= threshold).sum())
                active_means = means[opacity > 0.5]
                if active_means.size:
                    active_distance = cKDTree(active_means).query(
                        xyz[occupied_ids], workers=1)[0]
                    sums['occupied_active_d12'] += int((active_distance <= 1.2).sum())
                for class_index, class_name in enumerate(CLASSES):
                    class_ids = occupied_ids[labels[occupied_ids] == class_index]
                    if not class_ids.size:
                        continue
                    per_class[class_name]['total'] += class_ids.size
                    class_means = means[pred_class == class_index]
                    if class_means.size:
                        class_dist = cKDTree(class_means).query(xyz[class_ids], workers=1)[0]
                        per_class[class_name]['same_class_d12'] += int((class_dist <= 1.2).sum())
            if empty_ids.size:
                empty_distance = tree.query(xyz[empty_ids], workers=1)[0]
                counts['empty'] += empty_ids.size
                sums['empty_d12'] += int((empty_distance <= 1.2).sum())
            if (sample_index + 1) % 16 == 0:
                print(f'processed {sample_index + 1}', flush=True)
    result = {
        'samples': min(args.limit, len(loader)),
        'checkpoint': args.checkpoint,
        'py_config': args.py_config,
        'max_sampled_occupied_voxels_per_scene': args.voxels_per_class,
        'counts': counts,
        'coverage': {key: value / max(counts['empty' if key == 'empty_d12' else 'occupied'], 1)
                     for key, value in sums.items()},
        'active_gaussian_fraction': counts['active_gaussian'] / max(counts['gaussian'], 1),
        'same_class_coverage_at_1p2m': {
            name: (value['same_class_d12'] / value['total'] if value['total'] else None)
            for name, value in per_class.items()},
        'per_class_counts': per_class,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(out)


if __name__ == '__main__':
    main()
