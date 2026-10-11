#!/usr/bin/env python3
"""Inspect whether response KD mostly sees a few easy teacher classes."""

import argparse
import json
from pathlib import Path

import torch


def main(args):
    root = Path(args.output_dir)
    manifest = json.loads((root / 'manifest.json').read_text())['train']
    old = Path('out/future_alignment_hypotheses/teacher_query_cache_512x256')
    result = {}
    for h in (2, 4, 6):
        counts = torch.zeros(17, dtype=torch.long)
        candidates = []
        selected = []
        for index in manifest['indices'][:args.samples]:
            source = torch.load(root / 'source_cache' / 'train' / f'{index:06d}.pth', map_location='cpu')
            teacher_path = old / 'train' / f'{index:06d}.pth'
            if not teacher_path.exists():
                teacher_path = root / 'teacher_cache' / 'train' / f'{index:06d}.pth'
            teacher = torch.load(teacher_path, map_location='cpu')['states'][h]
            points = teacher['query_points_metric'].reshape(-1, 3).float()
            score, classes = teacher['point_logits'].reshape(-1, 17).float().sigmoid().max(-1)
            voxel = torch.floor((points - points.new_tensor([-40., -40., -1.])) / .4).long()
            inside = ((voxel >= 0) & (voxel < voxel.new_tensor([200, 200, 16]))).all(-1)
            ground = source['metas']['future_occ_labels'][0, h-1].long()
            true_class = torch.full_like(classes, 17)
            valid = inside.nonzero().flatten()
            true_class[valid] = ground[voxel[valid, 0], voxel[valid, 1], voxel[valid, 2]]
            keep = ((score >= args.confidence) & (true_class == classes)).nonzero().flatten()
            candidates.append(len(keep))
            if len(keep) > args.limit:
                keep = keep[score[keep].topk(args.limit).indices]
            selected.append(len(keep))
            counts += torch.bincount(classes[keep], minlength=17)
        result[f'{h*.5:g}s'] = dict(
            samples=args.samples, mean_candidates=sum(candidates) / args.samples,
            mean_selected=sum(selected) / args.samples,
            selected_class_counts=counts.tolist(),
            selected_class_fraction=(counts / counts.sum()).tolist())
    destination = root / 'teacher_selection_audit.json'
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='out/future_distillation_response_based/2048x512')
    parser.add_argument('--samples', type=int, default=128)
    parser.add_argument('--confidence', type=float, default=.55)
    parser.add_argument('--limit', type=int, default=2048)
    main(parser.parse_args())
