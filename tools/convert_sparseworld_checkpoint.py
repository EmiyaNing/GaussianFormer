#!/usr/bin/env python3
"""Convert the released SparseWorld checkpoint to the strict local port.

The converter builds the target config and refuses to write a checkpoint unless
every target tensor is accounted for and every non-runtime source tensor is
either mapped or explicitly documented as intentionally unused.
"""
import argparse
import hashlib
from pathlib import Path
import sys

import torch
from mmengine import Config
from mmseg.models import build_segmentor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _assigned_stamps(statistics, capacities):
    ranking = statistics.float()
    ranking = ranking / ranking.sum(-1, keepdim=True).clamp_min(1.)
    ranking = ranking.clone()
    assigned = torch.full((ranking.shape[0],), -1, dtype=torch.long)
    for _ in range(ranking.shape[0]):
        index = ranking.amax(-1).argmax()
        stamp = ranking[index].argmax()
        assigned[index] = stamp
        ranking[index].fill_(-float('inf'))
        if int((assigned == stamp).sum()) == capacities[stamp]:
            ranking[:, stamp] = -float('inf')
    if (assigned < 0).any():
        raise RuntimeError('failed to assign every SparseWorld query timestamp')
    return assigned


def _map_key(key):
    if key.startswith(('img_backbone.', 'img_neck.', 'plan_head.')):
        return key
    top_level = {
        'points_scale_branch.': 'points_scale.',
        'position_encoder.': 'position.',
        'reg_branch.': 'reg.',
        'vel_branch.': 'velocity.',
        'cls_branch.': 'semantic.',
        'traj_head.': 'traj.',
        'ego_cross_attn.attention.attn.': 'ego_attention.attention.',
        'ego_cross_attn.gen_tau.': 'ego_attention.tau.',
        'ego_cross_attn.pos_embed.': 'ego_attention.position.',
    }
    for source, target in top_level.items():
        if key.startswith(source):
            return target + key[len(source):]
    if key == 'pts_bbox_head.init_points.weight':
        return 'lifter.reference_points'
    if key == 'pts_bbox_head.num_stamps_all':
        return 'lifter.stamp_statistics'
    prefix = 'pts_bbox_head.transformer.decoder.decoder_layers.'
    if not key.startswith(prefix):
        return None
    suffix = key[len(prefix):]
    replacements = (
        ('.position_encoder.', '.position.'),
        ('.self_attn.attention.attn.', '.self_attn.attention.'),
        ('.self_attn.gen_tau.', '.self_attn.tau.'),
        ('.mixing.parameter_generator.', '.mixing.generator.'),
        ('.ffn.layers.0.0.', '.ffn.0.'),
        ('.ffn.layers.1.', '.ffn.3.'),
        ('.cls_branch.', '.cls.'),
        ('.reg_branch.', '.reg.'),
    )
    for source, target in replacements:
        suffix = suffix.replace(source, target)
    return 'encoder.layers.' + suffix


def convert(source_path, config_path):
    import model  # noqa: F401 - register local model components

    cfg = Config.fromfile(str(config_path))
    target_model = build_segmentor(cfg.model)
    target = target_model.state_dict()
    checkpoint = torch.load(source_path, map_location='cpu')
    source = checkpoint.get('state_dict', checkpoint)
    source = {key.removeprefix('module.'): value for key, value in source.items()}

    converted, used = {}, set()
    for source_key, value in source.items():
        target_key = _map_key(source_key)
        if target_key is None:
            continue
        if target_key not in target:
            raise KeyError(f'mapped target key does not exist: {source_key} -> {target_key}')
        if value.shape != target[target_key].shape:
            raise ValueError(
                f'shape mismatch: {source_key} {tuple(value.shape)} -> '
                f'{target_key} {tuple(target[target_key].shape)}')
        converted[target_key] = value.to(dtype=target[target_key].dtype)
        used.add(source_key)

    # These buffers are deterministic runtime state, not learned parameters.
    converted['lifter.query_features'] = target['lifter.query_features'].clone()
    statistics = converted['lifter.stamp_statistics']
    capacities = (target_model.lifter.num_queries,) + target_model.lifter.future_queries
    converted['lifter.query_stamps'] = _assigned_stamps(statistics, capacities)

    missing = sorted(set(target) - set(converted))
    if missing:
        raise RuntimeError('unmapped target tensors:\n' + '\n'.join(missing))
    allowed_unused = {
        'pts_bbox_head.pc_range', 'pts_bbox_head.scene_size',
        'pts_bbox_head.voxel_size', 'pts_bbox_head.voxel_num',
        'semantic_loss.weight',
    }
    unused = sorted(set(source) - used - allowed_unused)
    if unused:
        raise RuntimeError('unexpected unused source tensors:\n' + '\n'.join(unused))
    target_model.load_state_dict(converted, strict=True)
    return checkpoint, converted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--config', type=Path, default=Path(
        'config/opus/sparseworld_strict_r50_704x256_5f_occ3d.py'))
    args = parser.parse_args()
    checkpoint, state_dict = convert(args.source, args.config)
    output = {
        'state_dict': state_dict,
        'meta': {
            'checkpoint_format': 'gaussianformer_sparseworld_strict_v1',
            'sparseworld_source_sha256': _sha256(args.source),
            'sparseworld_source_meta': checkpoint.get('meta', {}),
            'config': str(args.config),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(f'wrote {args.output} with {len(state_dict)} tensors')
    print(f'source sha256: {output["meta"]["sparseworld_source_sha256"]}')


if __name__ == '__main__':
    main()
