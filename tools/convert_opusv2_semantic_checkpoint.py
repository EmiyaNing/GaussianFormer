#!/usr/bin/env python3
"""Initialize an OPUSv2 Semantic Gaussian model from official OPUSv2 weights."""
import argparse
import os
import sys

import torch
from mmengine import Config

try:
    # Package import used by tests and library callers.
    from tools.convert_opusv2_checkpoint import MAPPING_VERSION, map_key, sha256
except ModuleNotFoundError:
    # Direct ``python tools/...py`` execution puts this directory on sys.path.
    from convert_opusv2_checkpoint import MAPPING_VERSION, map_key, sha256


SEMANTIC_MAPPING_VERSION = 'official-opusv2-semantic-init-v1'
GAUSSIAN_PREFIX = 'head.semantic_gaussians.'


def convert_semantic_state_dict(source, target):
    """Strictly map every official key and fill only new Gaussian parameters."""
    converted = {}
    for source_key, value in source.items():
        target_key = map_key(source_key)
        if target_key in converted:
            raise RuntimeError(f'key collision while mapping {source_key!r}')
        converted[target_key] = value
    unexpected = sorted(set(converted) - set(target))
    missing = sorted(set(target) - set(converted))
    illegal_missing = [key for key in missing if not key.startswith(GAUSSIAN_PREFIX)]
    mismatched = sorted(
        (key, tuple(converted[key].shape), tuple(target[key].shape))
        for key in set(converted) & set(target)
        if converted[key].shape != target[key].shape)
    if unexpected or illegal_missing or mismatched:
        raise RuntimeError(
            'OPUSv2 Semantic mapping failed:\n'
            f'  illegal missing ({len(illegal_missing)}): {illegal_missing[:20]}\n'
            f'  unexpected ({len(unexpected)}): {unexpected[:20]}\n'
            f'  shape mismatch ({len(mismatched)}): {mismatched[:20]}')
    # Materialize a complete checkpoint so every later load remains strict.
    for key in missing:
        converted[key] = target[key].clone()
    return converted, missing


def build_model(config_path):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    import model  # noqa: F401
    from mmseg.models import build_segmentor
    cfg = Config.fromfile(config_path)
    if cfg.get('checkpoint_mapping') != 'official_opusv2_semantic':
        raise ValueError(
            'config must declare checkpoint_mapping="official_opusv2_semantic"')
    model = build_segmentor(cfg.model)
    model.init_weights()
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--src', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--dst')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    if not args.dry_run and not args.dst:
        parser.error('--dst is required unless --dry-run is used')
    if args.dst and os.path.exists(args.dst) and not args.force:
        raise FileExistsError(f'refusing to overwrite {args.dst}; pass --force explicitly')

    checkpoint = torch.load(args.src, map_location='cpu')
    source = checkpoint.get('state_dict', checkpoint)
    target_model = build_model(args.config)
    converted, initialized = convert_semantic_state_dict(
        source, target_model.state_dict())
    target_model.load_state_dict(converted, strict=True)
    source_hash = sha256(args.src)
    print(f'mapped={len(source)} gaussian_initialized={len(initialized)} '
          'missing=0 unexpected=0 shape_mismatch=0')
    if args.dry_run:
        return

    meta = dict(checkpoint.get('meta', {})) if isinstance(checkpoint, dict) else {}
    meta.update(
        opusv2_mapping_version=MAPPING_VERSION,
        opusv2_semantic_mapping_version=SEMANTIC_MAPPING_VERSION,
        opusv2_source_sha256=source_hash,
        opusv2_source=os.path.basename(args.src),
        opusv2_config=os.path.abspath(args.config),
        model_family='opusv2_semantic_gaussian',
        semantic_gaussian_version=1)
    output = {'meta': meta, 'state_dict': converted}
    os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)
    torch.save(output, args.dst)
    print(f'wrote {args.dst}')


if __name__ == '__main__':
    main()
