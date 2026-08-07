#!/usr/bin/env python3
"""Convert an official OPUSv2 checkpoint with a verified key bijection."""
import argparse
import hashlib
import os
import sys

import torch
from mmengine import Config


MAPPING_VERSION = 'official-opusv2-prefix-v1'


def map_key(key):
    prefix = 'pts_bbox_head.'
    return 'head.' + key[len(prefix):] if key.startswith(prefix) else key


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def convert_state_dict(source, target):
    converted = {}
    origins = {}
    for source_key, value in source.items():
        target_key = map_key(source_key)
        if target_key in converted:
            raise RuntimeError(
                f'key collision: {source_key!r} and {origins[target_key]!r} -> {target_key!r}')
        converted[target_key] = value
        origins[target_key] = source_key
    missing = sorted(set(target) - set(converted))
    unexpected = sorted(set(converted) - set(target))
    mismatched = sorted(
        (key, tuple(converted[key].shape), tuple(target[key].shape))
        for key in set(converted) & set(target)
        if converted[key].shape != target[key].shape)
    if missing or unexpected or mismatched:
        raise RuntimeError(
            'OPUSv2 mapping is not bijective:\n'
            f'  missing ({len(missing)}): {missing[:20]}\n'
            f'  unexpected ({len(unexpected)}): {unexpected[:20]}\n'
            f'  shape mismatch ({len(mismatched)}): {mismatched[:20]}')
    return converted


def build_model(config_path):
    # Ensure the project root takes precedence when invoked from any cwd.
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    import model  # noqa: F401
    from mmseg.models import build_segmentor
    cfg = Config.fromfile(config_path)
    if cfg.get('checkpoint_mapping') != 'official_opusv2':
        raise ValueError('config must declare checkpoint_mapping="official_opusv2"')
    return build_segmentor(cfg.model)


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
    converted = convert_state_dict(source, target_model.state_dict())
    # This is the authoritative final check; it must never become strict=False.
    target_model.load_state_dict(converted, strict=True)
    source_hash = sha256(args.src)
    print(f'mapped={len(converted)} missing=0 unexpected=0 shape_mismatch=0')
    print(f'source_sha256={source_hash}')
    if args.dry_run:
        return

    meta = dict(checkpoint.get('meta', {})) if isinstance(checkpoint, dict) else {}
    meta.update(opusv2_mapping_version=MAPPING_VERSION,
                opusv2_source_sha256=source_hash,
                opusv2_source=os.path.basename(args.src),
                opusv2_config=os.path.abspath(args.config))
    output = {'meta': meta, 'state_dict': converted}
    os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)
    torch.save(output, args.dst)
    print(f'wrote {args.dst}')


if __name__ == '__main__':
    main()
