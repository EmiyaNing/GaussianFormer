#!/usr/bin/env python3
"""Convert and prove a SparseWorld checkpoint for SparseWorldStrict."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from mmengine import Config
from mmseg.models import build_segmentor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def tensor_sha256(tensor):
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/opus/sparseworld_strict.py')
    parser.add_argument('--src', required=True, help='official SparseWorld checkpoint')
    parser.add_argument('--dst', required=True, help='converted checkpoint')
    parser.add_argument('--mapping', default='config/opus/sparseworld_official_keymap.json',
                        help='JSON object of exact source-key to target-key mappings')
    parser.add_argument('--report', default='', help='audit JSON (default: DST.audit.json)')
    return parser.parse_args()


def main():
    args = parse_args()
    source_path, destination_path = Path(args.src), Path(args.dst)
    mapping_path = Path(args.mapping)
    explicit = json.loads(mapping_path.read_text()) if mapping_path.is_file() else {}
    if not isinstance(explicit, dict) or not all(isinstance(k, str) and isinstance(v, str)
                                                for k, v in explicit.items()):
        raise TypeError('mapping JSON must be an object of exact key strings')

    checkpoint = torch.load(source_path, map_location='cpu', weights_only=False)
    source = checkpoint.get('state_dict', checkpoint)
    wrapped = [key.startswith('module.') for key in source]
    if any(wrapped) and not all(wrapped):
        raise RuntimeError('mixed module. checkpoint prefixes are not allowed')
    strip_wrapper = bool(wrapped and all(wrapped))
    mapped, applied = {}, {}
    for source_key, tensor in source.items():
        canonical = source_key[7:] if strip_wrapper else source_key
        target_key = explicit.get(source_key, explicit.get(canonical, canonical))
        if target_key in mapped:
            raise KeyError(f'multiple source keys map to {target_key}')
        mapped[target_key] = tensor
        if target_key != source_key:
            applied[source_key] = target_key

    # Import registrations only after arguments are valid.
    import model  # noqa: F401
    config = Config.fromfile(args.config)
    network = build_segmentor(config.model)
    target = network.state_dict()
    missing = sorted(set(target) - set(mapped))
    unexpected = sorted(set(mapped) - set(target))
    shape_mismatch = {
        key: {'source': list(mapped[key].shape), 'target': list(target[key].shape)}
        for key in sorted(set(mapped) & set(target))
        if mapped[key].shape != target[key].shape}
    dtype_mismatch = {
        key: {'source': str(mapped[key].dtype), 'target': str(target[key].dtype)}
        for key in sorted(set(mapped) & set(target))
        if mapped[key].dtype != target[key].dtype}
    if missing or unexpected or shape_mismatch or dtype_mismatch:
        raise RuntimeError(json.dumps(dict(
            missing=missing, unexpected=unexpected,
            shape_mismatch=shape_mismatch, dtype_mismatch=dtype_mismatch), indent=2))
    network.load_state_dict(mapped, strict=True)

    manifest = {key: tensor_sha256(mapped[key]) for key in sorted(mapped)}
    source_file_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    output = {'state_dict': mapped,
              'meta': dict(checkpoint.get('meta', {})) if isinstance(checkpoint, dict) else {}}
    output['meta']['sparseworld_strict_conversion'] = dict(
        source=str(source_path.resolve()), source_sha256=source_file_hash,
        config=str(Path(args.config).resolve()), tensor_count=len(mapped),
        element_count=sum(value.numel() for value in mapped.values()))
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, destination_path)

    # Re-open the artifact and perform the same strict proof on serialized data.
    serialized = torch.load(destination_path, map_location='cpu', weights_only=False)
    network.load_state_dict(serialized['state_dict'], strict=True)
    output_manifest = {key: tensor_sha256(value)
                       for key, value in serialized['state_dict'].items()}
    if output_manifest != manifest:
        raise RuntimeError('serialized checkpoint tensor hashes changed')

    report = dict(
        source=str(source_path.resolve()), destination=str(destination_path.resolve()),
        source_sha256=source_file_hash, tensor_count=len(mapped),
        source_tensor_count=len(source), target_tensor_count=len(target),
        mapped_tensor_count=len(mapped), collisions=[],
        element_count=sum(value.numel() for value in mapped.values()),
        applied_mapping=applied, missing_keys=[], unexpected_keys=[],
        shape_mismatch={}, dtype_mismatch={}, strict_load=True,
        tensor_sha256=manifest)
    report_path = Path(args.report) if args.report else destination_path.with_suffix(
        destination_path.suffix + '.audit.json')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(f'Converted {len(mapped)} tensors ({report["element_count"]} elements).')
    print(f'Checkpoint: {destination_path}')
    print(f'Audit: {report_path}')
    print('Strict load: PASS')


if __name__ == '__main__':
    main()
