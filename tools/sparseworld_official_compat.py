#!/usr/bin/env python3
"""Strict SparseWorld checkpoint conversion and official-runtime bridge.

This repository deliberately does not reimplement SparseWorld internals here:
an approximate reimplementation cannot satisfy bitwise parameter loading or
official-evaluation equivalence.  Instead this tool treats the sibling
official checkout as the compatibility runtime, preserves its model state
namespace, and proves that every tensor survives conversion unchanged.

The tool must be run with the Python environment used by ``../SparseWorld``.
It never uses ``strict=False`` and never filters a model-state key.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OFFICIAL_ROOT = REPO_ROOT.parent / 'SparseWorld'
DEFAULT_CONFIG = Path('configs/sparseworld/nuscenes-temporal/sparseworld-traj-finetune.py')


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on user environment
        raise RuntimeError(
            'PyTorch is unavailable. Run this tool in the official SparseWorld '
            'Python environment, not the system Python.') from exc
    return torch


def _state_dict(checkpoint: dict[str, Any]) -> dict[str, Any]:
    state = checkpoint.get('state_dict', checkpoint)
    if not isinstance(state, dict) or not state:
        raise ValueError('checkpoint must contain a non-empty state_dict mapping')
    if not all(hasattr(value, 'shape') for value in state.values()):
        raise ValueError('state_dict contains a non-tensor entry')
    # DataParallel prefix is a naming wrapper, not a learned tensor. It is
    # normalized only when it is present on every key; mixed prefixes fail.
    prefixed = [key.startswith('module.') for key in state]
    if any(prefixed) and not all(prefixed):
        raise ValueError('state_dict has mixed module. prefixes; refusing ambiguous conversion')
    return ({key[7:]: value for key, value in state.items()} if all(prefixed) else state)


def _load_key_map(path: str | None) -> list[tuple[str, str]]:
    """Read explicit, prefix-only checkpoint-name mappings.

    Prefix mapping is intentionally the only supported transformation.  It is
    sufficient for wrappers such as ``module.``/``model.`` and, unlike a
    regular-expression replacement, cannot silently rewrite unrelated keys.
    The JSON file must be an object of ``source_prefix: destination_prefix``.
    Longest source prefixes are applied first.
    """
    if path is None:
        return []
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict) or not all(isinstance(k, str) and isinstance(v, str)
                                            for k, v in raw.items()):
        raise ValueError('--key-map must be a JSON object of string prefixes')
    return sorted(raw.items(), key=lambda item: len(item[0]), reverse=True)


def _remap_state(state: dict[str, Any], mappings: list[tuple[str, str]]) -> tuple[dict[str, Any], dict[str, str]]:
    mapped, audit = {}, {}
    for old_key, value in state.items():
        new_key = old_key
        for source_prefix, destination_prefix in mappings:
            if old_key.startswith(source_prefix):
                new_key = destination_prefix + old_key[len(source_prefix):]
                break
        if new_key in mapped:
            raise ValueError(f'key mapping collision: {old_key!r} and {audit[new_key]!r} both map to {new_key!r}')
        mapped[new_key], audit[new_key] = value, old_key
    return mapped, audit


def _tensor_digest(tensor: Any) -> str:
    torch = _torch()
    value = tensor.detach().cpu().contiguous()
    # A checkpoint may contain scalar buffers (for example BatchNorm's
    # ``num_batches_tracked``). PyTorch forbids ``scalar.view(uint8)`` when
    # the element size changes, whereas flattening to [1] preserves precisely
    # the same storage bytes and works for every tensor rank.
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _manifest(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        key: {'shape': list(value.shape), 'dtype': str(value.dtype), 'sha256': _tensor_digest(value)}
        for key, value in sorted(state.items())
    }


def convert(args: argparse.Namespace) -> None:
    torch = _torch()
    source_path, output_path = Path(args.source).resolve(), Path(args.output).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if output_path.exists() and not args.force:
        raise FileExistsError(f'{output_path} exists; pass --force to replace it')
    source = torch.load(source_path, map_location='cpu')
    source_state = _state_dict(source)
    converted_state, key_audit = _remap_state(source_state, _load_key_map(args.key_map))
    source_manifest = _manifest(source_state)
    converted_manifest = _manifest(converted_state)
    # Preserve all checkpoint metadata/optimizer fields while replacing only
    # the normalized model state namespace. Every model tensor is cloned; no
    # truncation, casting, interpolation, or missing-key default is allowed.
    converted = dict(source) if isinstance(source, dict) else {}
    converted['state_dict'] = {key: value.detach().cpu().clone() for key, value in converted_state.items()}
    converted['sparseworld_official_compat'] = {
        'format': 1,
        'source': str(source_path),
        'model_tensor_count': len(converted_state),
        'source_manifest_sha256': hashlib.sha256(json.dumps(source_manifest, sort_keys=True).encode()).hexdigest(),
        'converted_manifest_sha256': hashlib.sha256(json.dumps(converted_manifest, sort_keys=True).encode()).hexdigest(),
        'key_map': key_audit,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, output_path)
    reloaded = _state_dict(torch.load(output_path, map_location='cpu'))
    output_manifest = _manifest(reloaded)
    if converted_manifest != output_manifest:
        raise RuntimeError('conversion integrity failure: tensor keys, metadata, or bytes changed')
    manifest_path = output_path.with_suffix(output_path.suffix + '.manifest.json')
    manifest_path.write_text(json.dumps(converted_manifest, indent=2, sort_keys=True) + '\n')
    audit_path = output_path.with_suffix(output_path.suffix + '.keymap.json')
    audit_path.write_text(json.dumps(key_audit, indent=2, sort_keys=True) + '\n')
    print(f'converted {len(converted_state)} tensors without loss: {output_path}')
    print(f'manifest: {manifest_path}')
    print(f'key mapping audit: {audit_path}')


def _official_imports(official_root: Path):
    if not (official_root / 'mmdet3d').is_dir():
        raise FileNotFoundError(f'not an official SparseWorld checkout: {official_root}')
    sys.path.insert(0, str(official_root))
    try:
        from mmcv import Config
    except ImportError as exc:
        try:
            import mmcv
            version = getattr(mmcv, '__version__', 'unknown')
        except ImportError:
            version = 'not installed'
        raise RuntimeError(
            'Strict SparseWorld verification requires the official OpenMMLab '
            f'environment (MMCV 1.x exposing mmcv.Config and mmcv.runner); '
            f'the active environment provides MMCV {version}. Do not use an '
            'MMEngine Config fallback because it would no longer validate the '
            'official model graph. Activate/install the environment described '
            f'in {official_root}/requirements.txt, then rerun this command.') from exc
    from mmdet3d.models import build_model
    from mmdet3d.utils import patch_config
    return Config, build_model, patch_config


def strict_check(args: argparse.Namespace) -> None:
    """Build the official model and require an exact state-dict match."""
    torch = _torch()
    official_root, checkpoint_path = Path(args.official_root).resolve(), Path(args.checkpoint).resolve()
    Config, build_model, patch_config = _official_imports(official_root)
    config_path = (official_root / args.config).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    cfg = patch_config(Config.fromfile(str(config_path)))
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    state = _state_dict(torch.load(checkpoint_path, map_location='cpu'))
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    shape_errors = [key for key in sorted(set(expected) & set(state))
                    if tuple(expected[key].shape) != tuple(state[key].shape)]
    dtype_errors = [key for key in sorted(set(expected) & set(state))
                    if expected[key].dtype != state[key].dtype]
    if missing or unexpected or shape_errors or dtype_errors:
        report = dict(missing=missing, unexpected=unexpected, shape_errors=shape_errors,
                      dtype_errors=dtype_errors)
        raise RuntimeError('strict compatibility audit failed:\n' + json.dumps(report, indent=2))
    model.load_state_dict(state, strict=True)
    print(f'strict load passed: {len(state)} tensors match the official model exactly')


def _run_in_official_python(args: argparse.Namespace, action: str) -> None:
    """Run an action in the legacy OpenMMLab interpreter, never in-process.

    The current GaussianFormer environment uses MMCV 2/MMDet 3 while the
    released SparseWorld graph is MMCV 1/MMDet 2.  Loading both in one Python
    interpreter is unsupported and can give a misleading partial load.
    """
    runtime = args.python or os.environ.get('SPARSEWORLD_OFFICIAL_PYTHON')
    if not runtime:
        raise RuntimeError('Set --python /path/to/official/python (or SPARSEWORLD_OFFICIAL_PYTHON).')
    runtime = str(Path(runtime).expanduser())
    if not Path(runtime).is_file():
        raise FileNotFoundError(f'official Python executable does not exist: {runtime}')
    command = [runtime, str(Path(__file__).resolve()), action,
               '--official-root', str(args.official_root), '--config', str(args.config)]
    if action == 'strict-check':
        command += ['--checkpoint', str(args.checkpoint), '--in-official-runtime']
    else:
        command += ['--checkpoint', str(args.checkpoint), '--in-official-runtime']
        command += list(args.passthrough)
    subprocess.run(command, check=True, env=os.environ.copy())


def run_official(args: argparse.Namespace, command: str) -> None:
    if not args.in_official_runtime:
        _run_in_official_python(args, command)
        return
    official_root = Path(args.official_root).resolve()
    # Evaluation always consumes a complete converted model checkpoint.
    # Training may intentionally start from the official configuration's
    # backbone initialization instead, in which case there is no model state
    # to verify at this boundary.
    if args.checkpoint:
        strict_check(args)
    config_path = str((official_root / args.config).resolve())
    command_path = official_root / 'tools' / ('test_temporal.py' if command == 'eval' else 'train.py')
    if command == 'eval':
        invocation = [sys.executable, str(command_path), config_path, str(Path(args.checkpoint).resolve()), '--eval', 'segm']
        invocation.extend(args.passthrough)
    else:
        invocation = [sys.executable, str(command_path), config_path]
        invocation.extend(args.passthrough)
    print('executing official runtime:', ' '.join(invocation))
    subprocess.run(invocation, cwd=official_root, check=True, env=os.environ.copy())


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest='action', required=True)
    conversion = sub.add_parser('convert', help='losslessly normalize and copy an official checkpoint')
    conversion.add_argument('--source', default=str(DEFAULT_OFFICIAL_ROOT / 'ckpts/epoch_56.pth'))
    conversion.add_argument('--output', required=True)
    conversion.add_argument('--key-map', help='JSON source-prefix -> destination-prefix mapping')
    conversion.add_argument('--force', action='store_true')
    verification = sub.add_parser('strict-check', help='load converted checkpoint into official model with strict=True')
    for item in (verification,):
        item.add_argument('--official-root', default=str(DEFAULT_OFFICIAL_ROOT))
        item.add_argument('--config', default=str(DEFAULT_CONFIG))
        item.add_argument('--checkpoint', required=True)
        item.add_argument('--python', help='Python executable of the official MMCV-1/MMDet-2 environment')
        item.add_argument('--in-official-runtime', action='store_true', help=argparse.SUPPRESS)
    for name in ('eval', 'train'):
        invocation = sub.add_parser(name, help=f'run official SparseWorld {name} after strict verification')
        invocation.add_argument('--official-root', default=str(DEFAULT_OFFICIAL_ROOT))
        invocation.add_argument('--config', default=str(DEFAULT_CONFIG))
        invocation.add_argument('--checkpoint', required=(name == 'eval'))
        invocation.add_argument('--python', help='Python executable of the official MMCV-1/MMDet-2 environment')
        invocation.add_argument('--in-official-runtime', action='store_true', help=argparse.SUPPRESS)
        invocation.add_argument('passthrough', nargs=argparse.REMAINDER)
    return root


def main() -> None:
    args = parser().parse_args()
    if args.action == 'convert': convert(args)
    elif args.action == 'strict-check':
        if args.in_official_runtime:
            strict_check(args)
        else:
            _run_in_official_python(args, 'strict-check')
    else: run_official(args, args.action)


if __name__ == '__main__':
    main()
