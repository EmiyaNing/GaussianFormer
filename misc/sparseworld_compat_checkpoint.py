"""Strict checkpoint loading used only by the local official-compat model."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


def _model_state(checkpoint):
    state = checkpoint.get('state_dict', checkpoint)
    if not isinstance(state, dict) or not state:
        raise ValueError('compat checkpoint must contain a non-empty state_dict')
    prefixes = [key.startswith('module.') for key in state]
    if any(prefixes) and not all(prefixes):
        raise ValueError('mixed DataParallel prefixes are not valid for strict loading')
    return {key[7:]: value for key, value in state.items()} if all(prefixes) else state


def _digest(tensor):
    value = tensor.detach().cpu().contiguous().reshape(-1)
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def load_sparseworld_compat_checkpoint(model, checkpoint_path, manifest_path=None):
    """Validate every key/shape/dtype, then load a compat checkpoint strictly.

    Returns the number of verified tensors.  There is deliberately no
    ``strict`` argument: this path is never permitted to partially load.
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state = _model_state(checkpoint)
    expected = model.state_dict()
    missing, unexpected = sorted(set(expected) - set(state)), sorted(set(state) - set(expected))
    shape = sorted(key for key in set(expected) & set(state)
                   if tuple(expected[key].shape) != tuple(state[key].shape))
    dtype = sorted(key for key in set(expected) & set(state)
                   if expected[key].dtype != state[key].dtype)
    if missing or unexpected or shape or dtype:
        raise RuntimeError('SparseWorld official-compat checkpoint mismatch: ' + json.dumps(
            dict(missing=missing, unexpected=unexpected, shape=shape, dtype=dtype), indent=2))
    if manifest_path is not None:
        manifest = json.loads(Path(manifest_path).read_text())
        if set(manifest) != set(state):
            raise RuntimeError('checkpoint manifest key set differs from checkpoint state_dict')
        bad = [key for key, value in state.items()
               if manifest[key]['shape'] != list(value.shape)
               or manifest[key]['dtype'] != str(value.dtype)
               or manifest[key]['sha256'] != _digest(value)]
        if bad:
            raise RuntimeError(f'checkpoint tensor digest mismatch for {bad[:16]}')
    model.load_state_dict(state, strict=True)
    return len(state)


def load_sparse_gaussian_world_warmstart(
        model, checkpoint_path,
        allowed_missing_prefixes=('semantic_gaussian.',
                                  'gaussian_horizon_embedding.')):
    """Load every official SparseWorld tensor and only initialize new adapters.

    Unlike an unrestricted ``strict=False`` load, this rejects missing base
    parameters, unexpected checkpoint keys, and all shape/dtype mismatches.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = _model_state(checkpoint)
    expected = model.state_dict()
    unexpected = sorted(set(state) - set(expected))
    missing = sorted(set(expected) - set(state))
    illegal_missing = [
        key for key in missing
        if not any(key.startswith(prefix) for prefix in allowed_missing_prefixes)]
    shape = sorted(key for key in set(expected) & set(state)
                   if tuple(expected[key].shape) != tuple(state[key].shape))
    dtype = sorted(key for key in set(expected) & set(state)
                   if expected[key].dtype != state[key].dtype)
    if unexpected or illegal_missing or shape or dtype:
        raise RuntimeError('SparseGaussianWorld warm-start mismatch: ' + json.dumps(
            dict(unexpected=unexpected, illegal_missing=illegal_missing,
                 shape=shape, dtype=dtype), indent=2))
    incompatible = model.load_state_dict(state, strict=False)
    if sorted(incompatible.missing_keys) != missing or incompatible.unexpected_keys:
        raise RuntimeError('PyTorch warm-start report differs from validated key sets')
    return dict(loaded=len(state), missing=missing)
