"""Compatibility shims for the PyTorch versions used by this repository.

The React execution path was developed against PyTorch 2.x, while the target
training environment may run an older 1.x release.  Keep the handful of
version-dependent helpers in one place so model code stays readable.
"""

import inspect

import torch


def _module_apply_accepts_recurse():
    """Whether ``nn.Module._apply`` exposes the ``recurse`` keyword.

    Newer PyTorch releases accept ``_apply(fn, recurse=...)``; the older
    1.12/1.13 stacks this repository also supports define ``_apply(self, fn)``
    and always recurse into child modules.
    """
    try:
        parameters = inspect.signature(torch.nn.Module._apply).parameters
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return False
    return 'recurse' in parameters


MODULE_APPLY_ACCEPTS_RECURSE = _module_apply_accepts_recurse()
