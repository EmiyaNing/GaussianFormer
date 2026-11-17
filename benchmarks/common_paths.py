"""Shared path formatting for benchmark CLIs.

Several benchmark scripts print ``target.relative_to(ROOT)`` for a path the user
supplied via ``--json``/``--out``.  That raises ``ValueError`` whenever the path
is outside the repository -- e.g. ``--json /tmp/x.json``, which is exactly what a
dry run or a test wants to do.  The bug is cosmetic (the work is already saved)
but it turns a successful run into a traceback and exit code 1, which in a queued
pipeline reads as a failure.

Use ``display(path)`` instead of ``path.relative_to(ROOT)`` in messages.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def display(path) -> str:
    """Repo-relative string when possible, absolute otherwise."""
    try:
        return str(Path(path).relative_to(ROOT))
    except ValueError:
        return str(Path(path))
