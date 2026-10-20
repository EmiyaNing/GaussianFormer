"""Tests for the A4 kernel-form control (report P1-7).

Plan T1.1 lists A4 (fixed isotropic Gaussian) as the cell that separates "the
kernel form adds nothing" from "the kernel works but was parameterised wrong".
Two things must not drift:

* the contrast labelling -- only A5-vs-A4 is single-factor (only
  ``freeze_geometry`` differs), while A4-vs-A2 confounds the weighting with the
  support-box selection, exactly like A5-vs-A2;
* the verdict, which reads the thin-class penalty to decide whether the deficit
  travels with the *learned geometry* or with the kernel *form*.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.p1.kernel_form_controls import (  # noqa: E402
    CONTRASTS, VARIANTS, geometry_verdict)


def test_only_the_a5_vs_a4_contrast_is_labelled_single_factor():
    by_label = {label: note for label, _, _, note in CONTRASTS}
    assert 'single factor' in by_label['A5 - A4']
    assert 'confounded' in by_label['A4 - A2']
    assert 'confounded' in by_label['A5 - A2']
    # A4 must keep A5's selection, which is what makes A4-vs-A2 confounded.
    assert 'shell' in by_label['A4 - A2']


def test_verdict_attributes_the_thin_class_deficit():
    # Frozen geometry keeps the gain A5 loses -> the learned geometry is the harm.
    assert geometry_verdict(-0.05, -0.21) == 'geometry-is-the-harm'
    # Same penalty -> the kernel form is the harm.
    assert geometry_verdict(-0.19, -0.21) == 'form-is-the-harm'
    assert geometry_verdict(-0.21, -0.21) == 'form-is-the-harm'
    # In between -> no single attribution.
    assert geometry_verdict(-0.13, -0.21) == 'mixed'
    # Missing A4 never claims an attribution.
    assert geometry_verdict(None, -0.21) == 'INCOMPLETE'
    assert geometry_verdict(-0.05, None) == 'INCOMPLETE'


def test_a4_is_in_the_family_and_the_artifact_marks_it_missing_for_now():
    assert VARIANTS['A4 fixed isotropic Gaussian'] == 'a4_fixed_gaussian'
    path = ROOT / 'benchmarks/results/p1_kernel_form_controls.json'
    if not path.exists():
        pytest.skip('kernel-form artifact not present')
    import json
    payload = json.loads(path.read_text(encoding='utf-8'))
    if 'A4 fixed isotropic Gaussian' in payload['missing']:
        assert payload['verdict'] == 'INCOMPLETE'
        # The reference contrast is still computed and matches P1-6's seed-42 pair.
        assert payload['contrasts']['A5 - A2']['delta'] == pytest.approx(
            -0.058, abs=0.005)
        assert payload['contrasts']['A5 - A2']['thin_delta'] == pytest.approx(
            -0.211, abs=0.005)
