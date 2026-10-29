"""Foreground mixing can include empty; a no-empty support is a strict subset."""
import pytest
from tools.finalize_boundary_gaussian_statistics import foreground_metrics


def test_foreground_definition_and_two_distinct_denominators():
    g=dict(predicted=100,boundary_any=10,boundary_foreground=7,boundary_fg_empty=8)
    r=foreground_metrics(g)
    assert r['foreground_boundary_fraction']==.07
    assert r['strict_no_empty_boundary']==2
    assert r['strict_no_empty_boundary_fraction']==.02
    assert r['foreground_fraction_within_original_boundary']==.7


def test_no_boundary_has_zero_primary_fraction_and_undefined_boundary_fraction():
    r=foreground_metrics(dict(predicted=100,boundary_any=0,boundary_foreground=0,boundary_fg_empty=0))
    assert r['foreground_boundary_fraction']==0
    assert r['strict_no_empty_boundary']==0
    assert r['foreground_fraction_within_original_boundary'] is None


def test_inconsistent_label_set_counts_are_rejected():
    with pytest.raises(AssertionError):
        foreground_metrics(dict(predicted=100,boundary_any=10,boundary_foreground=1,boundary_fg_empty=8))
