"""Every control config must differ from its base in exactly the intended key.

An ablation is only an ablation if *one* thing changes.  A stray edit -- a
learning rate bumped in one file, a schedule copied with a different value, a
mask flag left on -- silently turns a controlled comparison into a confounded
one, and no accuracy number will reveal it.  This module makes the single-factor
property mechanical for every adapter/ablation config the reports rely on.

Two kinds of pairs:

* **single-leaf controls** (A2', A2'', A2-gauss, A2-max, A4, robustness): exactly
  one or two leaves differ, and they are the ones the report names;
* **protocol switches** (reduced -> full val): exactly one leaf differs,
  ``val_dataset_config.imageset``.  That is what makes the reduced/full
  comparison isolate the *evaluation set* rather than the model.

The full-val pair is the one that carries the published-anchor question
(report P1-9): because nothing but the imageset changes, a difference in mIoU
between the two protocols cannot be attributed to the configuration.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ABLATIONS = 'config/opus/ablations/'


def leaves(path):
    """Flatten a config to ``{dotted.path: scalar-or-repr}``."""
    from mmengine import Config
    flattened = {}

    def walk(node, prefix=''):
        if isinstance(node, dict):
            for key, value in node.items():
                if str(key).startswith('_'):
                    continue
                walk(value, f'{prefix}{key}.')
        elif isinstance(node, (list, tuple)):
            flattened[prefix[:-1]] = repr(node)
        else:
            flattened[prefix[:-1]] = node

    walk(Config.fromfile(str(ROOT / path)).to_dict())
    return flattened


def differing_leaves(base, variant):
    first, second = leaves(base), leaves(variant)
    return {key: (first.get(key, '<absent>'), second.get(key, '<absent>'))
            for key in set(first) | set(second)
            if first.get(key, '<absent>') != second.get(key, '<absent>')}


SINGLE_LEAF = {
    "A2' k=215": (ABLATIONS + 'p1_a2_knn_mean.py',
                  ABLATIONS + 'p1_a2_knn_k215.py',
                  {'model.semantic_gaussian.num_neighbors': (8, 215)}),
    "A2'' k=59": (ABLATIONS + 'p1_a2_knn_mean.py',
                  ABLATIONS + 'p1_a2_knn_k59.py',
                  {'model.semantic_gaussian.num_neighbors': (8, 59)}),
    'A2-gauss': (ABLATIONS + 'p1_a2_knn_mean.py',
                 ABLATIONS + 'p1_a2_knn_gaussian8.py',
                 {'model.semantic_gaussian.mode': ('mean', 'gaussian'),
                  'model.semantic_gaussian.gaussian_scale': ('<absent>', 0.09)}),
    'A2-max': (ABLATIONS + 'p1_a2_knn_mean.py',
               ABLATIONS + 'p1_a2_knn_max.py',
               {'model.semantic_gaussian.mode': ('mean', 'max')}),
    'A4 fixed geometry': (ABLATIONS + 'p1_a5_gaussian.py',
                          ABLATIONS + 'p1_a4_fixed_gaussian.py',
                          {'model.semantic_gaussian.freeze_geometry':
                           ('<absent>', True)}),
    'robustness cell 0.9': (ABLATIONS + 'p1_a5_gaussian.py',
                            ABLATIONS + 'p4_rob_cell0p9.py',
                            {'model.semantic_gaussian.cell_size': (1.8, 0.9)}),
    'robustness cell 3.6': (ABLATIONS + 'p1_a5_gaussian.py',
                            ABLATIONS + 'p4_rob_cell3p6.py',
                            {'model.semantic_gaussian.cell_size': (1.8, 3.6)}),
    'robustness support 2.0': (ABLATIONS + 'p1_a5_gaussian.py',
                               ABLATIONS + 'p4_rob_support2.py',
                               {'model.semantic_gaussian.support_sigma':
                                (3.0, 2.0)}),
    'robustness support 4.0': (ABLATIONS + 'p1_a5_gaussian.py',
                               ABLATIONS + 'p4_rob_support4.py',
                               {'model.semantic_gaussian.support_sigma':
                                (3.0, 4.0)}),
    # The cap/cell interaction probe: each variant adds exactly one knob to the
    # cap=8 reference, so the comparison against a5_cap8 (cell 1.8) isolates the
    # bin edge rather than mixing in the cap.
    'cap8 at cell 0.9': (ABLATIONS + 'p1_a5_cap8.py',
                         ABLATIONS + 'p4_cap8_cell0p9.py',
                         {'model.semantic_gaussian.cell_size': (1.8, 0.9)}),
    'cap8 at cell 3.6': (ABLATIONS + 'p1_a5_cap8.py',
                         ABLATIONS + 'p4_cap8_cell3p6.py',
                         {'model.semantic_gaussian.cell_size': (1.8, 3.6)}),
    'cap8 cell 0.9 vs uncapped': (ABLATIONS + 'p1_a5_gaussian.py',
                                  ABLATIONS + 'p4_cap8_cell0p9.py',
                                  {'model.semantic_gaussian.max_neighbours':
                                   ('<absent>', 8),
                                   'model.semantic_gaussian.cell_size':
                                   (1.8, 0.9)}),
    # The cap-cost curve ("concave marginal gain") rests on these differing only
    # in the cap -- a stray change in one of them would confound the whole curve.
    'cap 32': (ABLATIONS + 'p1_a5_gaussian.py',
               ABLATIONS + 'p1_a5_cap32.py',
               {'model.semantic_gaussian.max_neighbours': ('<absent>', 32)}),
    'cap 64': (ABLATIONS + 'p1_a5_gaussian.py',
               ABLATIONS + 'p1_a5_cap64.py',
               {'model.semantic_gaussian.max_neighbours': ('<absent>', 64)}),
    'cap 128': (ABLATIONS + 'p1_a5_gaussian.py',
                ABLATIONS + 'p1_a5_cap128.py',
                {'model.semantic_gaussian.max_neighbours': ('<absent>', 128)}),
    # The optimised dense baseline behind "8.14x" differs from the naive one in
    # the operator switch only.
    'block dense vs naive dense': (ABLATIONS + 'p3_dense_g2v.py',
                                   ABLATIONS + 'p3_dense_g2v_block.py',
                                   {'model.semantic_gaussian.dense_op':
                                    ('<absent>', 'block')}),
    # The horizon-curriculum cost probe (P3-2) changes the horizon list and the
    # cap, and nothing else.
    'sparse cap32 single-horizon': (ABLATIONS + 'p1_a5_gaussian.py',
                                    ABLATIONS + 'p4_a5_sparse32_h0.py',
                                    {'model.gaussian_horizons': ('[0, 2, 4, 6]',
                                                                 '[0]'),
                                     'model.semantic_gaussian.max_neighbours':
                                     ('<absent>', 32)}),
}

FULLVAL = {
    'A0': (ABLATIONS + 'p1_a0_base.py', ABLATIONS + 'p4_fullval_a0.py'),
    'A2': (ABLATIONS + 'p1_a2_knn_mean.py', ABLATIONS + 'p4_fullval_a2.py'),
    'A4': (ABLATIONS + 'p1_a4_fixed_gaussian.py', ABLATIONS + 'p4_fullval_a4.py'),
    'A5': (ABLATIONS + 'p1_a5_gaussian.py', ABLATIONS + 'p4_fullval_a5.py'),
}

IMAGESET = 'val_dataset_config.imageset'


@pytest.mark.parametrize('label', sorted(SINGLE_LEAF))
def test_control_differs_from_its_base_in_exactly_the_intended_keys(label):
    base, variant, expected = SINGLE_LEAF[label]
    got = differing_leaves(base, variant)
    assert set(got) == set(expected), (
        f'{label}: unexpected differing leaves.\n'
        f'  expected: {sorted(expected)}\n  got:      {sorted(got)}')
    for key, (want_base, want_variant) in expected.items():
        assert got[key] == (want_base, want_variant), (key, got[key])


@pytest.mark.parametrize('label', sorted(FULLVAL))
def test_fullval_switch_changes_only_the_val_imageset(label):
    base, variant = FULLVAL[label]
    got = differing_leaves(base, variant)
    assert set(got) == {IMAGESET}, (
        f'{label}: the reduced -> full-val switch must change only the val '
        f'imageset; it also changed {sorted(set(got) - {IMAGESET})}.  Model, '
        f'loss and training config must be byte-identical between protocols, '
        f'otherwise the full-val anchor does not isolate the evaluation set.')
    assert got[IMAGESET][0].endswith('val_reduced.pkl')
    assert got[IMAGESET][1].endswith('nuscenes_infos_val_sweeps_occ.pkl')


def test_gaussian_control_scale_matches_the_preregistered_ess_matching():
    """The A2-gauss scale is 0.09, chosen to match A5's ESS/accepted = 0.278.

    A scale of 0.35 (A5's initialisation) gives ESS 6.97/8 -- near-uniform and
    therefore vacuous as a weight-profile control.  The change to 0.09 was made
    before that training started; this pins it so a later edit cannot quietly
    restore the vacuous value.
    """
    _, _, expected = SINGLE_LEAF['A2-gauss']
    assert expected['model.semantic_gaussian.gaussian_scale'] == ('<absent>', 0.09)


def test_cap_seed_check_variants_reuse_the_seed42_cap_configs():
    """The cap curve's seed dimension must vary the checkpoint only.

    P1-11 re-evaluates cap 8/32/64/128 on the seed-43/44 checkpoints to find out
    whether P1-1's cap-cost curve is seed-robust.  That comparison is only about
    the seed if the *configs* are the same files the seed-42 points used; if one
    of the new registry entries pointed at a different config, the seed spread
    would be confounded with a knob change.
    """
    sys.path.insert(0, str(ROOT))
    from benchmarks.p1.eval_matrix import VARIANTS  # noqa: PLC0415

    for cap in (8, 32, 64, 128):
        expected_config = f'config/opus/ablations/p1_a5_cap{cap}.py'
        assert VARIANTS[f'a5_cap{cap}'][0] == expected_config
        for seed in ('s43', 's44'):
            name = f'a5_cap{cap}_{seed}'
            assert name in VARIANTS, f'cap seed-check variant {name} missing'
            config, checkpoint = VARIANTS[name]
            assert config == expected_config, (name, config)
            assert checkpoint == f'out/a5_{seed}/epoch_40.pth', (name, checkpoint)

    # No two variants may share both a config and a checkpoint: that would be a
    # silent duplicate, i.e. a point on the curve that is really another point.
    seen = {}
    for name, (config, checkpoint) in VARIANTS.items():
        key = (config, checkpoint)
        assert key not in seen, f'{name} duplicates {seen.get(key)}'
        seen[key] = name
