"""Guard the OPUSv1-only baseline protocol for sparse-render experiments.

SparseWorld uses a different query generator, temporal protocol, checkpoint,
and evaluator.  It must therefore never be substituted into the OPUSv1 P1/P3
experiments merely because an old SparseWorld result directory happens to be
available locally.
"""
from pathlib import Path

from mmengine import Config


ROOT = Path(__file__).resolve().parents[1]
P1 = ROOT / 'config/opus/ablations_opusv1'


def _config(name):
    return Config.fromfile(str(P1 / name))


def _all_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for nested in value.values():
            yield from _all_strings(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _all_strings(nested)


def test_p1_is_a_frozen_one_frame_opusv1_adapter_protocol():
    variants = ('p1_a1_pointwise.py', 'p1_a2_knn_mean.py',
                'p1_a3_graph_attention.py', 'p1_a4_fixed_gaussian.py',
                'p1_a5_gaussian.py')
    for name in variants:
        cfg = _config(name)
        assert cfg.model.type == 'FrozenOPUSAdapterSegmentor'
        assert cfg.model.freeze_opus_backbone is True
        assert cfg.model.encoder.num_frames == 1
        assert cfg.occ3d_eval_mask == 'camera'
        assert cfg.train_load_from == 'out/opusv1_1f/epoch_3.pth'
        assert cfg.train_load_strict is False


def test_p1_denominator_is_the_matching_opusv1_checkpoint():
    cfg = _config('p1_a0_base.py')
    assert cfg.model.type == 'OPUSSegmentor'
    assert cfg.model.encoder.num_frames == 1
    assert cfg.occ3d_eval_mask == 'camera'
    assert cfg.load_from == 'out/opusv1_1f/epoch_3.pth'
    assert cfg.train_load_from == cfg.load_from


def test_p3_compares_opusv1_eight_frame_systems_under_one_protocol():
    ours = _config('p3_ours_8f.py')
    dense = _config('p3_dense_g2v_block_8f.py')
    for cfg in (ours, dense):
        assert cfg.model.type == 'OPUSSegmentor'
        assert cfg.model.encoder.num_frames == 8
        assert cfg.occ3d_eval_mask == 'camera'
        assert cfg.model.head.point_multipliers == [1, 4, 16, 32, 64, 128]
        assert cfg.model.head.pc_range == [-40., -40., -1., 40., 40., 5.4]
    assert ours.model.head.type == 'OPUSSemanticGaussianLocalAggCUDAHead'
    assert dense.model.head.type == 'OPUSGaussianResidualHead'
    assert dense.model.head.gaussian_aggregator.type == 'DenseG2VResidualAggregator'


def test_opusv1_experiment_configs_have_no_sparseworld_dependency():
    for path in P1.glob('*.py'):
        cfg = Config.fromfile(str(path))
        assert all('sparseworld' not in value.lower()
                   for value in _all_strings(cfg.to_dict())), path
