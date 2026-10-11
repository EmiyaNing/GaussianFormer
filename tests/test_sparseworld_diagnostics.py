import numpy as np
import torch
from torch import nn

from misc.sparseworld_diagnostics import (
    InterventionAccumulator, feature_intervention, metrics_from_confusion,
    position_intervention, privileged_query_translation, query_selection)
from model.segmentor.sparseworld_strict_segmentor import (
    SparseWorldStrictSegmentor, _EgoAttention)


def snapshot():
    positions = torch.tensor([[[[.25, .25, .5]], [[.75, .75, .5]],
                               [[.40, .60, .5]], [[.60, .40, .5]]]])
    extent = torch.tensor([80., 80., 6.4])
    lower = torch.tensor([-40., -40., -1.])
    return dict(
        positions=positions, positions_m=positions * extent + lower,
        features=torch.ones(1, 4, 3), slot_ids=torch.arange(4),
        completed_steps=1)


def test_zero_position_intervention_is_exact_and_does_not_alias():
    original = snapshot()
    changed, metadata, _ = position_intervention(original, 0., 'all', 7)
    assert torch.equal(changed['positions'], original['positions'])
    assert changed['positions'].data_ptr() != original['positions'].data_ptr()
    assert metadata['actual_coordinate_rms_m'] == 0.


def test_position_noise_is_deterministic_shared_per_query_and_unclipped():
    original = snapshot()
    first, first_meta, first_delta = position_intervention(original, 100., 'all', 9)
    second, second_meta, second_delta = position_intervention(original, 100., 'all', 9)
    assert torch.equal(first_delta, second_delta)
    assert first_meta == second_meta
    assert first_meta['out_of_bounds_after'] > 0.
    assert torch.allclose(first['positions'] - original['positions'],
                          first_delta / torch.tensor([80., 80., 6.4]))


def test_random_ten_percent_has_at_least_one_fixed_query():
    state = snapshot()
    assert torch.equal(query_selection(state, 'random10', 3),
                       query_selection(state, 'random10', 3))
    assert query_selection(state, 'random10', 3).sum() == 1


def test_feature_intervention_uses_channel_std_and_preserves_positions():
    state = snapshot()
    changed, metadata = feature_intervention(state, 1., [0., 1., 2.], 'all', 4)
    assert torch.equal(changed['positions'], state['positions'])
    assert torch.equal(changed['features'][..., 0], state['features'][..., 0])
    assert not torch.equal(changed['features'][..., 2], state['features'][..., 2])
    assert metadata['feature_std_source'] == 'fixed_external_vector'


def test_confusion_aggregation_precedes_iou():
    accumulator = InterventionAccumulator(num_classes=3)
    accumulator.add(('x',), [0, 1], [0, 1], None,
                    torch.zeros(2, 3), torch.zeros(2, 3), 'a')
    accumulator.add(('x',), [1, 1], [0, 2], None,
                    torch.zeros(2, 3), torch.zeros(2, 3), 'b')
    result = accumulator.results()['x']
    expected = metrics_from_confusion(np.array([[1, 1, 0], [0, 1, 0], [0, 1, 0]]),
                                      free_label=17)
    assert result['mIoU'] == expected['mIoU']
    assert result['sample_tokens'] == ['a', 'b']


def tiny_forecaster():
    network = SparseWorldStrictSegmentor.__new__(SparseWorldStrictSegmentor)
    nn.Module.__init__(network)
    network.pc_range = (-4., -4., -1., 4., 4., 3.)
    network.num_refines = 2
    network.ego_attention = _EgoAttention(4, 1, 0., network.pc_range)
    network.position = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 4))
    network.semantic = nn.Linear(4, 34)
    network.reg = nn.Linear(4, 6)
    network.velocity = nn.Linear(4, 4)
    network.traj = nn.Linear(4, 2)
    network.eval()
    return network


def test_multi_snapshot_replay_and_clean_path_overrides():
    torch.manual_seed(11)
    network = tiny_forecaster()
    stamps = torch.tensor([0, 0, 1, 2])
    all_features = torch.randn(1, 4, 4)
    all_points = torch.rand(1, 4, 2, 3)
    current = stamps == 0
    state = dict(
        features=all_features[:, current], positions=all_points[:, current],
        timestamp=torch.zeros(1, 2, 2, 1), ego=torch.randn(1, 1, 4),
        all_features=all_features, all_points=all_points, stamps=stamps,
        slot_ids=torch.nonzero(current, as_tuple=False).flatten(), completed_steps=0)
    rollout = network._forecast_rollout(state, 2, snapshot_after_step=(1, 2))
    assert set(rollout['snapshots']) == {1, 2}
    replay = network._replay_forecast(rollout['snapshots'][1], 2, collect_trace=True)
    assert torch.equal(replay['points'][0], rollout['points'][1])
    assert torch.equal(replay['logits'][0], rollout['logits'][1])
    perturbed, _, _ = position_intervention(
        rollout['snapshots'][1], .2, 'all', 5, network.pc_range)
    d3 = network._replay_forecast(
        perturbed, 2,
        distance_positions=[replay['trace'][0]['input_positions']])
    d4 = network._replay_forecast(
        perturbed, 2,
        encoding_positions=[replay['trace'][0]['interaction_positions']])
    ego_blocked = network._replay_forecast(
        perturbed, 2, collect_trace=True,
        fused_ego_features=[replay['trace'][0]['used_fused_ego']])
    assert d3['points'][0].shape == replay['points'][0].shape
    assert d4['points'][0].shape == replay['points'][0].shape
    assert torch.equal(ego_blocked['trace'][0]['used_fused_ego'],
                       replay['trace'][0]['used_fused_ego'])
    assert 'raw_fused_ego' in ego_blocked['trace'][0]


def test_privileged_translation_is_capped_and_reduces_fixed_match_loss():
    points = torch.tensor([[0., 0., 0.], [.1, 0., 0.],
                           [1., 1., 0.], [1.1, 1., 0.]])
    targets = points + torch.tensor([.4, 0., 0.])
    delta, losses = privileged_query_translation(
        points, targets, num_refines=2, steps=10,
        learning_rate=.1, max_translation_m=.25)
    assert delta.shape == (2, 3)
    assert delta.norm(dim=-1).max() <= .250001
    assert losses[-1] < losses[0]
