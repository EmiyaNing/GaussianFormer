import torch

from model.head.opus_head import OPUSHead


def _run_head(batch_size):
    head = OPUSHead(
        embed_dims=8, num_classes=3, point_multipliers=(1,),
        pc_range=(0.0, 0.0, 0.0, 2.0, 2.0, 2.0), grid_size=1.0,
        grid_shape=(2, 2, 2), empty_label=3)
    metas = {
        'occ_xyz': torch.rand(batch_size, 2, 2, 2, 3),
        'occ_label': torch.randint(0, 4, (batch_size, 2, 2, 2)),
        'occ_mask': torch.ones(batch_size, 2, 2, 2, dtype=torch.bool),
        'occ_cam_mask': torch.ones(batch_size, 2, 2, 2, dtype=torch.bool),
        'occ_lidar_mask': torch.ones(batch_size, 2, 2, 2, dtype=torch.bool),
        'occ_nonempty_mask': torch.ones(batch_size, 2, 2, 2, dtype=torch.bool),
        'occ_loss_mask': torch.ones(batch_size, 2, 2, 2, dtype=torch.bool),
    }
    output = head([{
        'query_features': torch.randn(batch_size, 2, 8),
        'query_points': torch.rand(batch_size, 2, 3),
    }], metas=metas)
    assert output['final_occ'].shape == (batch_size, 8)
    assert output['sampled_label'].shape == (batch_size, 8)
    assert output['sampled_xyz'].shape == (batch_size, 8, 3)
    for key in ('occ_mask', 'occ_cam_mask', 'occ_lidar_mask',
                'occ_nonempty_mask', 'occ_loss_mask'):
        assert output[key].shape == (batch_size, 8)


def test_opus_head_metric_contract_for_full_batch():
    _run_head(batch_size=4)


def test_opus_head_metric_contract_for_tail_batch():
    _run_head(batch_size=1)


def test_opus_head_exposes_point_valid_mask():
    head = OPUSHead(embed_dims=8, num_classes=3, point_multipliers=(2,))
    metas = {key: torch.ones(1, 1, 1, 1, dtype=torch.bool) for key in (
        'occ_mask', 'occ_cam_mask', 'occ_lidar_mask', 'occ_nonempty_mask', 'occ_loss_mask')}
    metas['occ_xyz'] = torch.zeros(1, 1, 1, 1, 3)
    metas['occ_label'] = torch.zeros(1, 1, 1, 1, dtype=torch.long)
    output = head([{
        'query_features': torch.randn(1, 3, 8),
        'query_points': torch.rand(1, 3, 3),
        'query_valid_mask': torch.tensor([[True, False, True]]),
    }], metas=metas)
    assert output['opus_point_valid_masks'][0].tolist() == [[True, True, False, False, True, True]]
