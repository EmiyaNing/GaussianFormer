import torch

from model.encoder.opus_encoder import OPUSEncoder


def test_opus_encoder_ignores_padded_queries_and_accepts_lidar_memory():
    encoder = OPUSEncoder(
        embed_dims=8, num_decoder=1, num_heads=2, feedforward_channels=16,
        dropout=0.0, pc_range=(0., 0., 0., 4., 4., 4.), lidar_ball_k=2)
    query_features = torch.randn(2, 3, 8)
    query_points = torch.full((2, 3, 3), 0.5)
    valid = torch.tensor([[True, False, True], [True, True, True]])
    output = encoder(
        query_features=query_features,
        query_points=query_points,
        query_valid_mask=valid,
        ms_img_feats=[torch.randn(2, 1, 8, 4, 4)],
        metas={
            'projection_mat': torch.eye(4).view(1, 1, 4, 4).expand(2, -1, -1, -1),
            'image_wh': torch.full((2, 1, 2), 4.0),
        },
        lidar_memory_features=[torch.randn(2, 8), torch.randn(1, 8)],
        lidar_memory_points=[torch.tensor([[0.5, 0.5, 0.5], [0.6, 0.5, 0.5]]),
                             torch.tensor([[0.5, 0.5, 0.5]])],
    )
    state = output['representation'][0]
    assert torch.isfinite(state['query_features']).all()
    assert state['query_valid_mask'].equal(valid)
    assert torch.equal(state['query_features'][0, 1], torch.zeros(8))
    assert torch.equal(state['query_points'][0, 1], torch.zeros(3))
