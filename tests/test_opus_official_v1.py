import torch

from loss.opus_set_loss import OPUSSetLoss
from model.encoder.opus_encoder import (
    OfficialOPUSV1Encoder, StrictOPUSV1Encoder, _StrictOPUSSelfAttention)
from model.head.opus_rasterizer import OPUSRasterizer


def test_official_v1_loss_supervises_every_prediction_without_fixed_truncation():
    points = torch.tensor([[[0.05, 0.05, 0.05], [0.45, 0.05, 0.05],
                            [0.05, 0.45, 0.05], [0.45, 0.45, 0.05]]], requires_grad=True)
    logits = torch.zeros(1, 4, 2, requires_grad=True)
    loss_fn = OPUSSetLoss(stage_weights=(1.0,), chunk_size=2, lambda_cls=1.0,
                          lambda_pts=1.0, class_weights=[1, 1],
                          pc_range=(0., 0., 0., 1., 1., 1.))
    loss, _ = loss_fn({
        'opus_pred_points': [points], 'opus_pred_logits': [logits],
        'metas': {
            'opus_gt_points': torch.tensor([[[0.10, 0.10, 0.10], [0.40, 0.40, 0.10]]]),
            'opus_gt_labels': torch.tensor([[0, 1]]),
            'opus_gt_valid': torch.tensor([[True, True]]),
            'opus_gt_camera_valid': torch.tensor([[True, True]]),
        }})
    loss.backward()
    assert torch.all(points.grad.abs().sum(dim=-1) > 0)


def test_official_v1_rasterizer_uses_sigmoid_threshold_not_softmax_argmax():
    rasterizer = OPUSRasterizer([0, 0, 0, 1, 1, 1], 1, [1, 1, 1],
                                score_threshold=0.5, padding=False)
    dense = rasterizer(torch.tensor([[[0.1, 0.1, 0.1]]]), torch.zeros(1, 1, 2))
    assert dense.item() == 17


def test_official_encoder_preserves_coarse_to_fine_point_groups():
    encoder = OfficialOPUSV1Encoder(
        embed_dims=8, num_decoder=2, num_heads=2, feedforward_channels=16,
        num_refines=(1, 2), pc_range=(0., 0., 0., 2., 2., 2.))
    output = encoder(
        torch.rand(1, 3, 8), torch.rand(1, 3, 3), [torch.rand(1, 1, 8, 4, 4)],
        {'projection_mat': torch.eye(4).view(1, 1, 4, 4),
         'image_wh': torch.tensor([[[4., 4.]]])})
    assert [tuple(state['query_points'].shape) for state in output['representation']] == [
        (1, 3, 1, 3), (1, 3, 2, 3)]


def test_strict_encoder_uses_all_eight_temporal_frames_and_decoder_logits():
    encoder = StrictOPUSV1Encoder(
        embed_dims=8, num_decoder=1, num_frames=8, num_views=6,
        num_points=4, num_levels=4, num_groups=4, num_heads=2,
        feedforward_channels=16, num_classes=3, num_refines=(1,),
        pc_range=(0., 0., 0., 2., 2., 2.))
    features = [torch.rand(1, 48, 8, 2, 2, requires_grad=True) for _ in range(4)]
    output = encoder(
        torch.zeros(1, 2, 8), torch.full((1, 2, 3), 0.5), features,
        {'projection_mat': torch.eye(4).view(1, 1, 4, 4).expand(1, 48, -1, -1),
         'image_wh': torch.full((1, 48, 2), 2.)})
    state = output['representation'][0]
    assert state['query_points'].shape == (1, 2, 1, 3)
    assert state['opus_logits'].shape == (1, 2, 1, 3)
    # Four sampling points from each of eight frames reach Adaptive Mixing.
    assert encoder.layers[0].mixing.in_points == 32
    (state['query_points'].sum() + state['opus_logits'].sum()).backward()
    assert all(feature.grad is not None for feature in features)


def test_strict_self_attention_eval_accepts_multi_batch_multi_head_mask():
    attention = _StrictOPUSSelfAttention(
        embed_dims=8, num_heads=2, dropout=0., pc_range=(0., 0., 0., 2., 2., 2.))
    attention.eval()
    with torch.no_grad():
        output = attention(torch.rand(4, 3, 1, 3), torch.rand(4, 3, 8))
    assert output.shape == (4, 3, 8)
    assert torch.isfinite(output).all()
