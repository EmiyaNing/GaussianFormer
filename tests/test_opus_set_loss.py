import torch

from loss.opus_set_loss import OPUSSetLoss


def test_opus_set_loss_has_finite_coordinate_and_class_gradients():
    loss_fn = OPUSSetLoss(
        stage_weights=(1.0,), lambda_cd=5.0,
        pc_range=(-40.0, -40.0, -1.0, 40.0, 40.0, 5.4),
        max_match_points=64, chunk_size=16)
    normalized_points = torch.rand(1, 64, 3, requires_grad=True)
    extent = torch.tensor([80.0, 80.0, 6.4])
    predicted_points = normalized_points * extent + torch.tensor([-40.0, -40.0, -1.0])
    logits = torch.randn(1, 64, 17, requires_grad=True)
    targets = torch.rand(1, 64, 3) * extent + torch.tensor([-40.0, -40.0, -1.0])
    loss, metrics = loss_fn({
        'opus_pred_points': [predicted_points],
        'opus_pred_logits': [logits],
        'metas': {
            'opus_gt_points': targets,
            'opus_gt_labels': torch.randint(0, 17, (1, 64)),
            'opus_gt_valid': torch.ones(1, 64, dtype=torch.bool),
        },
    })
    loss.backward()
    assert metrics['loss_cls'] > 0
    assert torch.isfinite(normalized_points.grad).all()
    assert torch.isfinite(logits.grad).all()
