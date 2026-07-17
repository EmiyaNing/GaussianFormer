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


def test_opus_set_loss_ignores_padded_prediction_points():
    loss_fn = OPUSSetLoss(stage_weights=(1.0,), max_match_points=8)
    points = torch.tensor([[[-39.0, -39.0, -0.5], [30.0, 30.0, 4.0]]], requires_grad=True)
    logits = torch.randn(1, 2, 17, requires_grad=True)
    loss, metrics = loss_fn({
        'opus_pred_points': [points],
        'opus_pred_logits': [logits],
        'opus_point_valid_masks': [torch.tensor([[True, False]])],
        'metas': {
            'opus_gt_points': torch.tensor([[[-39.0, -39.0, -0.5]]]),
            'opus_gt_labels': torch.tensor([[1]]),
            'opus_gt_valid': torch.tensor([[True]]),
        },
    })
    loss.backward()
    assert metrics['valid_pred_points'].item() == 1
    assert torch.allclose(points.grad[:, 1], torch.zeros_like(points.grad[:, 1]))


def test_opus_set_loss_prefers_all_dense_nonempty_voxels_over_packed_targets():
    loss_fn = OPUSSetLoss(stage_weights=(1.0,), class_weights=[1, 1], chunk_size=4,
                          pc_range=(0., 0., 0., 2., 2., 2.))
    points = torch.tensor([[[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]]], requires_grad=True)
    logits = torch.zeros(1, 2, 2, requires_grad=True)
    dense_xyz = torch.tensor([[[[[0.5, 0.5, 0.5]], [[1.5, 1.5, 1.5]]]]])
    dense_label = torch.tensor([[[[0], [1]]]])
    loss, metrics = loss_fn({
        'opus_pred_points': [points], 'opus_pred_logits': [logits],
        'metas': {
            'occ_xyz': dense_xyz, 'occ_label': dense_label,
            'occ_cam_mask': torch.tensor([[[[True], [False]]]]),
            # Deliberately contradictory legacy payload: it must be ignored.
            'opus_gt_points': torch.zeros(1, 1, 3),
            'opus_gt_labels': torch.zeros(1, 1, dtype=torch.long),
            'opus_gt_valid': torch.zeros(1, 1, dtype=torch.bool),
        }})
    loss.backward()
    assert metrics['valid_gt_points'].item() == 2
    assert torch.isfinite(points.grad).all()
    assert logits.grad.abs().sum() > 0
