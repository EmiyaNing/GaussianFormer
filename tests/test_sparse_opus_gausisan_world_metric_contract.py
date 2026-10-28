"""Exercise the model-output boundary used by the unmodified generic runner.

Backbone/rollout/rendering are stubbed so the production forward and real metric
can be checked on CPU, including the full 200 x 200 x 16 no-mask failure case.
"""

import pytest
import torch

from misc.metric_util import MeanIoU
from model.segmentor.sparse_opus_gausisan_world import SparseOpusGaussianWorld
from train import select_occ3d_eval_mask


def forward_result(monkeypatch, batch_size, shape=(200, 200, 16)):
    model = SparseOpusGaussianWorld.__new__(SparseOpusGaussianWorld)
    torch.nn.Module.__init__(model)
    model.plan_head = torch.nn.Identity()
    model.points_scale_branch = torch.nn.Linear(21, 3, bias=False)
    model.pc_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
    model.pretrain = False
    outputs = dict(
        init_points=torch.zeros(batch_size, 2, 1, 3),
        all_refine_pts=torch.zeros(1, batch_size, 2, 1, 3),
        all_cls_scores=torch.zeros(1, batch_size, 2, 1, 17),
    )

    class Head(torch.nn.Module):
        ind_stamps_all = torch.tensor([0, 1])
        num_stamps_all = torch.ones(2)

        def forward(self, *args, **kwargs):
            return outputs

    model.pts_bbox_head = Head()
    monkeypatch.setattr(model, "extract_feat", lambda images, metas: ())
    forecasts = (
        [torch.zeros(batch_size, 1, 1, 3) for _ in range(6)],
        [torch.zeros(batch_size, 1, 1, 17) for _ in range(6)],
        [torch.zeros(batch_size, 1, 4) for _ in range(6)],
        [torch.zeros(batch_size, 1, 2) for _ in range(6)],
        [torch.ones(batch_size, 1, 1, dtype=torch.bool) for _ in range(6)],
    )
    monkeypatch.setattr(model, "_forecast", lambda *args: forecasts)
    voxels = shape[0] * shape[1] * shape[2]
    ids = torch.arange(batch_size * voxels).reshape(batch_size, *shape)
    labels = ids.remainder(18)
    predictions = (labels + (ids.remainder(5) == 0)).remainder(18)
    grids = [predictions] + [predictions.clone() for _ in range(6)]
    monkeypatch.setattr(
        model, "_render_occupancy", lambda *args: ({0: torch.zeros(1)}, grids)
    )
    metas = dict(
        temporal_ego_states=torch.zeros(batch_size, 21),
        occ_xyz=torch.zeros(batch_size, *shape, 3),
        occ_label=labels,
        occ_cam_mask=ids.remainder(3) != 0,
        occ_lidar_mask=ids.remainder(4) != 0,
        occ_nonempty_mask=labels != 17,
    )
    result = model(imgs=torch.zeros(batch_size, 1), metas=metas)
    return result, metas


def metric_on_cpu():
    metric = MeanIoU(list(range(17)), 17, [str(c) for c in range(17)],
                     filter_minmax=False)
    # reset() allocates on CUDA in production; keep this boundary test on CPU.
    metric.total_seen = torch.zeros(18)
    metric.total_correct = torch.zeros(18)
    metric.total_positive = torch.zeros(18)
    return metric


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("mask_name", ["none", "camera", "lidar", "nonempty"])
def test_generic_runner_receives_matching_predictions_and_targets(
    monkeypatch, batch_size, mask_name
):
    torch.set_num_threads(4)
    result, metas = forward_result(monkeypatch, batch_size)
    assert result["final_occ"].shape == result["sampled_label"].shape
    assert result["sampled_label"].shape == (batch_size, 640000)
    assert metas["occ_label"].shape == (batch_size, 200, 200, 16)
    torch.testing.assert_close(result["sampled_label"], metas["occ_label"].flatten(1))
    assert result["sampled_label"].data_ptr() == metas["occ_label"].data_ptr()
    assert result["final_occ_grid"].shape == metas["occ_label"].shape
    metric = metric_on_cpu()
    confusion = torch.zeros(18, 18, dtype=torch.int64)
    for idx, pred in enumerate(result["final_occ"]):
        # The same selection and metric call as train.py's validation loop.
        target = result["sampled_label"][idx]
        mask = select_occ3d_eval_mask(
            metas, idx, {"occ3d_eval_mask": mask_name}, expected_numel=pred.numel()
        )
        metric._after_step(pred, target, mask)
        if mask is not None:
            pred, target = pred[mask], target[mask]
        confusion += torch.bincount(target * 18 + pred, minlength=18 * 18).view(18, 18)
    torch.testing.assert_close(metric.total_seen[:17], confusion.sum(1)[:17].float())
    torch.testing.assert_close(metric.total_positive[:17], confusion.sum(0)[:17].float())
    torch.testing.assert_close(metric.total_correct[:17], confusion.diag()[:17].float())
    assert metric.total_seen[-1] == confusion[:17].sum()
    assert metric.total_positive[-1] == confusion[:, :17].sum()
    assert metric.total_correct[-1] == confusion[:17, :17].sum()


def test_legacy_dense_target_reproduces_reported_unmasked_error(monkeypatch):
    result, metas = forward_result(monkeypatch, 1)
    with pytest.raises(RuntimeError, match="640000"):
        metric_on_cpu()._after_step(result["final_occ"][0], metas["occ_label"][0], None)
