"""Contracts for full-space supervision and integrated epoch diagnostics."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from mmengine.config import Config

from loss import OPENOCC_LOSS
from .fullspace_diagnostics import add_evaluation_batch, region_metrics
from .metrics import HorizonMetrics
from .queue_fullspace import CONFIG
from .train_fullspace import save_diagnostic


def test_standalone_config_changes_only_supervision_and_run_metadata():
    path = Path(CONFIG)
    cfg = Config.fromfile(str(path)).to_dict()
    old = Config.fromfile('config/sparseworld_opus_gaussian_bias/sparseworld_opus_gaussian_bias_4096_1024_15e.py').to_dict()
    assert '_base_' not in path.read_text()
    assert cfg['max_epochs'] == 15
    assert cfg['loss']['mask_type'] == 'none'
    cfg['loss']['mask_type'] = 'camera'
    for key in ('epoch_diagnostics', 'experiment_name', 'report_directory'):
        cfg.pop(key, None)
        old.pop(key, None)
    assert cfg == old


@OPENOCC_LOSS.register_module()
class FullspaceContractZeroSetLoss(nn.Module):
    def forward(self, inputs):
        return sum(x.sum() * 0. for x in inputs['gaussian_logits_by_horizon'].values()), {}


def test_fullspace_loss_corrects_invisible_free_voxels_and_respects_future_weight():
    cfg = Config.fromfile(CONFIG)
    options = cfg.loss.to_dict()
    options['strict_loss'] = dict(type='FullspaceContractZeroSetLoss')
    options['occupancy_loss']['use_lovasz_loss'] = False
    options['occupancy_loss']['use_sem_geo_scal_loss'] = False
    metas = dict(occ_label=torch.tensor([[[[4, 17]]]]),
                 occ_xyz=torch.zeros(1, 1, 1, 2, 3),
                 occ_cam_mask=torch.tensor([[[[True, False]]]]))
    metas['future_occ_labels'] = torch.stack([metas['occ_label']] * 6, dim=1)
    metas['future_occ_cam_masks'] = torch.stack([metas['occ_cam_mask']] * 6, dim=1)

    def gradient(mask, horizon):
        setting = dict(options, mask_type=mask)
        loss_fn = OPENOCC_LOSS.build(setting)
        values = torch.zeros(1, 18, 2)
        values[:, 4] = 8.
        raw = values.requires_grad_(True)
        total, _ = loss_fn(dict(metas=metas, gaussian_logits_by_horizon={horizon: raw}))
        total.backward()
        return raw.grad

    full = gradient('none', 0)
    camera = gradient('camera', 0)
    assert full[0, 4, 1] > 0  # Gradient descent suppresses invisible occupied FP.
    assert full[0, 17, 1] < 0  # The invisible free target is supervised.
    assert camera[:, :, 1].count_nonzero() == 0
    future = gradient('none', 2)
    torch.testing.assert_close(future, full * cfg.loss.future_weight / cfg.loss.current_weight)


def test_integrated_metrics_equal_original_all_horizons_and_partition_exactly():
    generator = torch.Generator().manual_seed(17)
    target = torch.randint(0, 18, (2, 2, 2, 4), generator=generator)
    target[0, 0, 0, 0] = 255
    predictions = [torch.randint(0, 18, target.shape, generator=generator) for _ in range(7)]
    cam = torch.rand(target.shape, generator=generator) > .5
    metas = dict(occ_label=target, future_occ_labels=torch.stack([target] * 6, dim=1),
                 occ_cam_mask=cam, future_occ_cam_masks=torch.stack([cam] * 6, dim=1),
                 temporal_trajs=torch.randn(2, 6, 2, generator=generator))
    result = dict(final_occ_grid=predictions[0], future_occ_grids=predictions[1:],
                  pred_traj=torch.randn(2, 6, 2, generator=generator))
    original, integrated = HorizonMetrics(), HorizonMetrics()
    visibility = {key: np.zeros((7, 18, 18), dtype=np.int64)
                  for key in ('full', 'camera', 'noncamera')}
    original.add(result, metas)
    add_evaluation_batch(integrated, visibility, result, metas)
    assert original.results() == integrated.results()
    assert np.array_equal(original.confusion, visibility['full'])
    assert np.array_equal(visibility['full'], visibility['camera'] + visibility['noncamera'])
    assert original.samples == 2


def test_diagnostic_serialization_and_126_class_rows(tmp_path):
    metric = HorizonMetrics()
    metric.confusion[:, 4, 4] = 1
    metric.confusion[:, 17, 17] = 1
    metric.samples = 1
    formal = dict(metric.results(), epoch=1)
    arrays = dict(full=metric.confusion, camera=metric.confusion,
                  noncamera=np.zeros_like(metric.confusion))
    payload = dict(attributes=[], gradient_probes=[], render_probes={},
                   attribute_sample_indices=[], _visibility_arrays=arrays,
                   visibility={f'h{h}/{region}': region_metrics(values[h])
                               for h in range(7) for region, values in arrays.items()})
    save_diagnostic(tmp_path, 1, payload, formal)
    saved = json.loads((tmp_path / 'epoch_01_diagnostics.json').read_text())
    assert saved['visibility']['h0/noncamera']['voxel_count'] == 0
    assert len(list(csv.DictReader((tmp_path / 'epoch_01_class_iou.csv').open()))) == 126
    with np.load(tmp_path / 'epoch_01_visibility_confusion.npz') as stored:
        assert np.array_equal(stored['full'], stored['camera'] + stored['noncamera'])


def test_report_rejects_partial_epoch_diagnostics(tmp_path):
    from .report_fullspace import gather_diagnostics
    work, output = tmp_path / 'work', tmp_path / 'analysis'
    work.mkdir()
    (output / 'epoch_diagnostics').mkdir(parents=True)
    (output / 'epoch_diagnostics' / 'epoch_01_diagnostics.json').write_text(json.dumps(
        dict(epoch=1, num_samples=1, is_smoke=True, voxel_supervision_mask='none')))
    (work / 'epoch_01_metrics.json').write_text('{}')
    with pytest.raises(ValueError, match='incomplete'):
        gather_diagnostics(work, output)


def test_fullspace_report_includes_15_diagnostics_and_rejects_histogram_mismatch(tmp_path):
    from .report_fullspace import create_report, gather_diagnostics
    from .report import VARIANTS
    root, output = tmp_path / 'runs', tmp_path / 'analysis'
    diagnostic_dir = output / 'epoch_diagnostics'
    diagnostic_dir.mkdir(parents=True)
    for variant_index, variant in enumerate(VARIANTS):
        work = root / variant
        work.mkdir(parents=True)
        (work / 'best_recheck').mkdir()
        for epoch in range(1, 16):
            metric = HorizonMetrics()
            metric.samples = 1024
            metric.confusion[:, 4, 4] = epoch + variant_index
            metric.confusion[:, 4, 17] = 1024 - epoch - variant_index
            metric.confusion[:, 17, 17] = 1024
            formal = dict(metric.results(), epoch=epoch)
            (work / f'epoch_{epoch:02d}_metrics.json').write_text(json.dumps(formal))
            np.savez_compressed(work / f'epoch_{epoch:02d}_confusion.npz', confusion=metric.confusion)
            if variant_index == 1:
                attributes = [dict(horizon=h, scale_upper_saturation_percent=0.,
                                   opacity_upper_saturation_percent=0., empty_bias=.1)
                              for _ in range(16) for h in range(7)]
                diagnostic = dict(epoch=epoch, num_samples=1024, is_smoke=False,
                                  voxel_supervision_mask='none', attributes=attributes,
                                  attribute_sample_indices=list(range(16)),
                                  selection_mean_miou=formal['selection_mean_miou'],
                                  visibility={'h0/noncamera': dict(occupied_fp=0)})
                (diagnostic_dir / f'epoch_{epoch:02d}_diagnostics.json').write_text(json.dumps(diagnostic))
                np.savez_compressed(diagnostic_dir / f'epoch_{epoch:02d}_visibility_confusion.npz',
                                    full=metric.confusion, camera=metric.confusion,
                                    noncamera=np.zeros_like(metric.confusion))
        (work / 'best.json').write_text(json.dumps(formal))
        (work / 'best_mIoU.pth').write_bytes(b'synthetic report fixture')
        (work / 'best_recheck' / 'epoch_15_metrics.json').write_text(json.dumps(formal))
        np.savez_compressed(work / 'best_recheck' / 'epoch_15_confusion.npz', confusion=metric.confusion)
        (work / 'initialization.json').write_text(json.dumps(dict(sha256='common_image_init')))
        (work / 'completed.json').write_text(json.dumps(dict(epochs=15, manifest_sha256=dict(train='same', val='same'))))
    create_report(root, output)
    text = (output / 'report.md').read_text()
    assert '全空间监督' in text and 'camera-mask Gaussian 体素监督' not in text
    assert 'Strict复用原实验' in text
    assert json.loads((output / 'diagnostics_summary.json').read_text())['verified_epochs'] == 15
    assert len(list(csv.DictReader((output / 'per_horizon_per_class_iou.csv').open()))) == 126
    malformed = metric.confusion.copy()
    malformed[0, 0, 0] += 1
    np.savez_compressed(diagnostic_dir / 'epoch_01_visibility_confusion.npz',
                        full=malformed, camera=metric.confusion, noncamera=np.zeros_like(metric.confusion))
    with pytest.raises(ValueError, match='differs from formal'):
        gather_diagnostics(root / VARIANTS[1], output)
