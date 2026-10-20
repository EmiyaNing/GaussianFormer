from pathlib import Path

import pytest
import torch
from mmengine import Config
from mmseg.models import build_segmentor


def test_sparseworld_strict_checkpoint_contract():
    checkpoint = Path('../SparseWorld/ckpts/epoch_56.pth')
    if not checkpoint.is_file():
        pytest.skip('official SparseWorld checkpoint is not present')
    import model  # noqa: F401
    config = Config.fromfile('config/opus/sparseworld_strict.py')
    network = build_segmentor(config.model)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)['state_dict']
    assert len(network.state_dict()) == len(state) == 679
    assert sum(value.numel() for value in state.values()) == 76089910
    network.load_state_dict(state, strict=True)


def test_sparseworld_strict_official_sample_window():
    required = [
        Path('data/nuscenes_cam/nuscenes_infos_val_sweeps_occ.pkl'),
        Path('admlp/stp3_val/data_nuscene.pkl'),
        Path('occworld/nuscenes_infos_val_temporal_v3_scene.pkl')]
    if not all(path.is_file() for path in required):
        pytest.skip('SparseWorld temporal data artifacts are not present')
    import dataset
    config = Config.fromfile('config/opus/sparseworld_strict.py')
    value = dataset.OPENOCC_DATASET.build(config.val_dataset_config)
    # Official temp2nusc_map keeps current keyframes [5, scene_len - 8].
    assert len(value) == 4219
    scene, dense_index = value.keyframes[0]
    assert value.keyframe_indices_by_scene[scene].index(dense_index) == 5


def test_sparseworld_strict_metric_horizons():
    from misc.sparseworld_strict_eval import SparseWorldStrictMetric
    target = torch.tensor([[[[0], [17]], [[1], [17]]]])
    result = dict(final_occ_grid=target.clone(),
                  future_occ_grids=[target.clone() for _ in range(6)])
    metas = dict(occ_label=target,
                 future_occ_labels=target[:, None].repeat(1, 6, 1, 1, 1))
    metric = SparseWorldStrictMetric()
    metric.add_batch(result, metas)
    output = metric.results()
    assert output['avg/mIoU'] == 100.
    assert output['avg/IoU'] == 100.
