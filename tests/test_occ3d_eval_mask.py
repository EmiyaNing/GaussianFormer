import torch

from train import select_occ3d_eval_mask


def test_occ3d_eval_mask_selects_one_dense_batch_item():
    mask = torch.zeros(2, 2, 2, 2, dtype=torch.bool)
    mask[1] = True
    selected = select_occ3d_eval_mask({'occ_cam_mask': mask}, 1,
                                      {'occ3d_eval_mask': 'camera'})
    assert selected.shape == (8,)
    assert selected.all()


def test_occ3d_eval_mask_selects_one_flattened_batch_item():
    mask = torch.tensor([[True, False, True], [False, True, False]])
    selected = select_occ3d_eval_mask({'occ_cam_mask': mask}, 1,
                                      {'occ3d_eval_mask': 'camera'})
    assert selected.tolist() == [False, True, False]


def test_occ3d_eval_mask_validates_per_rank_prediction_length():
    mask = torch.ones(2, 2, 2, 2, dtype=torch.bool)
    selected = select_occ3d_eval_mask({'occ_cam_mask': mask}, 1,
                                      {'occ3d_eval_mask': 'camera'}, expected_numel=8)
    assert selected.shape == (8,)
