import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch
from mmengine import Config
from loss.gaussian_core_coverage_loss import GaussianCoreCoverageLoss, clipped_core_distance


def grid(shape=(4,4,4),h=.4):
    return (torch.stack(torch.meshgrid(*[torch.arange(n) for n in shape],indexing='ij'),-1).float().reshape(-1,3)+.5)*h


def test_clipped_distance_matches_all_gaussians_and_renderer_covariance():
    torch.manual_seed(928)
    shape=(5,4,3);xyz=grid(shape)
    means=torch.rand(13,3)*torch.tensor(shape)*.4
    scales=.08+torch.rand(13,3)*.32
    q=torch.randn(13,4)
    foreground=torch.rand(len(xyz))>.3
    actual=clipped_core_distance(means,scales,q,xyz,foreground,shape,(0.,0.,0.),chunk_size=3)
    source=Path(__file__).resolve().parents[1]/'model/utils/utils.py'
    spec=importlib.util.spec_from_file_location('coverage_rotation_reference',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rotation=module.get_rotation_matrix(q)
    inverse=rotation.transpose(-1,-2)@torch.diag_embed(scales.reciprocal().square())@rotation
    delta=xyz[:,None]-means[None]
    expected=torch.einsum('vni,nij,vnj->vn',delta,inverse,delta).clamp_min(0).sqrt().min(1).values.clamp_max(2)
    torch.testing.assert_close(actual[foreground],expected[foreground],atol=2e-6,rtol=2e-6)
    assert (actual[~foreground]==2).all()


def inputs(means,foreground_cells=(0,),camera=True,valid=None):
    shape=(4,4,4);xyz=grid(shape)
    labels=torch.full((1,*shape),17,dtype=torch.long)
    labels.flatten()[list(foreground_cells)]=4
    g=SimpleNamespace(means=means,scales=torch.full_like(means,.08,requires_grad=True),
        rotations=torch.tensor([1.,0.,0.,0.]).expand(*means.shape[:2],4).clone().requires_grad_(),
        semantics=torch.ones(*means.shape[:2],17,requires_grad=True),
        opacities=torch.ones(*means.shape[:2],1,requires_grad=True))
    result={'gaussian':g,'metas':dict(occ_label=labels,occ_cam_mask=torch.full_like(labels,camera,dtype=torch.bool),occ_xyz=xyz[None])}
    if valid is not None:result['opus_point_valid_masks']=[valid]
    return result


def objective():
    return GaussianCoreCoverageLoss(grid_shape=(4,4,4),pc_range=(0.,0.,0.,1.6,1.6,1.6),chunk_size=2)


def test_uncovered_targets_have_finite_center_gradient_even_far_away():
    m=torch.tensor([[[1.2,.2,.2]]],requires_grad=True)
    data=inputs(m)
    value,stats=objective()(data)
    gradients=torch.autograd.grad(value,[m,data['gaussian'].scales,data['gaussian'].rotations,
        data['gaussian'].semantics,data['gaussian'].opacities],allow_unused=True)
    assert value>0 and stats['active_gt']==1
    assert gradients[0][0,0,0]>0 and torch.isfinite(gradients[0]).all()
    assert all(g is None for g in gradients[1:])


def test_other_gaussian_can_cover_when_euclidean_nearest_does_not():
    m=torch.tensor([[[.35,.2,.2],[.5,.2,.2]]],requires_grad=True)
    data=inputs(m)
    data['gaussian'].scales=torch.tensor([[[.08,.08,.08],[.4,.4,.4]]],requires_grad=True)
    value,stats=objective()(data)
    assert value==0 and stats['active_gt']==0
    assert torch.autograd.grad(value,m)[0].abs().sum()==0


def test_camera_mask_and_invalid_gaussians_are_respected():
    m=torch.tensor([[[.2,.2,.2],[1.2,.2,.2]]],requires_grad=True)
    data=inputs(m,valid=torch.tensor([[False,True]]))
    value,stats=objective()(data)
    grad,=torch.autograd.grad(value,m)
    assert stats['active_gt']==1 and grad[0,0].abs().sum()==0 and grad[0,1,0]>0
    value,stats=objective()(inputs(m,camera=False))
    assert value==0 and stats['gt_foreground']==0


def test_denominator_includes_already_covered_foreground():
    m=torch.tensor([[[1.4,.2,.2]]],requires_grad=True)
    one,_=objective()(inputs(m,foreground_cells=(0,)))
    two,stats=objective()(inputs(m,foreground_cells=(0,48)))
    assert stats['gt_foreground']==2 and stats['active_gt']==1
    torch.testing.assert_close(two,one/2)


def test_full_config_preserves_original_losses_and_data():
    root=Path(__file__).resolve().parents[1]
    new=Config.fromfile(root/'config/full_occ3d_coverage/opusv1_gaussian_tiny_8f_coverage_1ep.py')
    old=Config.fromfile(root/'config/full_occ3d_oups_guassian/opusv1_gaussian_softplus_learned_bias_tiny_8f.py')
    assert new.model==old.model and new.loss.loss_cfgs[:2]==old.loss.loss_cfgs
    assert new.train_dataset_config==old.train_dataset_config
    assert new.val_dataset_config==old.val_dataset_config
    assert new.max_epochs==1 and new.train_load_from=='ckpts/opus_gaussian_8f_tiny.pth'


def test_recovery_metadata_is_compatible_with_native_eval_torch_load(tmp_path):
    import random
    import numpy as np
    from tools.train_coverage_finetune import numpy_rng_checkpoint,restore_numpy_rng
    np.random.seed(928)
    state=dict(state_dict={'sample':torch.ones(2)},rng_numpy=numpy_rng_checkpoint(),
               rng_random=random.getstate(),rng_torch=torch.get_rng_state())
    expected=np.random.rand(5)
    path=tmp_path/'checkpoint.pth';torch.save(state,path)
    restored=torch.load(path,weights_only=True)
    restore_numpy_rng(restored['rng_numpy'])
    np.testing.assert_array_equal(np.random.rand(5),expected)
