"""Decoder memory paths: formulas, all gradients, RNG and checkpoint keys."""

from copy import deepcopy

import pytest
import torch
from mmengine import Config
from mmseg.registry import MODELS

from model.encoder.gaussian_encoder.deformable_module import DeformableFeatureAggregation as DFA
from model.encoder.gaussian_encoder.ops.masked_softmax import normalize_weights


def legacy_weights(logits, mask):
    logits = logits.clone()
    missing = (mask.sum(dim=(2,3,4), keepdim=True) == 0).expand_as(logits)
    logits[~mask] = -torch.inf
    logits[missing] = 0.
    return logits.flatten(2,4).softmax(-2).reshape_as(logits) * (1 - missing.float())


def module_config():
    return dict(embed_dims=8, num_groups=2, num_levels=2, num_cams=2,
                attn_drop=.15, proj_drop=.1, use_deformable_func=True,
                use_camera_embed=True,
                kps_generator=dict(type='SparseGaussian3DKeyPointsGenerator',
                    embed_dims=8, num_learnable_pts=2, fix_scale=[[0.,0.,0.]],
                    pc_range=[0.,0.,0.,2.,2.,2.], scale_range=[.08,.64]))


@pytest.mark.parametrize('image_size', [None, [4.,3.]])
def test_batched_projection_and_both_input_gradients(image_size):
    torch.manual_seed(84)
    points = torch.rand(2,7,3,6,dtype=torch.double)[...,::2].requires_grad_()
    matrices = torch.eye(4,dtype=torch.double)[None,None].repeat(2,6,1,1)[:,::2]
    matrices = (matrices + torch.randn_like(matrices)*.02).requires_grad_()
    wh = None if image_size is None else points.new_tensor(image_size).expand(2,3,2)
    expected, mask = DFA.project_points(points,matrices,wh)
    actual, actual_mask = DFA.project_points_batched(points,matrices,wh)
    assert torch.equal(mask,actual_mask)
    torch.testing.assert_close(actual,expected,atol=1e-8,rtol=1e-10)
    probe = torch.randn_like(expected)
    eg = torch.autograd.grad(expected,(points,matrices),probe,retain_graph=True)
    ag = torch.autograd.grad(actual,(points,matrices),probe)
    for a,b in zip(ag,eg):
        torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-9)


@pytest.mark.parametrize('camera_embed', [False, True])
def test_factored_camera_linear_preserves_outputs_bias_gradients_and_mask_rng(camera_embed):
    torch.manual_seed(71)
    config = module_config()
    config['use_camera_embed'] = camera_embed
    module = DFA(**config).double().train()
    features = torch.randn(2,11,8,dtype=torch.double,requires_grad=True)
    embed = torch.randn_like(features,requires_grad=True)
    projection = torch.randn(2,2,4,4,dtype=torch.double,requires_grad=True)
    variables = (features,embed) + tuple(module.weights_fc.parameters())
    if camera_embed:
        variables += (projection,) + tuple(module.camera_encoder.parameters())
    torch.manual_seed(42)
    expected, mask = module._get_weights(features,embed,dict(projection_mat=projection))
    probe = torch.randn_like(expected)
    eg = torch.autograd.grad(expected,variables,probe,retain_graph=True)
    module.factor_camera_weights = True
    torch.manual_seed(42)
    actual, actual_mask = module._get_weights(features,embed,dict(projection_mat=projection))
    ag = torch.autograd.grad(actual,variables,probe)
    assert torch.equal(mask,actual_mask)
    torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
    for a,b in zip(ag,eg):
        torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-10)


@pytest.mark.parametrize('fused', [False, True])
@pytest.mark.parametrize('shape', [(2,7,3,2,2,2), (1,5,9,6,4,4),
                                  (2,3,14,6,4,3), (1,5,1,1,1,1), (1,0,9,6,4,4)])
def test_masked_softmax_outputs_gradients_and_all_miss(fused,shape):
    if fused and not torch.cuda.is_available():
        pytest.skip('CUDA softmax unavailable')
    device = 'cuda' if fused else 'cpu'
    torch.manual_seed(82)
    logits = (torch.randn(*shape[:-1],shape[-1]*2,device=device)*4)[...,::2].requires_grad_()
    mask = (torch.rand_like(logits) > .4)
    if shape[1]:
        mask[:,0] = False
        mask[:,-1,:,:,:,-1] = False
    expected = legacy_weights(logits,mask)
    actual = normalize_weights(logits,mask,fused=fused)
    probe = torch.randn_like(expected).transpose(0,1).contiguous().transpose(0,1)
    eg = torch.autograd.grad(expected,logits,probe,retain_graph=True)[0]
    ag = torch.autograd.grad(actual,logits,probe)[0]
    torch.testing.assert_close(actual,expected,atol=4e-7,rtol=3e-6)
    torch.testing.assert_close(ag,eg,atol=5e-7,rtol=4e-5)
    assert actual[~mask].count_nonzero() == 0
    assert ag[~mask].count_nonzero() == 0
    torch.testing.assert_close(actual.sum((2,3,4)), mask.any((2,3,4)).float(),
                               atol=3e-7,rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_fused_softmax_saves_only_probabilities_and_follows_current_stream():
    stream = torch.cuda.Stream()
    saved = []
    with torch.cuda.stream(stream):
        logits = torch.randn(1,7,9,6,4,4,device='cuda',requires_grad=True)
        mask = torch.rand_like(logits)>.3
        mask[:,0] = False
        with torch.autograd.graph.saved_tensors_hooks(
                lambda t: (saved.append(t.untyped_storage().data_ptr()) or t), lambda t:t):
            actual = normalize_weights(logits,mask,fused=True)
        expected = legacy_weights(logits,mask)
        actual.square().sum().backward()
    stream.synchronize()
    assert saved == [actual.untyped_storage().data_ptr()]
    torch.testing.assert_close(actual,expected,atol=2e-7,rtol=2e-6)
    assert torch.isfinite(logits.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
@pytest.mark.parametrize('flags', [dict(batched_projection=True),
    dict(factor_camera_weights=True), dict(compact_weight_mask=True),
    dict(fused_weight_softmax=True), dict(batched_projection=True,
        factor_camera_weights=True,compact_weight_mask=True,fused_weight_softmax=True)])
def test_daf_memory_paths_preserve_inputs_parameters_and_dropout(flags):
    torch.manual_seed(75)
    module = DFA(**module_config(),fuse_point_sum=True).cuda().train()
    features = torch.randn(1,13,8,device='cuda',requires_grad=True)
    embed = torch.randn_like(features,requires_grad=True)
    anchor = torch.randn(1,13,28,device='cuda',requires_grad=True)
    maps = [torch.randn(1,2,8,4,4,device='cuda',requires_grad=True) for _ in range(2)]
    metas = dict(projection_mat=torch.eye(4,device='cuda')[None,None].repeat(1,2,1,1),
                 image_wh=torch.full((1,2,2),4.,device='cuda'))
    variables = (features,embed,anchor)+tuple(maps)+tuple(module.parameters())
    probe = torch.randn_like(features)
    torch.manual_seed(97)
    expected = module(features,anchor,embed,maps,metas)
    eg = torch.autograd.grad(expected,variables,probe,retain_graph=True)
    for name,value in flags.items():
        setattr(module,name,value)
    torch.manual_seed(97)
    actual = module(features,anchor,embed,maps,metas)
    ag = torch.autograd.grad(actual,variables,probe)
    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-6)
    for a,b in zip(ag,eg):
        torch.testing.assert_close(a,b,atol=2e-5,rtol=3e-4)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
@pytest.mark.parametrize('cp_ffn,cp_daf', [(True,False),(False,True),(True,True)])
def test_encoder_checkpoint_recomputation_preserves_all_gradients_and_rng(cp_ffn,cp_daf):
    cfg = deepcopy(Config.fromfile('config/img_point_lite_react/img_voxel_react_occ3d_fast_v4.py').model.encoder)
    cfg.operation_order = ['deformable','ffn','norm','refine']*2+['densify']
    cfg.anchor_encoder.embed_dims = 8
    cfg.norm_layer.normalized_shape = 8
    cfg.ffn.update(in_channels=16,embed_dims=8,feedforward_channels=32)
    cfg.deformable_model.update(module_config())
    cfg.refine_layer.update(embed_dims=8,pc_range=[0.,0.,0.,2.,2.,2.])
    cfg.densify_layer.update(feat_embed_dim=8,topk_count=3,pc_range=[0.,0.,0.,2.,2.,2.])
    torch.manual_seed(72)
    baseline = MODELS.build(cfg).cuda().train()
    actual_model = MODELS.build(cfg).cuda().train()
    actual_model.load_state_dict(baseline.state_dict(),strict=True)
    actual_model.checkpoint_ffn, actual_model.checkpoint_deformable = cp_ffn,cp_daf
    features = torch.randn(1,11,8,device='cuda',requires_grad=True)
    anchors = torch.randn(1,11,28,device='cuda',requires_grad=True)
    maps = [torch.randn(1,2,8,4,4,device='cuda',requires_grad=True) for _ in range(2)]
    metas = dict(projection_mat=torch.eye(4,device='cuda')[None,None].repeat(1,2,1,1),
                 image_wh=torch.full((1,2,2),4.,device='cuda'))
    def run(network):
        torch.manual_seed(93)
        output = network(anchors,features,ms_img_feats=maps,metas=metas)['representation']
        values = [value for stage in output for value in stage['gaussian'][:5]]
        probe = sum(value.square().mean() for value in values)
        rng = torch.cuda.get_rng_state().clone()
        gradients = torch.autograd.grad(probe,(anchors,features)+tuple(maps)+
            tuple(network.parameters()),allow_unused=True)
        assert torch.equal(rng,torch.cuda.get_rng_state())
        return values,gradients
    expected, eg = run(baseline)
    actual, ag = run(actual_model)
    for a,b in zip(actual,expected):
        torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6)
    for a,b in zip(ag,eg):
        if a is None or b is None:
            assert a is b
        else:
            torch.testing.assert_close(a,b,atol=2e-5,rtol=4e-4)


def test_v4_configs_keep_data_objective_precision_and_model_budget():
    baseline = Config.fromfile('config/img_point_lite_react/img_voxel_react_occ3d_fast_v3.py')
    for path in ['img_voxel_react_occ3d_fast_v4.py','img_voxel_react_occ3d_fast_v4_mem.py']:
        cfg = Config.fromfile('config/img_point_lite_react/'+path)
        for key in ('loss','optimizer','train_dataset_config','val_dataset_config',
                    'train_loader','val_loader','input_shape','amp','load_from'):
            assert baseline.get(key) == cfg.get(key)
        for key in ('img_backbone','img_neck','lifter','head'):
            assert baseline.model[key] == cfg.model[key]
        assert baseline.model.encoder.operation_order == cfg.model.encoder.operation_order
        assert baseline.model.encoder.ffn == cfg.model.encoder.ffn
