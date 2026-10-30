"""Image-only BF16, explicit static grids, and probability-free DAF."""
from copy import deepcopy
import pytest
import torch
from dataset.utils import custom_collate_fn_temporal, move_batch_to_device, _OCC_GRID_CACHE
from dataset.react_voxel import CastReactImage
from dataset.transform_3d import LoadOccupancyOcc3D
from model.segmentor.image_execution import FixedImageBatchNorm2d
from model.encoder.gaussian_encoder.ops.normalized_sampling import NormalizedSampling
from model.encoder.gaussian_encoder.ops import DeformableAggregationSumFunction as DAFSum
from model.encoder.gaussian_encoder.ops.masked_softmax import normalize_weights
from model.encoder.gaussian_encoder.deformable_module import DeformableFeatureAggregation as DFA


def test_static_grid_exact_batch_cache_and_spec_change():
    spec = dict(origin=[-40.,-40.,-1.],shape=[17,13,7],resolution=.4)
    batch = custom_collate_fn_temporal([dict(static_occ_grid=spec)]*2)
    first = move_batch_to_device(batch,'cpu')['occ_xyz']
    expected = LoadOccupancyOcc3D.get_meshgrid(None,spec['origin'],spec['shape'],.4)
    assert torch.equal(first[0],torch.from_numpy(expected))
    assert first.shape == (2,17,13,7,3)
    second = move_batch_to_device(batch,'cpu')['occ_xyz']
    assert first.data_ptr() == second.data_ptr()
    changed = deepcopy(spec); changed['resolution'] = .5
    other = move_batch_to_device(dict(static_occ_grid=[changed]),'cpu')['occ_xyz']
    assert other.data_ptr() != first.data_ptr()
    with pytest.raises(ValueError):
        move_batch_to_device(dict(static_occ_grid=[spec,changed]),'cpu')
    with pytest.raises(ValueError):
        move_batch_to_device(dict(static_occ_grid=[spec],occ_xyz=first),'cpu')
    assert torch.equal(move_batch_to_device(dict(occ_xyz=first),'cpu')['occ_xyz'],first)
    _OCC_GRID_CACHE.clear()
    with torch.inference_mode():
        move_batch_to_device(batch,'cpu')
    assert not next(iter(_OCC_GRID_CACHE.values())).is_inference()
    _OCC_GRID_CACHE.clear()


@pytest.mark.parametrize('settings',[dict(perturb=True,model_coord='ego'),dict(model_coord='lidar'),dict(use_ego=True)])
def test_static_grid_rejects_dynamic_coordinates(settings):
    with pytest.raises(ValueError,match='unperturbed'):
        LoadOccupancyOcc3D('',defer_xyz=True,**settings)


def test_worker_image_cast_does_not_change_geometry():
    coords = torch.randn(2,3)
    source = torch.randn(6,3,11,17)
    result = CastReactImage()(dict(img=source,projection_mat=coords))
    assert result['img'].dtype == torch.bfloat16
    assert result['projection_mat'] is coords
    assert torch.equal(result['img'],source.bfloat16())


def test_sync_bn_conversion_preserves_fixed_image_bn_and_lidar_training_bn():
    from model.segmentor.image_execution import convert_sync_batchnorm_preserving_fixed
    fixed=FixedImageBatchNorm2d(torch.nn.BatchNorm2d(4).eval().requires_grad_(False)).eval()
    network=torch.nn.ModuleDict(dict(image=fixed,lidar=torch.nn.Sequential(torch.nn.BatchNorm1d(4))))
    keys=set(network.state_dict())
    converted=convert_sync_batchnorm_preserving_fixed(network)
    assert converted['image'] is fixed
    assert isinstance(converted['lidar'][0],torch.nn.SyncBatchNorm)
    assert converted['lidar'][0].training and converted['lidar'][0].weight.requires_grad
    assert keys==set(converted.state_dict())


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
@pytest.mark.parametrize('channels_last',[False,True])
def test_fixed_image_bn_state_dtype_layout_output_and_gradient(dtype,channels_last):
    torch.manual_seed(7)
    source = torch.nn.BatchNorm2d(19).cuda().eval().requires_grad_(False)
    with torch.no_grad():
        source.running_mean.normal_(); source.running_var.uniform_(.3,2.)
        source.weight.normal_(); source.bias.normal_()
    fixed = FixedImageBatchNorm2d(source).cuda().eval()
    with torch.inference_mode():
        fixed(torch.zeros(1,19,2,2,device='cuda',dtype=dtype))
    assert not fixed._scale.is_inference()
    assert set(source.state_dict()) == set(fixed.state_dict())
    x = torch.randn(3,19,13,17,device='cuda',dtype=dtype)
    if channels_last: x=x.contiguous(memory_format=torch.channels_last)
    x.requires_grad_()
    reference = source(x)
    actual = fixed(x)
    assert actual.dtype == dtype
    assert actual.is_contiguous(memory_format=torch.channels_last if channels_last else torch.contiguous_format)
    torch.testing.assert_close(actual,reference,atol=.032 if dtype==torch.bfloat16 else 2e-6,rtol=.008 if dtype==torch.bfloat16 else 2e-6)
    probe = torch.randn_like(x)
    a, = torch.autograd.grad(actual,x,probe)
    b, = torch.autograd.grad(reference,x,probe)
    torch.testing.assert_close(a,b,atol=.032 if dtype==torch.bfloat16 else 2e-6,rtol=.008 if dtype==torch.bfloat16 else 2e-6)
    with torch.no_grad(): source.running_mean.add_(2)
    fixed.load_state_dict(source.state_dict())
    torch.testing.assert_close(fixed(x),source(x),atol=.032 if dtype==torch.bfloat16 else 2e-6,rtol=.008 if dtype==torch.bfloat16 else 2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('dropout',[False,True])
@pytest.mark.parametrize('shape',[(2,7,3,2,2,2,8),(1,5,9,6,4,4,128),(1,0,9,6,4,4,128)])
def test_fused_sampling_outputs_all_gradients_and_all_masked(dropout,shape):
    torch.manual_seed(17)
    B,N,K,V,L,G,C=shape
    sizes=[(7,11),(4,6),(2,3),(1,2)][:L]
    starts=[sum(h*w for h,w in sizes[:l]) for l in range(L)]
    dims=torch.tensor(sizes,device='cuda',dtype=torch.int32)
    offsets=torch.tensor(starts,device='cuda',dtype=torch.int32)
    f=torch.randn(B,V,sum(h*w for h,w in sizes),C,device='cuda',requires_grad=True)
    loc=torch.rand(B,N*K,V,2,device='cuda').requires_grad_()
    visible=torch.rand(B,N,K,V,device='cuda')>.4
    if N: visible[:,0]=False
    mask=torch.rand(B,N,V,L,K,G,device='cuda')>.2
    if not dropout: mask=torch.empty(0,dtype=torch.bool,device='cuda')
    q=torch.randn(B,N,L,K,G,device='cuda',requires_grad=True)
    cam=torch.randn(B,V,L,K,G,device='cuda',requires_grad=True)
    logits=(q[:,:,None]+cam[:,None]).permute(0,1,4,2,3,5).contiguous()
    valid=visible[...,None,None].expand_as(logits)
    if dropout: valid=valid & mask.permute(0,1,4,2,3,5)
    p=normalize_weights(logits,valid,fused=True).reshape(B,N*K,V,L,G)
    reference=DAFSum.apply(f,dims,offsets,loc,p,K)
    saved=[]
    def pack(tensor):
        saved.append((tuple(tensor.shape),tensor.dtype))
        return tensor
    with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t):
        actual=NormalizedSampling.apply(f,dims,offsets,loc,visible,mask,q,cam)
    assert ((B,N*K,V,L,G),torch.float32) not in saved
    assert ((B,N,K,V,L,G),torch.float32) not in saved
    torch.testing.assert_close(actual,reference,atol=2e-6,rtol=2e-5)
    probe=torch.randn_like(actual)
    eg=torch.autograd.grad(reference,(f,loc,q,cam),probe,retain_graph=True)
    ag=torch.autograd.grad(actual,(f,loc,q,cam),probe)
    for a,b in zip(ag,eg): torch.testing.assert_close(a,b,atol=2e-5,rtol=2e-4)
    if N: assert actual[:,0].count_nonzero()==0


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_fused_module_parameters_gradients_dropout_rng_and_checkpoint():
    torch.manual_seed(17)
    cfg=dict(embed_dims=8,num_groups=2,num_levels=2,num_cams=2,attn_drop=.15,
             proj_drop=.1,use_deformable_func=True,use_camera_embed=True,
             kps_generator=dict(type='SparseGaussian3DKeyPointsGenerator',
                 num_learnable_pts=2,fix_scale=[[0.,0.,0.]],
                 pc_range=[0.,0.,0.,2.,2.,2.],scale_range=[.08,.64]))
    cfg.update(fuse_point_sum=True,batched_projection=True,factor_camera_weights=True,
               compact_weight_mask=True,fused_weight_softmax=True)
    reference=DFA(**cfg).cuda().train()
    actual=deepcopy(reference);actual.fuse_normalized_sampling=True
    features=torch.randn(2,11,8,device='cuda',requires_grad=True)
    embed=torch.randn_like(features,requires_grad=True)
    anchor=torch.randn(2,11,23,device='cuda',requires_grad=True)
    images=[torch.randn(2,2,8,9,13,device='cuda',requires_grad=True),
            torch.randn(2,2,8,5,7,device='cuda',requires_grad=True)]
    metas=dict(projection_mat=torch.eye(4,device='cuda')[None,None].repeat(2,2,1,1),
               image_wh=torch.ones(2,2,2,device='cuda')*4)
    inputs=(features,anchor,embed,images,metas)
    torch.manual_seed(123)
    expected=reference(*inputs);rng=torch.cuda.get_rng_state()
    torch.manual_seed(123)
    from torch.utils.checkpoint import checkpoint
    result=checkpoint(actual,*inputs,use_reentrant=False,preserve_rng_state=True)
    assert torch.equal(rng,torch.cuda.get_rng_state())
    torch.testing.assert_close(result,expected,atol=2e-6,rtol=2e-5)
    probe=torch.randn_like(result)
    variables=(features,anchor,embed,*images)
    eg=torch.autograd.grad(expected,variables+tuple(reference.parameters()),probe)
    ag=torch.autograd.grad(result,variables+tuple(actual.parameters()),probe)
    for a,b in zip(ag,eg): torch.testing.assert_close(a,b,atol=4e-5,rtol=5e-4)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_new_kernels_use_current_stream():
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):
        test_fixed_image_bn_state_dtype_layout_output_and_gradient(torch.bfloat16,True)
        test_fused_sampling_outputs_all_gradients_and_all_masked(True,(1,5,9,6,4,4,128))
    stream.synchronize()
