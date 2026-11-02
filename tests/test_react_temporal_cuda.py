"""Temporal CUDA equivalence: layouts, RNG, coordinates and all trainable inputs."""

from copy import deepcopy
import os

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from model.encoder.gaussian_encoder.deformable_module import DeformableFeatureAggregation as DFA
from model.encoder.gaussian_encoder.ops.normalized_sampling import NormalizedSampling
from model.encoder.gaussian_encoder.temporal_deformable import TemporalDeformableFeatureAggregation
from model.encoder.gaussian_encoder.temporal_ops.pyramid import pack_pyramid
from model.encoder.gaussian_encoder.temporal_ops.sampling import packed_dropout, temporal_sampling


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def unpack_bits(bits, count):
    if not bits.numel():
        return torch.ones(count, device=bits.device, dtype=torch.bool)
    index = torch.arange(count, device=bits.device)
    return ((bits[index // 32].long() >> (index % 32)) & 1).bool()


@pytest.mark.parametrize('count', [0, 1, 31, 32, 33, 1024, 1000003])
@pytest.mark.parametrize('probability', [0., .15, 1.])
def test_packed_dropout_matches_torch_draws_and_generator_state(count, probability):
    reference = torch.empty(0, device='cuda', dtype=torch.float32)
    torch.cuda.manual_seed(1928)
    expected = (torch.rand(count, device='cuda') > probability if probability else
                torch.ones(count, device='cuda', dtype=torch.bool))
    expected_state = torch.cuda.get_rng_state()
    torch.cuda.manual_seed(1928)
    bits = packed_dropout(reference, count, probability)
    torch.testing.assert_close(torch.cuda.get_rng_state(), expected_state)
    assert torch.equal(unpack_bits(bits, count), expected)
    assert bits.numel() == ((count + 31) // 32 if probability else 0)


@pytest.mark.skipif(os.environ.get('RUN_LARGE_TEMPORAL_RNG') != '1',
                    reason='Opt-in 2--3.5 GB torch.rand reference')
@pytest.mark.parametrize('count', [(1 << 29) + 17, 60098 * 66 * 4 * 9 * 4,
                                  100000 * 60 * 4 * 9 * 4])
def test_large_dropout_matches_real_torch_split_iterator(count):
    torch.cuda.manual_seed(1928)
    reference = torch.rand(count, device='cuda')
    state = torch.cuda.get_rng_state()
    torch.cuda.manual_seed(1928)
    bits = packed_dropout(reference, count, .15)
    assert torch.equal(torch.cuda.get_rng_state(), state)
    indices = torch.cat([
        torch.arange(1 << 20, device='cuda'),
        torch.arange(count // 2 - 64, count // 2 + 64, device='cuda'),
        torch.arange(count - (1 << 20), count, device='cuda')])
    actual = ((bits[indices // 32].long() >> (indices % 32)) & 1).bool()
    assert torch.equal(actual, reference[indices] > .15)


@pytest.mark.parametrize('channels_last', [False, True])
@pytest.mark.parametrize('batch', [1, 2])
def test_pyramid_pack_transpose_and_backward_are_exact(channels_last, batch):
    levels = []
    for h, w in [(9, 13), (5, 7), (3, 4), (2, 2)]:
        feature = torch.randn(batch * 6, 8, h, w, device='cuda')
        if channels_last:
            feature = feature.contiguous(memory_format=torch.channels_last)
        levels.append(feature.view(batch, 6, 8, h, w).requires_grad_())
    expected = torch.cat([level.flatten(3) for level in levels], dim=-1).permute(0, 1, 3, 2).contiguous()
    actual, shapes, starts = pack_pyramid(levels)
    assert torch.equal(actual, expected)
    assert shapes.tolist() == [[9, 13], [5, 7], [3, 4], [2, 2]]
    assert starts.tolist() == [0, 117, 152, 164]
    probe = torch.randn_like(actual)
    expected_grads = torch.autograd.grad(expected, levels, probe)
    actual_grads = torch.autograd.grad(actual, levels, probe)
    for a, b, original in zip(actual_grads, expected_grads, levels):
        assert torch.equal(a, b)
        assert a.stride() == original.stride()


@pytest.mark.parametrize('learnable', ['projection', 'image_wh'])
def test_learnable_camera_metadata_is_rejected_explicitly(learnable):
    projection = torch.eye(4, device='cuda')[None, None].requires_grad_(learnable == 'projection')
    wh = torch.ones(1, 1, 2, device='cuda', requires_grad=learnable == 'image_wh')
    query = torch.zeros(1, 1, 1, 1, 1, device='cuda')
    with pytest.raises(ValueError, match='fixed camera metadata'):
        temporal_sampling(torch.zeros(1, 1, 1, 3, device='cuda'), projection, wh,
                          torch.ones(1, 1, device='cuda', dtype=torch.bool),
                          torch.ones(1, 2, device='cuda', dtype=torch.int32),
                          torch.zeros(1, device='cuda', dtype=torch.int32), query, query,
                          [torch.zeros(1, 1, 1, 1, device='cuda')])


@pytest.mark.parametrize('all_frame_grad', [False, True])
@pytest.mark.parametrize('probability', [0., .15])
@pytest.mark.parametrize('dimensions', [(2, 7, 3, 4, 2, 2, 2, 8),
                                      (1, 33, 9, 4, 6, 4, 4, 128),
                                      (1, 0, 9, 4, 6, 4, 4, 128),
                                      (1, 5, 9, 10, 6, 4, 4, 128),
                                      (1, 5, 9, 11, 6, 4, 4, 128),
                                      (1, 5, 2, 2, 3, 1, 1, 7)])
def test_projection_sampling_and_selective_backward(dimensions, probability, all_frame_grad):
    torch.manual_seed(928)
    B, N, K, T, V0, L, G, C = dimensions
    V = T * V0
    sizes = [(7, 11), (4, 6), (2, 3), (1, 2)][:L]
    S = sum(h * w for h, w in sizes)
    shapes = torch.tensor(sizes, device='cuda', dtype=torch.int32)
    starts = torch.tensor([sum(h * w for h, w in sizes[:l]) for l in range(L)],
                          device='cuda', dtype=torch.int32)
    frames = [torch.randn(B, V0, S, C, device='cuda', requires_grad=t == 0 or all_frame_grad)
              for t in range(T)]
    points = torch.rand(B, N, K, 3, device='cuda') + .3
    points[..., 2] += 1.
    if N:
        points[:, 0, :, 0] = -100  # every view misses the first query
    points.requires_grad_()
    projection = torch.eye(4, device='cuda')[None, None].repeat(B, V, 1, 1)
    projection += torch.randn_like(projection) * .08
    projection[:, :, 0, 0] += 3.
    projection[:, :, 1, 1] += 3.
    projection[:, :, 0, 3] = torch.linspace(-2., 1., V, device='cuda')
    wh = torch.ones(B, V, 2, device='cuda') * 4
    valid = torch.rand(B, V, device='cuda') > .2
    valid[:, -V0:] = False  # padded last time step
    query = torch.randn(B, N, L, K, G, device='cuda', requires_grad=True)
    camera = torch.randn(B, V, L, K, G, device='cuda', requires_grad=True)
    locations, visible = DFA.project_points_batched(points, projection, wh)
    visible = visible & valid[:, :, None, None]
    locations = locations.permute(0, 2, 3, 1, 4).reshape(B, N * K, V, 2)
    torch.cuda.manual_seed(1828)
    mask = (torch.rand(B, N, V, L, K, G, device='cuda') > probability if probability else
            torch.empty(0, device='cuda', dtype=torch.bool))
    expected = NormalizedSampling.apply(torch.cat(frames, 1), shapes, starts, locations,
                                        visible.permute(0, 2, 3, 1), mask, query, camera)
    state = torch.cuda.get_rng_state()
    torch.cuda.manual_seed(1828)
    actual = temporal_sampling(points, projection, wh, valid, shapes, starts,
                                query, camera, frames, probability)
    assert torch.equal(torch.cuda.get_rng_state(), state)
    torch.testing.assert_close(actual, expected, atol=4e-6, rtol=4e-5)
    inputs = [points, query, camera] + [frame for frame in frames if frame.requires_grad]
    probe = torch.randn_like(actual)
    expected_grads = torch.autograd.grad(expected, inputs, probe)
    actual_grads = torch.autograd.grad(actual, inputs, probe)
    for a, b in zip(actual_grads, expected_grads):
        torch.testing.assert_close(a, b, atol=6e-5, rtol=4e-4)
    if N:
        assert actual[:, 0].count_nonzero() == 0


@pytest.mark.parametrize('recompute', [False, True])
def test_temporal_module_gradients_dropout_rng_and_checkpoint(recompute):
    torch.manual_seed(928)
    options = dict(num_image_frames=4, num_cameras=2, embed_dims=8, num_groups=2,
                   num_levels=2, use_deformable_func=True, use_camera_embed=True,
                   fuse_point_sum=True, batched_projection=True, factor_camera_weights=True,
                   compact_weight_mask=True, fused_weight_softmax=True,
                   fuse_normalized_sampling=True, attn_drop=.15, proj_drop=.1, residual_mode='cat',
                   kps_generator=dict(type='SparseGaussian3DKeyPointsGenerator', embed_dims=8,
                                      num_learnable_pts=1, fix_scale=[[0., 0., 0.]],
                                      pc_range=[0., 0., 1., 2., 2., 3.], scale_range=[.08, .64]))
    reference = TemporalDeformableFeatureAggregation(**options).cuda().train()
    actual = deepcopy(reference)
    actual.fuse_temporal_sampling = True
    feature = torch.randn(1, 17, 8, device='cuda', requires_grad=True)
    embedding = torch.randn_like(feature, requires_grad=True)
    anchor = torch.zeros(1, 17, 23, device='cuda', requires_grad=True)
    frames = [[torch.randn(1, 2, 8, 7, 9, device='cuda', requires_grad=t == 0),
               torch.randn(1, 2, 8, 4, 5, device='cuda', requires_grad=t == 0)] for t in range(4)]
    images = [torch.cat([frame[l] for frame in frames], 1) for l in range(2)]
    packed = [pack_pyramid(frame) for frame in frames]
    metas = dict(projection_mat=torch.eye(4, device='cuda')[None, None].repeat(1, 8, 1, 1),
                 image_wh=torch.ones(1, 8, 2, device='cuda') * 4,
                 image_frame_valid=torch.tensor([[True, True, True, False]], device='cuda'))
    torch.cuda.manual_seed(1928)
    expected = reference(feature, anchor, embedding, images, metas)
    state = torch.cuda.get_rng_state()
    torch.cuda.manual_seed(1928)
    call = checkpoint if recompute else lambda function, *args, **kwargs: function(*args, **kwargs)
    kwargs = dict(temporal_img_feats=tuple(frame[0] for frame in packed),
                  temporal_spatial_shapes=packed[0][1], temporal_level_starts=packed[0][2])
    if recompute:
        kwargs.update(use_reentrant=False, preserve_rng_state=True)
    output = call(actual, feature, anchor, embedding, None, metas, **kwargs)
    assert torch.equal(torch.cuda.get_rng_state(), state)
    torch.testing.assert_close(output, expected, atol=3e-6, rtol=4e-5)
    probe = torch.randn_like(output)
    extra_inputs = [feature, anchor, embedding, *frames[0]]
    eg = torch.autograd.grad(expected, [*reference.parameters(), *extra_inputs], probe)
    ag = torch.autograd.grad(output, [*actual.parameters(), *extra_inputs], probe)
    for a, b in zip(ag, eg):
        torch.testing.assert_close(a, b, atol=6e-5, rtol=4e-4)
