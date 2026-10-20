import torch

from model.lifter.semantic_gaussian_template_lifter import SemanticGaussianTemplateLifter
from model.encoder.gaussian_opus_encoder import GaussianOPUSEncoder, _next_stage_parent
from model.head.gaussian_opus_head import GaussianOPUSHead
from model.encoder.gaussian_encoder.utils import GaussianPrediction
from loss.gaussian_center_chamfer_loss import GaussianCenterChamferLoss


def test_gaussian_template_lifter_contract():
    lifter = SemanticGaussianTemplateLifter(
        num_templates=3, embed_dims=8, semantic_dim=17,
        pc_range=(0., 0., 0., 2., 2., 2.))
    output = lifter([torch.rand(2, 1, 8, 2, 2)])
    template = output['query_templates']
    assert output['query_features'].shape == (2, 3, 8)
    assert template.means.shape == (2, 3, 3)
    assert template.semantics.shape == (2, 3, 17)
    assert torch.all(template.scales > 0)
    assert torch.allclose(template.rotations.norm(dim=-1), torch.ones(2, 3))
    assert not lifter.rotation.requires_grad
    assert not lifter.opacity_logits.requires_grad
    assert not lifter.semantic_logits.requires_grad


def test_phase_a_encoder_keeps_fixed_opus_cardinality_and_gradients():
    lifter = SemanticGaussianTemplateLifter(
        num_templates=2, embed_dims=8, semantic_dim=17,
        pc_range=(0., 0., 0., 2., 2., 2.))
    encoder = GaussianOPUSEncoder(
        embed_dims=8, num_decoder=2, num_frames=1, num_views=1, num_points=4,
        num_levels=4, num_groups=4, num_heads=2, feedforward_channels=16,
        semantic_dim=17, num_refines=(1, 2), stage_steps=(1., .5),
        pc_range=(0., 0., 0., 2., 2., 2.))
    features = [torch.rand(1, 1, 8, 2, 2, requires_grad=True) for _ in range(4)]
    lifted = lifter(features)
    output = encoder(
        lifted['query_features'], lifted['query_templates'], features,
        {'projection_mat': torch.eye(4).view(1, 1, 4, 4),
         'image_wh': torch.full((1, 1, 2), 2.)})
    states = output['representation']
    assert [state['gaussian'].means.shape for state in states] == [
        (1, 2, 1, 3), (1, 2, 2, 3)]
    final = states[-1]['gaussian']
    assert encoder.layers[0].refine.semantic is None
    assert encoder.layers[-1].refine.semantic is not None
    assert torch.all(final.scales >= .15) and torch.all(final.scales < 1.)
    assert torch.allclose(final.rotations.norm(dim=-1), torch.ones_like(final.opacities.squeeze(-1)))
    (final.means.sum() + final.scales.sum() + final.rotations.sum() +
     final.opacities.sum() + final.semantics.sum()).backward()
    assert lifter.query_features.grad is not None
    assert all(feature.grad is not None for feature in features)


def test_gaussian_opus_head_batch_adapter_stacks_single_scene_renders():
    class DummyHead:
        def __init__(self):
            self.calls = []

        def aggregator(self, pts, means, opacities, semantics, scales, cov_inv):
            self.calls.append((pts.shape, means.shape, opacities.shape))
            # [V, C], while retaining a differentiable dependency on means.
            return means.mean().expand(pts.shape[1], semantics.shape[-1])

    dummy = DummyHead()
    batch, gaussians, voxels, classes = 2, 3, 4, 18
    means = torch.rand(batch, gaussians, 3, requires_grad=True)
    output = GaussianOPUSHead._render_batched(
        dummy, torch.rand(batch, voxels, 3), means,
        torch.ones(batch, gaussians, 1), torch.rand(batch, gaussians, classes),
        torch.ones(batch, gaussians, 3), torch.eye(3).view(1, 1, 3, 3).expand(batch, gaussians, -1, -1))
    assert output.shape == (batch, classes, voxels)
    assert [call[0][0] for call in dummy.calls] == [1, 1]
    output.sum().backward()
    assert means.grad is not None


def test_gaussian_center_chamfer_loss_backpropagates_to_decoder_centers():
    centers = torch.tensor([[[0.1, 0.1, 0.1], [0.8, 0.8, 0.8]]], requires_grad=True)
    gaussian = GaussianPrediction(
        means=centers, scales=torch.ones(1, 2, 3),
        rotations=torch.tensor([[[1., 0., 0., 0.], [1., 0., 0., 0.]]]),
        opacities=torch.ones(1, 2, 1), semantics=torch.zeros(1, 2, 17))
    loss, _ = GaussianCenterChamferLoss(stage_weights=(1,), lambda_center=1.0)(
        {'gaussians': [gaussian], 'metas': {
            'occ_xyz': torch.tensor([[[[0.2, 0.2, 0.2], [0.7, 0.7, 0.7]]]]),
            'occ_label': torch.tensor([[[0, 1]]]),
        }})
    loss.backward()
    assert centers.grad is not None and centers.grad.abs().sum() > 0


def test_next_stage_parent_keeps_only_geometry_graph_connected():
    means = torch.rand(1, 2, 1, 3, requires_grad=True)
    scales = torch.rand(1, 2, 1, 3, requires_grad=True)
    rotations = torch.rand(1, 2, 1, 4, requires_grad=True)
    opacities = torch.rand(1, 2, 1, 1, requires_grad=True)
    semantics = torch.rand(1, 2, 1, 17, requires_grad=True)
    parent = _next_stage_parent(GaussianPrediction(
        means, scales, rotations, opacities, semantics))

    (parent.means.sum() + parent.scales.sum()).backward()
    assert means.grad is not None and scales.grad is not None
    assert not parent.rotations.requires_grad
    assert not parent.opacities.requires_grad
    assert not parent.semantics.requires_grad
