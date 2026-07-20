"""Local-aggregate head for grouped Gaussian-OPUS decoder outputs."""
import torch
from mmengine.registry import MODELS

from .gaussian_head import GaussianHead
from model.encoder.gaussian_encoder.utils import GaussianPrediction
from model.utils.utils import get_rotation_matrix


def _flatten_grouped(gaussian):
    return GaussianPrediction(
        means=gaussian.means.flatten(1, 2), scales=gaussian.scales.flatten(1, 2),
        rotations=gaussian.rotations.flatten(1, 2), opacities=gaussian.opacities.flatten(1, 2),
        semantics=gaussian.semantics.flatten(1, 2))


@MODELS.register_module()
class GaussianOPUSHead(GaussianHead):
    """Reuse GaussianHead's standard local_aggregate renderer.

    The Phase-A model intentionally forbids localaggprob.  ``with_empty``
    adds the renderer's global empty Gaussian, yielding 18-class logits.
    """
    def __init__(self, num_classes=18, with_empty=True, use_localaggprob=False,
                 use_localagg_react=False, apply_loss_type='all', max_render_scale=1.2,
                 **kwargs):
        if use_localaggprob or use_localagg_react or not with_empty:
            raise ValueError('GaussianOPUSHead requires local_aggregate with with_empty=True')
        super().__init__(num_classes=num_classes, with_empty=with_empty,
                         use_localaggprob=False, use_localagg_react=False,
                         apply_loss_type=apply_loss_type, **kwargs)
        self.max_render_scale = max_render_scale

    def prepare_gaussian_args(self, gaussians):
        """Batch-safe variant of GaussianHead's standard localagg arguments."""
        means, scales = gaussians.means, gaussians.scales
        rotations, semantics = gaussians.rotations, gaussians.semantics
        opacity = gaussians.opacities
        if opacity.numel() == 0:
            opacity = torch.ones_like(semantics[..., :1])
        batch = means.shape[0]
        empty_mean = self.empty_mean.to(means).expand(batch, -1, -1)
        empty_scale = self.empty_scale.to(scales).expand(batch, -1, -1)
        empty_rotation = self.empty_rot.to(rotations).expand(batch, -1, -1)
        empty_semantic = self.empty_sem.to(semantics).expand(batch, -1, -1).clone()
        empty_semantic[..., self.empty_label] += self.empty_scalar
        means = torch.cat([means, empty_mean], dim=1)
        scales = torch.cat([scales, empty_scale], dim=1)
        rotations = torch.cat([rotations, empty_rotation], dim=1)
        semantics = torch.cat([semantics, torch.zeros_like(semantics[..., :1])], dim=-1)
        semantics = torch.cat([semantics, empty_semantic], dim=1)
        opacity = torch.cat([opacity, self.empty_opa.to(opacity).expand(batch, -1, -1)], dim=1)
        transform = torch.diag_embed(scales)
        rotation_matrix = get_rotation_matrix(rotations)
        covariance = (transform @ rotation_matrix).transpose(-1, -2) @ (transform @ rotation_matrix)
        return means, opacity, semantics, scales, torch.linalg.inv(covariance)

    def _render_batched(self, sampled_xyz, means, opacity, semantics, scales, cov_inv):
        """Render a batch through the existing single-scene localagg kernel.

        The vendored CUDA extension owns a single-scene ABI and asserts a
        leading size of one.  Calling it once per sample keeps its autograd
        implementation unchanged while exposing the normal [B, C, V] head
        contract to Gaussian-OPUS.  Each call remains differentiable.
        """
        batch_size, gaussian_count = means.shape[:2]
        if sampled_xyz.shape[0] != batch_size:
            raise ValueError('sampled_xyz and Gaussian tensors must have the same batch size')
        rendered = []
        for batch_index in range(batch_size):
            logits = self.aggregator(
                sampled_xyz[batch_index:batch_index + 1].clone().float(),
                means[batch_index:batch_index + 1],
                opacity[batch_index:batch_index + 1].reshape(1, gaussian_count),
                semantics[batch_index:batch_index + 1],
                scales[batch_index:batch_index + 1],
                cov_inv[batch_index:batch_index + 1],
            )
            # Single-scene localagg returns [V, C].
            rendered.append(logits)
        return torch.stack(rendered, dim=0).transpose(1, 2).contiguous()

    def forward(self, representation, metas=None, **kwargs):
        num_decoder = len(representation)
        if not self.training:
            apply_layers = [num_decoder - 1]
        elif self.apply_loss_type == 'all':
            apply_layers = list(range(num_decoder))
        elif self.apply_loss_type == 'random':
            apply_layers = [num_decoder - 1]
        elif self.apply_loss_type == 'fixed':
            apply_layers = self.fixed_apply_loss_layers
        else:
            raise RuntimeError('unsupported Gaussian-OPUS apply_loss_type')
        occ_xyz = metas['occ_xyz'].to(self.zero_tensor.device)
        occ_label = metas['occ_label'].to(self.zero_tensor.device)
        sampled_xyz, sampled_label = self._sampling(occ_xyz, occ_label, None)
        prediction, all_gaussians = [], []
        for state in representation:
            all_gaussians.append(_flatten_grouped(state['gaussian']))
        for index in apply_layers:
            gaussian = all_gaussians[index]
            # The fixed empty Gaussian is appended inside prepare_gaussian_args
            # and is intentionally exempt.  Every decoder-predicted Gaussian
            # must remain small enough for bounded localagg binning.
            if gaussian.scales.detach().amax() > self.max_render_scale:
                raise RuntimeError(
                    f'decoder Gaussian scale exceeds {self.max_render_scale}m before local_aggregate')
            means, opacity, semantic, scales, cov_inv = self.prepare_gaussian_args(gaussian)
            prediction.append(self._render_batched(
                sampled_xyz, means, opacity, semantic, scales, cov_inv))
        occ_mask = metas.get('occ_mask')
        return {
            'pred_occ': prediction,
            'supervised_stage_indices': apply_layers,
            'sampled_label': sampled_label,
            'sampled_xyz': sampled_xyz,
            'occ_mask': occ_mask,
            'occ_loss_mask': metas.get('occ_loss_mask', occ_mask),
            'occ_cam_mask': metas.get('occ_cam_mask'),
            'occ_lidar_mask': metas.get('occ_lidar_mask'),
            'occ_nonempty_mask': metas.get('occ_nonempty_mask'),
            # pred_occ is [B, C, V], matching the existing GaussianHead API.
            'final_occ': prediction[-1].argmax(dim=1),
            'gaussian': all_gaussians[-1],
            'gaussians': all_gaussians,
        }
