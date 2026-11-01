"""Original React perception followed by Gaussian forecasting and planning."""
from __future__ import annotations

from contextlib import nullcontext

import torch
from mmengine.registry import MODELS
from mmseg.models import SEGMENTORS

from .bev_segmentor import BEVSegmentor
from ..encoder.react_world_decoder import ReactWorldDecoder  # register the decoder
from ..head.react_world_render import render_world_gaussians


@SEGMENTORS.register_module()
class ReactGaussianWorld(BEVSegmentor):
    """Preserve perception parameters/state names and add ``world_decoder.*``.

    Only the feature-export flag is added to the perception call. The original
    image/LiDAR/encoder/densify/head path still executes without modification.
    """
    def __init__(self, world_decoder, perception_checkpoint=None,
                 freeze_perception=True, **kwargs):
        super().__init__(**kwargs)
        if not (self.head.use_localagg_react and self.head.analytic_covariance
                and self.head.separate_empty and self.head.cache_render_grid):
            raise ValueError('ReactGaussianWorld requires the v5_mem React head settings')
        self.perception_checkpoint = perception_checkpoint
        self.freeze_perception = freeze_perception
        self._perception_weights_loaded = False
        self.world_decoder = MODELS.build(world_decoder)
        self._perception_modules = (
            self.img_backbone, self.img_neck, self.lifter, self.encoder, self.head)
        if freeze_perception:
            for module in self._perception_modules:
                module.requires_grad_(False)
                module.eval()

    def init_weights(self):
        super().init_weights()
        if self.perception_checkpoint:
            self.load_perception_checkpoint(self.perception_checkpoint)

    def load_perception_checkpoint(self, path):
        """Require every original tensor; never silently freeze missing weights."""
        checkpoint = torch.load(path, map_location='cpu')
        state = checkpoint.get('state_dict', checkpoint)
        state = {key.removeprefix('module.'): value for key, value in state.items()}
        expected = {key for key in self.state_dict() if not key.startswith('world_decoder.')}
        missing, unexpected = expected - state.keys(), state.keys() - expected
        if missing or unexpected:
            raise ValueError(f'Perception checkpoint mismatch: missing={sorted(missing)}, '
                             f'unexpected={sorted(unexpected)}')
        self.load_state_dict(state, strict=False)

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        result = super().load_state_dict(state_dict, strict=strict, **kwargs)
        expected = {key for key in self.state_dict() if not key.startswith('world_decoder.')}
        if expected.issubset(state_dict):
            # A full world checkpoint can be resumed without the original
            # perception checkpoint file, which may have moved or been removed.
            self._perception_weights_loaded = True
        return result

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_perception:
            # Frozen BatchNorm statistics and dropout must stay frozen too.
            for module in self._perception_modules:
                module.eval()
        return self

    def _prepare_perception_metas(self, metas, device):
        """Permit inference without occupancy ground truth; queries are fixed."""
        from dataset.utils import static_occ_grid_to_device

        result = dict(metas)
        aggregator = self.head.aggregator
        shape = (aggregator.H, aggregator.W, aggregator.D)
        if 'occ_xyz' not in result:
            spec = dict(origin=aggregator.pc_min[0].tolist(), shape=list(shape),
                        resolution=aggregator.grid_size)
            result['occ_xyz'] = static_occ_grid_to_device([spec], device)
        if 'occ_label' not in result:
            result['occ_label'] = torch.zeros((1, *shape), dtype=torch.long, device=device)
        return result, shape

    def forward(self, imgs=None, metas=None, **kwargs):
        if self.freeze_perception and not self._perception_weights_loaded:
            raise RuntimeError(
                'Frozen perception weights have not been loaded. Set '
                'REACT_PERCEPTION_CHECKPOINT to a trained v5_mem checkpoint, '
                'or resume a complete ReactGaussianWorld checkpoint.')
        if imgs is None or metas is None or imgs.shape[0] != 1:
            raise ValueError('ReactGaussianWorld requires one current-frame sample')
        if 'temporal_ego_states' not in metas:
            raise ValueError('Current ego state is required for planning')
        perception_metas, shape = self._prepare_perception_metas(metas, imgs.device)
        context = torch.no_grad() if self.freeze_perception else nullcontext()
        with context:
            result = super().forward(
                imgs=imgs, metas=perception_metas, return_gaussian_features=True, **kwargs)

        forecast = self.world_decoder(
            result['gaussian'], result['gaussian_features'], metas['temporal_ego_states'])
        points = perception_metas['occ_xyz'].flatten(1, 3)
        render_context = self.head._react_context(points)
        poses = metas.get('future_ego_to_current')
        if poses is not None and poses.shape != (1, self.world_decoder.future_steps, 4, 4):
            raise ValueError('future_ego_to_current must contain one pose per future step')
        future_logits = []
        for step, gaussian in enumerate(forecast['future_gaussians']):
            current_to_output = (None if poses is None else
                                 torch.linalg.inv(poses[:, step].to(points)))
            future_logits.append(render_world_gaussians(
                self.head, gaussian, points, render_context, current_to_output))

        result.update(forecast)
        result.update(
            final_occ_grid=result['final_occ'].reshape(1, *shape),
            future_occ_logits=future_logits,
            future_occ_grids=[logits.argmax(1).reshape(1, *shape) for logits in future_logits],
            future_occ_frame='current_ego' if poses is None else 'future_ego')
        if 'occ_label' not in metas:
            result['sampled_label'] = None
        return result
