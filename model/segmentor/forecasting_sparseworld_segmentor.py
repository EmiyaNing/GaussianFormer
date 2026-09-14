"""ForecastOcc-style image forecasting with an OPUSv2-Tiny occupancy head."""
import os
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from mmseg.models import SEGMENTORS

from .base_segmentor import CustomBaseSegmentor


class _ForecastLayer(nn.Module):
    def __init__(self, dims=256, heads=16, mlp_ratio=4):
        super().__init__()
        self.cross = nn.MultiheadAttention(dims, heads, batch_first=True)
        self.self_attn = nn.MultiheadAttention(dims, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dims)
        self.norm2 = nn.LayerNorm(dims)
        self.norm3 = nn.LayerNorm(dims)
        self.ffn = nn.Sequential(
            nn.Linear(dims, dims * mlp_ratio), nn.ReLU(True),
            nn.Linear(dims * mlp_ratio, dims))

    def forward(self, query, context, position):
        query = self.norm1(query + self.cross(
            query + position, context + position, context, need_weights=False)[0])
        query = self.norm2(query + self.self_attn(
            query + position, query + position, query, need_weights=False)[0])
        return self.norm3(query + self.ffn(query))


class SharedForecastingModule(nn.Module):
    """Memory-bounded ForecastOcc decoder shared by all future horizons."""
    def __init__(self, dims=256, num_layers=8, num_context=6,
                 num_future=6, num_cams=6, heads=16, mlp_ratio=4,
                 with_checkpoint=True):
        super().__init__()
        self.num_context = num_context
        self.num_future = num_future
        self.num_cams = num_cams
        self.with_checkpoint = with_checkpoint
        self.layers = nn.ModuleList([
            _ForecastLayer(dims, heads, mlp_ratio) for _ in range(num_layers)])
        self.temporal_embed = nn.Parameter(torch.empty(num_context, dims))
        self.camera_embed = nn.Parameter(torch.empty(num_cams, dims))
        self.horizon_embed = nn.Parameter(torch.empty(num_future, dims))
        self.output = nn.Sequential(
            nn.Linear(dims, dims), nn.ReLU(True), nn.Linear(dims, dims))
        nn.init.normal_(self.temporal_embed, std=.02)
        nn.init.normal_(self.camera_embed, std=.02)
        nn.init.normal_(self.horizon_embed, std=.02)

    @staticmethod
    def _position(height, width, dims, device, dtype):
        y, x = torch.meshgrid(
            torch.linspace(-1., 1., height, device=device, dtype=dtype),
            torch.linspace(-1., 1., width, device=device, dtype=dtype),
            indexing='ij')
        frequency = torch.arange(max(dims // 4, 1), device=device, dtype=dtype)
        frequency = 1. / (10000 ** (frequency / max(dims // 4, 1)))
        value = torch.cat([
            torch.sin(x[..., None] * frequency), torch.cos(x[..., None] * frequency),
            torch.sin(y[..., None] * frequency), torch.cos(y[..., None] * frequency)], -1)
        if value.shape[-1] < dims:
            value = F.pad(value, (0, dims - value.shape[-1]))
        return value[..., :dims].reshape(1, height * width, dims)

    def forward(self, contexts, horizon):
        # contexts: [B,T,N,C,H,W], in chronological order.
        batch, frames, cameras, channels, height, width = contexts.shape
        if frames != self.num_context or cameras != self.num_cams:
            raise ValueError('forecast context layout does not match configuration')
        temporal = self.temporal_embed.view(1, frames, 1, channels, 1, 1)
        camera = self.camera_embed.view(1, 1, cameras, channels, 1, 1)
        encoded = contexts + temporal + camera
        query = contexts[:, -1].flatten(-2).permute(0, 1, 3, 2)
        query = query.reshape(batch * cameras, height * width, channels)
        query = query + self.horizon_embed[horizon].view(1, 1, channels)
        position = self._position(height, width, channels, query.device, query.dtype)
        position = position.expand(batch * cameras, -1, -1)
        for index, layer in enumerate(self.layers):
            context = encoded[:, index % frames].flatten(-2).permute(0, 1, 3, 2)
            context = context.reshape(batch * cameras, height * width, channels)
            if self.with_checkpoint and self.training and query.requires_grad:
                query = checkpoint(layer, query, context, position,
                                   use_reentrant=False)
            else:
                query = layer(query, context, position)
        query = self.output(query)
        return query.reshape(batch, cameras, height, width, channels).permute(0, 1, 4, 2, 3)


class FuturePyramidDecoder(nn.Module):
    def __init__(self, dims=256):
        super().__init__()
        def block():
            return nn.Sequential(nn.Conv2d(dims, dims, 3, padding=1),
                                 nn.ReLU(True), nn.Conv2d(dims, dims, 3, padding=1))
        self.blocks = nn.ModuleList([block() for _ in range(4)])

    def forward(self, latent):
        batch, cameras, channels, height, width = latent.shape
        x = latent.flatten(0, 1)
        targets = [(height * 4, width * 4), (height * 2, width * 2),
                   (height, width), ((height + 1) // 2, (width + 1) // 2)]
        outputs = []
        for index, size in enumerate(targets):
            resized = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
            value = self.blocks[index](resized)
            outputs.append(value.reshape(batch, cameras, channels, *size))
        return outputs


@SEGMENTORS.register_module()
class ForecastingSparseWorld(CustomBaseSegmentor):
    """Six-history image forecasting followed by shared OPUSv2-Tiny."""
    def __init__(self, num_history=5, num_future=6, num_cams=6,
                 freeze_img_backbone=True, freeze_img_neck=False,
                 freeze_occupancy_head=False, init_checkpoint=None,
                 current_occupancy_checkpoint=None,
                 forecast_layers=8, forecast_heads=16,
                 train_future_horizons_per_step=1,
                 forecast_curriculum_start_epoch=None,
                 forecast_curriculum_epochs_per_horizon=1,
                 observed_backbone_chunk_frames=0, **kwargs):
        super().__init__(**kwargs)
        self.num_history = int(num_history)
        self.num_future = int(num_future)
        self.num_cams = int(num_cams)
        self.freeze_img_backbone = bool(freeze_img_backbone)
        self.freeze_img_neck = bool(freeze_img_neck)
        self.freeze_occupancy_head = bool(freeze_occupancy_head)
        self.init_checkpoint = init_checkpoint
        self.current_occupancy_checkpoint = current_occupancy_checkpoint
        self.train_future_horizons_per_step = int(train_future_horizons_per_step)
        self.forecast_curriculum_start_epoch = forecast_curriculum_start_epoch
        self.forecast_curriculum_epochs_per_horizon = max(
            1, int(forecast_curriculum_epochs_per_horizon))
        self.observed_backbone_chunk_frames = int(
            observed_backbone_chunk_frames)
        self.current_epoch = 0
        self.scale_projections = nn.ModuleList([
            nn.Conv2d(256, width, 1) for width in (32, 64, 96, 64)])
        self.teacher_scale_projections = deepcopy(self.scale_projections)
        for parameter in self.teacher_scale_projections.parameters():
            parameter.requires_grad_(False)
        # A detached target generated by the same changing neck is still a
        # moving target. Keep a frozen checkpoint-initialized teacher neck.
        self.teacher_img_neck = deepcopy(self.img_neck)
        self.teacher_img_neck.eval()
        for parameter in self.teacher_img_neck.parameters():
            parameter.requires_grad_(False)
        self.forecast = SharedForecastingModule(
            dims=256, num_layers=forecast_layers,
            num_context=num_history + 1, num_future=num_future,
            num_cams=num_cams, heads=forecast_heads)
        self.pyramid_decoder = FuturePyramidDecoder(256)
        self._checkpoint_loaded = False
        self._apply_backbone_freeze()

    def _apply_backbone_freeze(self):
        if self.freeze_img_backbone:
            self.img_backbone.eval()
            for parameter in self.img_backbone.parameters():
                parameter.requires_grad_(False)
        if self.freeze_img_neck:
            self.img_neck.eval()
            for parameter in self.img_neck.parameters():
                parameter.requires_grad_(False)
        if self.freeze_occupancy_head:
            for parameter in self.head.parameters():
                parameter.requires_grad_(False)
        self.teacher_img_neck.eval()

    def train(self, mode=True):
        super().train(mode)
        self._apply_backbone_freeze()
        return self

    def init_weights(self):
        super().init_weights()
        if not self._checkpoint_loaded:
            self.teacher_scale_projections.load_state_dict(
                self.scale_projections.state_dict(), strict=True)
        if self.current_occupancy_checkpoint and not self._checkpoint_loaded:
            self._load_current_occupancy_checkpoint(
                self.current_occupancy_checkpoint)
            self._checkpoint_loaded = True
        elif self.init_checkpoint and not self._checkpoint_loaded:
            self._load_image_checkpoint(self.init_checkpoint)
            self._checkpoint_loaded = True
        self._apply_backbone_freeze()

    @torch.no_grad()
    def set_epoch(self, epoch):
        self.current_epoch = int(epoch)

    @staticmethod
    def _adapt_mixing_parameter(value, target, module):
        """Collapse a temporal OPUS mixing generator to fewer input frames.

        The generated channel-mixing parameters do not depend on frame count.
        Spatial parameters are summed over equally sized temporal groups. This
        preserves the source layer exactly when all source frames contain the
        same sampled feature, and is therefore a principled one-frame init.
        """
        groups = module.n_groups
        matrix_size = module.m_parameters
        out_points = module.out_points
        source_per_group = value.shape[0] // groups
        target_per_group = target.shape[0] // groups
        source_points = (source_per_group - matrix_size) // out_points
        target_points = (target_per_group - matrix_size) // out_points
        if (value.shape[0] % groups or target.shape[0] % groups or
                source_points * out_points + matrix_size != source_per_group or
                target_points * out_points + matrix_size != target_per_group or
                source_points % target_points):
            raise RuntimeError(
                f'cannot adapt mixing generator {tuple(value.shape)} to '
                f'{tuple(target.shape)}')
        tail = value.shape[1:]
        value = value.reshape(groups, source_per_group, *tail)
        channel = value[:, :matrix_size]
        spatial = value[:, matrix_size:].reshape(
            groups, out_points, source_points // target_points,
            target_points, *tail).sum(2)
        return torch.cat([channel, spatial.flatten(1, 2)], 1).reshape_as(target)

    def _load_current_occupancy_checkpoint(self, filename):
        if not os.path.isfile(filename):
            raise FileNotFoundError(filename)
        checkpoint_data = torch.load(filename, map_location='cpu')
        state = checkpoint_data.get('state_dict', checkpoint_data)
        own = self.state_dict()
        scoped = {key: value for key, value in state.items()
                  if key.startswith(('img_backbone.', 'img_neck.', 'head.'))}
        expected = {key for key in own
                    if key.startswith(('img_backbone.', 'img_neck.', 'head.'))}
        missing = expected - set(scoped)
        unexpected = set(scoped) - expected
        if missing or unexpected:
            raise RuntimeError(
                'current occupancy checkpoint manifest mismatch: '
                f'missing={sorted(missing)}, unexpected={sorted(unexpected)}')
        for index, layer in enumerate(self.head.transformer.decoder.decoder_layers):
            prefix = (f'head.transformer.decoder.decoder_layers.{index}.'
                      'mixing.parameter_generator.')
            for suffix in ('weight', 'bias'):
                key = prefix + suffix
                if scoped[key].shape != own[key].shape:
                    scoped[key] = self._adapt_mixing_parameter(
                        scoped[key], own[key], layer.mixing)
        bad = [key for key, value in scoped.items()
               if value.shape != own[key].shape]
        if bad:
            raise RuntimeError(f'current occupancy checkpoint shape mismatch: {bad}')
        result = self.load_state_dict(scoped, strict=False)
        if result.unexpected_keys or any(key in expected for key in result.missing_keys):
            raise RuntimeError(f'current occupancy checkpoint load failed: {result}')
        self.teacher_img_neck.load_state_dict(self.img_neck.state_dict(), strict=True)

    def _load_image_checkpoint(self, filename):
        if not os.path.isfile(filename):
            raise FileNotFoundError(filename)
        checkpoint_data = torch.load(filename, map_location='cpu')
        state = checkpoint_data.get('state_dict', checkpoint_data)
        backbone = {key[len('img_backbone.'):]: value for key, value in state.items()
                    if key.startswith('img_backbone.')}
        incompat = self.img_backbone.load_state_dict(backbone, strict=True)
        if incompat.missing_keys or incompat.unexpected_keys:
            raise RuntimeError(f'backbone checkpoint mismatch: {incompat}')
        target = self.img_neck.state_dict()
        mapped = {}
        for key, value in state.items():
            if key.startswith('img_neck.lateral_convs.'):
                suffix = key[len('img_neck.lateral_convs.'):]
                source_index, rest = suffix.split('.', 1)
                mapped[f'lateral_convs.{int(source_index) + 1}.{rest}'] = value
            elif key.startswith('img_neck.fpn_convs.'):
                suffix = key[len('img_neck.fpn_convs.'):]
                source_index, rest = suffix.split('.', 1)
                if int(source_index) < 3:
                    mapped[f'fpn_convs.{int(source_index) + 1}.{rest}'] = value
        bad = [key for key, value in mapped.items()
               if key not in target or target[key].shape != value.shape]
        if bad:
            raise RuntimeError(f'neck checkpoint mapping mismatch: {bad}')
        result = self.img_neck.load_state_dict(mapped, strict=False)
        expected_missing = {key for key in target
                            if key.startswith('lateral_convs.0.') or
                            key.startswith('fpn_convs.0.')}
        if set(result.missing_keys) != expected_missing or result.unexpected_keys:
            raise RuntimeError(
                'unexpected neck mapping manifest: '
                f'missing={result.missing_keys}, unexpected={result.unexpected_keys}')
        teacher_result = self.teacher_img_neck.load_state_dict(mapped, strict=False)
        if (set(teacher_result.missing_keys) != expected_missing or
                teacher_result.unexpected_keys):
            raise RuntimeError('teacher neck checkpoint mapping mismatch')

    def _frame_features(self, images, neck_grad=True):
        batch, cameras, channels, height, width = images.shape
        flat = images.reshape(batch * cameras, channels, height, width)
        if self.freeze_img_backbone:
            with torch.no_grad():
                backbone = self.img_backbone(flat)
        else:
            backbone = self.img_backbone(flat)
        if isinstance(backbone, dict):
            backbone = list(backbone.values())
        if neck_grad:
            pyramid = self.img_neck(backbone)
        else:
            with torch.no_grad():
                pyramid = self.teacher_img_neck(backbone)
        return [feature.reshape(batch, cameras, feature.shape[1],
                                feature.shape[2], feature.shape[3])
                for feature in pyramid]

    def _observed_features(self, frame_images, frame_indices):
        """Encode observed frames in larger batches to improve GPU occupancy."""
        batch = frame_images.shape[0]
        frame_count = len(frame_indices)
        selected = frame_images[:, frame_indices].contiguous()
        flat = selected.reshape(
            batch * frame_count * self.num_cams, *selected.shape[3:])
        chunk_frames = self.observed_backbone_chunk_frames
        if chunk_frames <= 0:
            chunk_frames = frame_count
        chunk_images = batch * self.num_cams * chunk_frames
        levels = None
        for images in flat.split(chunk_images, dim=0):
            if self.freeze_img_backbone:
                with torch.no_grad():
                    encoded = self.img_backbone(images)
            else:
                encoded = self.img_backbone(images)
            if isinstance(encoded, dict):
                encoded = list(encoded.values())
            if levels is None:
                levels = [[] for _ in encoded]
            for level, value in zip(levels, encoded):
                level.append(value)
        backbone = [torch.cat(level, dim=0) for level in levels]
        pyramid = self.img_neck(backbone)
        return [feature.reshape(
                    batch, frame_count, self.num_cams, feature.shape[1],
                    feature.shape[2], feature.shape[3])
                for feature in pyramid]

    def _fuse_pyramid(self, pyramid, teacher=False):
        height, width = pyramid[2].shape[-2:]
        pieces = []
        projections = (self.teacher_scale_projections if teacher
                       else self.scale_projections)
        for feature, projection in zip(pyramid, projections):
            flat = feature.flatten(0, 1)
            flat = F.interpolate(flat, (height, width), mode='bilinear',
                                 align_corners=False)
            projected = projection(flat)
            pieces.append(projected.reshape(*feature.shape[:2], projected.shape[1],
                                            height, width))
        return torch.cat(pieces, dim=2)

    @staticmethod
    def _alignment(prediction, target):
        huber = F.smooth_l1_loss(prediction.float(), target.float(), beta=2.)
        pred_flat = prediction.float().flatten(2)
        target_flat = target.float().flatten(2)
        cosine = 1. - F.cosine_similarity(pred_flat, target_flat, dim=2).mean()
        return huber + cosine

    def _opus_metas(self, metas):
        output = dict(metas)
        output['projection_mat'] = metas['projection_mat'][:, :self.num_cams]
        output['image_wh'] = metas['image_wh'][:, :self.num_cams]
        return output

    def _run_opus(self, pyramid, metas):
        # spconv 2.x on the local CUDA stack has a training FP16 kernel but no
        # compatible FP16 inference algorithm for the densifier.
        if self.training:
            return self.head(ms_img_feats=pyramid, metas=metas)
        with torch.cuda.amp.autocast(enabled=False):
            return self.head(ms_img_feats=[feature.float() for feature in pyramid],
                             metas=metas)

    def forward(self, imgs, metas, global_iter=0,
                oracle_future_features=False, **kwargs):
        observed = (1 + self.num_history) * self.num_cams
        expected = (1 + self.num_history + self.num_future) * self.num_cams
        selected_expected = observed + self.num_cams
        if imgs.shape[1] not in (observed, selected_expected, expected):
            raise ValueError(
                f'expected {observed}, {selected_expected}, or {expected} '
                f'training/validation images, got {imgs.shape[1]}')
        has_future_targets = imgs.shape[1] in (selected_expected, expected)
        selected_future_horizon = None
        if imgs.shape[1] == selected_expected:
            values = metas.get('future_horizon')
            if values is None:
                raise KeyError('single-future input requires future_horizon')
            values = torch.as_tensor(values, device=imgs.device).reshape(-1)
            unique = torch.unique(values)
            if unique.numel() != 1:
                raise ValueError(
                    'all samples in a single-future batch must use one horizon')
            selected_future_horizon = int(unique.item())
            if not 0 <= selected_future_horizon < self.num_future:
                raise ValueError(
                    f'invalid future_horizon={selected_future_horizon}')
        if self.training and not has_future_targets:
            raise ValueError('training requires future images for feature alignment')
        frame_images = imgs.reshape(imgs.shape[0], -1, self.num_cams,
                                    *imgs.shape[2:])
        # Loader order is current, near history -> far history, near future -> far future.
        context_indices = list(range(self.num_history, 0, -1)) + [0]
        observed_pyramid = self._observed_features(
            frame_images, context_indices)
        flat_observed_pyramid = [feature.flatten(0, 1)
                                 for feature in observed_pyramid]
        contexts = self._fuse_pyramid(flat_observed_pyramid).reshape(
            imgs.shape[0], len(context_indices), self.num_cams, 256,
            observed_pyramid[2].shape[-2], observed_pyramid[2].shape[-1])
        # Current is last because context_indices orders far history -> current.
        current_pyramid = [feature[:, -1] for feature in observed_pyramid]
        opus_metas = self._opus_metas(metas)
        current_opus = self._run_opus(current_pyramid, opus_metas)

        if self.training:
            if selected_future_horizon is not None:
                horizon_indices = [selected_future_horizon]
            else:
                count = max(1, self.train_future_horizons_per_step)
                active_horizons = self.num_future
                if self.forecast_curriculum_start_epoch is not None:
                    elapsed = self.current_epoch - int(
                        self.forecast_curriculum_start_epoch)
                    active_horizons = max(1, min(
                        self.num_future,
                        elapsed // self.forecast_curriculum_epochs_per_horizon + 1))
                horizon_indices = [int((global_iter + offset) % active_horizons)
                                   for offset in range(count)]
        else:
            horizon_indices = list(range(self.num_future))
        futures = []
        for horizon in horizon_indices:
            latent = self.forecast(contexts, horizon)
            predicted_pyramid = self.pyramid_decoder(latent)
            if has_future_targets:
                future_frame = (1 + self.num_history if
                                selected_future_horizon is not None else
                                1 + self.num_history + horizon)
                target_pyramid = self._frame_features(
                    frame_images[:, future_frame], neck_grad=False)
                latent_target = self._fuse_pyramid(
                    target_pyramid, teacher=True).detach()
                latent_loss = self._alignment(latent, latent_target)
                pyramid_loss = sum(self._alignment(pred, target.detach())
                                   for pred, target in zip(
                                       predicted_pyramid, target_pyramid)) / 4.
            else:
                latent_loss = latent.sum() * 0.
                pyramid_loss = latent_loss
            opus_pyramid = (target_pyramid if oracle_future_features and
                            has_future_targets else predicted_pyramid)
            future_opus = self._run_opus(opus_pyramid, opus_metas)
            futures.append(dict(horizon=horizon, opus=future_opus,
                                latent_loss=latent_loss,
                                pyramid_loss=pyramid_loss))
        forecasting_output = dict(current_opus=current_opus,
                                  future_predictions=futures)
        result = dict(forecasting_output=forecasting_output)
        result.update(current_opus)
        return result
