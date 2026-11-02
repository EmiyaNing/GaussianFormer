"""React image windows: bounded extraction, history stop-grad and eval cache."""

from collections import OrderedDict

import torch
from torch.utils.checkpoint import checkpoint
from mmseg.models import SEGMENTORS

from .bev_segmentor import BEVSegmentor


@SEGMENTORS.register_module()
class ReactTemporalSegmentor(BEVSegmentor):
    def __init__(self, num_image_frames=4, num_cameras=6, detach_history=True,
                 image_checkpoint=True, image_cache_size=8, pack_temporal_features=False, **kwargs):
        super().__init__(**kwargs)
        self.num_image_frames = int(num_image_frames)
        self.num_cameras = int(num_cameras)
        self.detach_history = bool(detach_history)
        self.image_checkpoint = bool(image_checkpoint)
        self.image_cache_size = int(image_cache_size)
        self.pack_temporal_features = bool(pack_temporal_features)
        if min(self.num_image_frames, self.num_cameras) < 1 or self.image_cache_size < 0:
            raise ValueError('Image window/camera counts must be positive and cache size nonnegative')
        self._image_cache = OrderedDict()
        self._image_cache_scene = None
        self._image_cache_hits = 0
        self._image_cache_misses = 0
        if self.image_checkpoint or not self.detach_history:
            # Shared backbone parameters must not trigger several reentrant
            # DDP ready hooks when multiple frames retain gradients.
            for module in self.img_backbone.modules():
                if hasattr(module, 'with_cp'):
                    module.with_cp = False

    def clear_image_cache(self, reset_stats=True):
        if hasattr(self, '_image_cache'):
            self._image_cache.clear()
            self._image_cache_scene = None
        if reset_stats:
            self._image_cache_hits = self._image_cache_misses = 0

    def image_cache_stats(self):
        size = sum(feature.numel() * feature.element_size()
                   for features in self._image_cache.values() for feature in features)
        return dict(hits=self._image_cache_hits, misses=self._image_cache_misses,
                    frames=len(self._image_cache), mib=size / 2**20)

    def train(self, mode=True):
        if mode or mode != self.training:
            self.clear_image_cache()
        return super().train(mode)

    def _apply(self, fn, recurse=True):
        self.clear_image_cache()
        return super()._apply(fn, recurse=recurse)

    def _load_from_state_dict(self, *args, **kwargs):
        self.clear_image_cache()
        return super()._load_from_state_dict(*args, **kwargs)

    def _extract_frame(self, images):
        # Retain React's local BF16 backbone autocast. The neck and features
        # remain FP32 even if a caller has enabled an outer autocast context.
        with torch.autocast(device_type=images.device.type, enabled=False):
            levels = tuple(super().extract_img_feat(images)['ms_img_feats'])
        if self.pack_temporal_features:
            from ..encoder.gaussian_encoder.temporal_ops.pyramid import pack_pyramid
            return pack_pyramid(levels)
        return levels

    def extract_img_feat(self, imgs, metas=None, **kwargs):
        batch, cameras = imgs.shape[:2]
        if cameras != self.num_image_frames * self.num_cameras:
            raise ValueError(f'Expected {self.num_image_frames} * {self.num_cameras} cameras, got {cameras}')
        use_cache = (not self.training and not torch.is_grad_enabled() and
                     batch == 1 and self.image_cache_size > 0 and metas is not None and
                     'image_frame_keys' in metas and 'image_aug_key' in metas and 'scene_token' in metas)
        if use_cache:
            scene = metas['scene_token'][0]
            if scene != self._image_cache_scene:
                self.clear_image_cache(reset_stats=False)
                self._image_cache_scene = scene

        frames = []
        for frame, images in enumerate(imgs.split(self.num_cameras, dim=1)):
            cache_key = None
            if use_cache:
                identity = tuple(metas['image_frame_keys'][0][frame])
                cache_key = (scene, identity, metas['image_aug_key'][0],
                             tuple(images.shape), images.dtype, images.device, self.pack_temporal_features)
            if cache_key is not None and cache_key in self._image_cache:
                features = self._image_cache[cache_key]
                self._image_cache.move_to_end(cache_key)
                self._image_cache_hits += 1
            else:
                keep_grad = torch.is_grad_enabled() and not (frame > 0 and self.detach_history)
                with torch.set_grad_enabled(keep_grad):
                    if self.training and keep_grad and self.image_checkpoint:
                        features = checkpoint(self._extract_frame, images,
                                              use_reentrant=False, preserve_rng_state=True)
                    else:
                        features = self._extract_frame(images)
                if cache_key is not None:
                    self._image_cache[cache_key] = tuple(feature.detach() for feature in features)
                    self._image_cache_misses += 1
            frames.append(features)
        # Trim after consuming the whole window: inserting its new current
        # frame must not evict history that this same forward still needs.
        while len(self._image_cache) > self.image_cache_size:
            self._image_cache.popitem(last=False)
        if self.pack_temporal_features:
            return dict(ms_img_feats=None, temporal_img_feats=tuple(frame[0] for frame in frames),
                        temporal_spatial_shapes=frames[0][1], temporal_level_starts=frames[0][2])
        return dict(ms_img_feats=[torch.cat(levels, dim=1) for levels in zip(*frames)])
