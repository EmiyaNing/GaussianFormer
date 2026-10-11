"""Memory bounded eight-frame image extraction, with all frames trainable."""
import torch
import json
import os
import time
from torch.utils.checkpoint import checkpoint
from mmseg.models import SEGMENTORS

from .opus_segmentor import OPUSSegmentor


@SEGMENTORS.register_module()
class OPUSTemporalSegmentor(OPUSSegmentor):
    def __init__(self, image_chunk_size=6, image_checkpoint=True,
                 image_amp_dtype='bfloat16', **kwargs):
        super().__init__(**kwargs)
        self.image_chunk_size = int(image_chunk_size)
        self.image_checkpoint = bool(image_checkpoint)
        if self.image_chunk_size < 1:
            raise ValueError('image_chunk_size must be positive')
        if image_amp_dtype not in ('bfloat16', 'float32'):
            raise ValueError('use bfloat16 or float32 image extraction')
        self.image_amp_dtype = image_amp_dtype
        self._memory_phase = None
        self._memory_samples = 0

    def flush_memory_phase(self):
        path = os.environ.get('OPUS_TEMPORAL_MEMORY_FILE')
        if path and self._memory_phase is not None:
            torch.cuda.synchronize()
            row = dict(time=time.time(), phase=self._memory_phase,
                       samples=self._memory_samples,
                       peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                       peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
            with open(path, 'a') as handle:
                handle.write(json.dumps(row) + '\n')
            self._memory_phase = None

    def forward(self, *args, **kwargs):
        if os.environ.get('OPUS_TEMPORAL_MEMORY_FILE'):
            phase = 'train' if self.training else 'val'
            if phase != self._memory_phase:
                self.flush_memory_phase()
                torch.cuda.reset_peak_memory_stats()
                self._memory_phase = phase
                self._memory_samples = 0
            self._memory_samples += 1
        return super().forward(*args, **kwargs)

    def _extract_chunk(self, images):
        # BF16 needs no loss scaling and its exponent range preserves small
        # history gradients. Keep it confined to the image backbone.
        with torch.autocast(device_type=images.device.type, dtype=torch.bfloat16,
                            enabled=self.image_amp_dtype == 'bfloat16'):
            if self.freeze_img_backbone:
                with torch.no_grad():
                    backbone = self.img_backbone(images)
            else:
                backbone = self.img_backbone(images)
        if isinstance(backbone, dict):
            backbone = list(backbone.values())
        # Older PyTorch CUDA kernels do not implement nearest-neighbour
        # upsampling for BF16. Casting only the neck inputs is insufficient
        # under autocast: FPN's lateral convolutions would produce BF16 again.
        # Run the entire neck in FP32, including checkpoint recomputation.
        with torch.autocast(device_type=images.device.type, enabled=False):
            features = [backbone[i].float() for i in self.img_backbone_out_indices]
            if self.freeze_img_neck:
                with torch.no_grad():
                    features = self.img_neck(features)
            else:
                features = self.img_neck(features)
            return tuple(feature.float() for feature in features)

    def extract_img_feat(self, imgs, **kwargs):
        batch, cameras, channels, height, width = imgs.shape
        flat = imgs.reshape(batch * cameras, channels, height, width)
        chunks = []
        for images in flat.split(self.image_chunk_size):
            if self.training and torch.is_grad_enabled() and self.image_checkpoint:
                features = checkpoint(self._extract_chunk, images,
                                      use_reentrant=False, preserve_rng_state=True)
            else:
                features = self._extract_chunk(images)
            chunks.append(features)
        levels = [torch.cat(parts, dim=0) for parts in zip(*chunks)]
        return {'ms_img_feats': [feature.reshape(batch, cameras, *feature.shape[1:])
                                 for feature in levels]}
