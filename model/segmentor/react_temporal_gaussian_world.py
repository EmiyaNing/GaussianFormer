"""Image-temporal React perception (img4f) followed by Gaussian forecasting.

The perceiver is the four-frame ``ReactTemporalSegmentor`` recipe moved from
the ``react_occ3d_img_history`` branch: the current camera frame plus three
past frames, six cameras each, fused by
``TemporalDeformableFeatureAggregation`` after the ten-frame LiDAR branch has
been voxelized as usual. The world decoder, the strict perception-checkpoint
protocol and the future-occupancy readout are inherited unchanged from
``ReactGaussianWorld``.
"""
from mmseg.models import SEGMENTORS

from .react_gaussian_world import ReactGaussianWorld
from .react_temporal_segmentor import ReactTemporalSegmentor


@SEGMENTORS.register_module()
class ReactTemporalGaussianWorld(ReactGaussianWorld, ReactTemporalSegmentor):
    """React world model whose perception consumes ``T`` image frames.

    Both parents derive from ``BEVSegmentor``, so the MRO keeps a single
    lifter/encoder/head execution:

    ``ReactTemporalGaussianWorld`` -> ``ReactGaussianWorld`` (checkpoint
    protocol, freeze policy, world decoder, future readout) ->
    ``ReactTemporalSegmentor`` (per-frame image extraction, history
    stop-gradient, inference feature cache) -> ``BEVSegmentor`` (shared
    forward/lifter/encoder/head).

    ``num_image_frames``, ``num_cameras``, ``detach_history``,
    ``image_checkpoint``, ``image_cache_size`` and ``pack_temporal_features``
    are forwarded to ``ReactTemporalSegmentor``; ``world_decoder``,
    ``perception_checkpoint`` and ``freeze_perception`` are consumed by
    ``ReactGaussianWorld``. ``init_weights`` therefore loads the img4f
    perception checkpoint into exactly the non-``world_decoder`` tensors.
    """
