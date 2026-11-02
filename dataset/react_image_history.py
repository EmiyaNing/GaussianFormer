"""Causal camera windows with cache identities and explicit padding masks."""

import numpy as np

from . import OPENOCC_TRANSFORMS
from .opus_temporal_history import LoadOPUSV1ImageHistory
from .transform_3d import LoadMultiViewImageHistory


@OPENOCC_TRANSFORMS.register_module()
class LoadReactImageHistory(LoadOPUSV1ImageHistory):
    """Select OPUS-style past frames and project from the current ego frame.

    Training samples a sweep stride in ``interval``; evaluation uses past
    keyframes, allowing sequential feature-cache reuse. Missing frames copy
    the oldest available image for a fixed tensor shape but are masked out
    of aggregation. Each real image is decoded only once per window.
    """

    def __call__(self, results):
        context = dict(results['history_context'])
        selected, real_count = self.select_indices(context)
        context['history_indices'] = selected[:real_count]
        results['history_context'] = context
        # The shared loader constructs historical ego2img/lidar2img from the
        # original historical camera poses, before image augmentations.
        results = LoadMultiViewImageHistory.__call__(self, results)

        for _ in range(self.num_history - real_count):
            begin = real_count * self.num_cams  # oldest real frame, or current
            end = begin + self.num_cams
            for key in ('img', 'lidar2img', 'ego2img'):
                results[key].extend([value.copy() for value in results[key][begin:end]])
            results['img_filename'].extend(results['img_filename'][begin:end])

        indices = [context['frame_index']] + selected
        frames = context['scene_infos'][context['scene_token']]
        timestamp = frames[indices[0]]['timestamp']
        results['image_frame_indices'] = np.asarray(indices, dtype=np.int64)
        results['image_time_offsets'] = np.asarray(
            [(timestamp - frames[index]['timestamp']) / 1e6 for index in indices],
            dtype=np.float32)
        results['image_frame_valid'] = np.asarray(
            [True] + [index < real_count for index in range(self.num_history)], dtype=bool)
        # A camera file identity is independent of the current ego pose. Cache
        # image features across ego frames, while recomputing projections.
        results['image_frame_keys'] = [
            tuple(results['img_filename'][start:start + self.num_cams])
            for start in range(0, len(results['img_filename']), self.num_cams)]
        results['image_aug_key'] = repr(results.get('aug_configs'))
        results['scene_token'] = context['scene_token']
        results['num_real_history_frame'] = real_count
        results['img_shape'] = [image.shape[:2] for image in results['img']]
        expected = self.num_cams * (self.num_history + 1)
        if len(results['img']) != expected or len(results['image_frame_keys']) != len(indices):
            raise ValueError('React history must contain exactly T * num_cams images')
        return results
