"""Causal OPUS-V1 temporal selection on the local sweep metadata format."""
import numpy as np

from . import OPENOCC_TRANSFORMS
from .transform_3d import LoadMultiViewImageHistory


@OPENOCC_TRANSFORMS.register_module()
class LoadOPUSV1ImageHistory(LoadMultiViewImageHistory):
    """Current frame + seven past sweeps; never crosses a scene boundary.

    Training uses the official random sweep interval [4, 8]. Validation uses
    previous keyframes. Short histories repeat the oldest selected frame;
    an empty history repeats the current frame, matching official padding.
    All historical cameras are projected from the CURRENT ego coordinate
    system by the parent loader, using their own sensor poses.
    """

    def __init__(self, num_history=7, test_mode=False, interval=(4, 8), **kwargs):
        super().__init__(num_history=num_history, pad_history=False, **kwargs)
        self.test_mode = bool(test_mode)
        self.interval = tuple(interval)
        if len(self.interval) != 2 or not 1 <= self.interval[0] <= self.interval[1]:
            raise ValueError('interval must be a positive inclusive range')

    @staticmethod
    def is_keyframe(info):
        # Local sweeps carry a token and LIDAR_TOP only on keyframes.
        return bool(info.get('token')) and 'LIDAR_TOP' in info.get('data', {})

    def select_indices(self, ctx):
        frames = ctx['scene_infos'][ctx['scene_token']]
        current = ctx['frame_index']
        previous = [i for i in range(current - 1, -1, -1)
                    if all(cam in frames[i].get('data', {})
                           for cam in ctx['sensor_types'])]
        if self.test_mode:
            selected = [i for i in previous if self.is_keyframe(frames[i])][:self.num_history]
        else:
            stride = int(np.random.randint(self.interval[0], self.interval[1] + 1))
            selected = [previous[(k + 1) * stride - 1]
                        for k in range(self.num_history)
                        if (k + 1) * stride - 1 < len(previous)]
        real_count = len(selected)
        fill = selected[-1] if selected else current
        selected += [fill] * (self.num_history - len(selected))
        return selected, real_count

    def __call__(self, results):
        ctx = dict(results['history_context'])
        indices, real_count = self.select_indices(ctx)
        ctx['history_indices'] = indices
        results['history_context'] = ctx
        results = super().__call__(results)
        results['history_frame_indices'] = np.asarray(
            [ctx['frame_index']] + indices, dtype=np.int64)
        results['num_real_history_frame'] = real_count
        if len(results['img']) != self.num_cams * (self.num_history + 1):
            raise ValueError('OPUS temporal loader did not produce T * 6 cameras')
        return results
