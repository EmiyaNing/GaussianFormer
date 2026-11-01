"""Current-frame React inputs plus future occupancy and planning supervision."""
import torch

from . import OPENOCC_DATASET
from .sparseworld_strict import NuScenesSparseWorldStrictDataset


@OPENOCC_DATASET.register_module()
class NuScenesReactWorldDataset(NuScenesSparseWorldStrictDataset):
    """Reuse token-matched targets without loading historical camera frames.

    Planned waypoints are the future ego origins expressed in current ego
    coordinates, in metres. This avoids mixing the released trajectory file's
    LiDAR convention with the React perception's ego coordinate convention.
    """
    def __init__(self, history_frames=0, **kwargs):
        if history_frames != 0:
            raise ValueError('This version only loads the current sensor frame')
        super().__init__(history_frames=0, **kwargs)

    def __getitem__(self, index):
        result = super().__getitem__(index)
        poses = torch.as_tensor(result['future_ego_to_current'])
        result['temporal_trajs'] = poses[:, :2, 3].clone()
        return result
