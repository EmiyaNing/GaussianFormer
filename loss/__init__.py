from mmengine.registry import Registry
OPENOCC_LOSS = Registry('openocc_loss')

from .multi_loss import MultiLoss
from .occupancy_loss import OccupancyLoss
from .opus_set_loss import OPUSSetLoss
from .opus_v2_loss import OPUSV2Loss
from .gaussian_point_occupancy_loss import GaussianPointOccupancyLoss
from .gaussian_occupancy_loss import GaussianOccupancyLoss
from .gaussian_center_chamfer_loss import GaussianCenterChamferLoss
from .adaptive_allocation_v6_loss import AdaptiveAllocationV6Loss
