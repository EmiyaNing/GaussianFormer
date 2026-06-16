from mmengine.registry import Registry
OPENOCC_LOSS = Registry('openocc_loss')

from .multi_loss import MultiLoss
from .occupancy_loss import OccupancyLoss
from .bce_loss import BinaryCrossEntropyLoss, PixelDistributionLoss
from .gaussian_converageloss import GaussianCoverageLoss
from .gaussian_semantic_loss import GaussianSemanticLoss
from .gaussian_semantic_mult_loss import GaussianSemanticMultLoss
from .gaussian_densify_loss import GaussianDensifyLoss