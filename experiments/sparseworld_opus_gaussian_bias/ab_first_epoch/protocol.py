"""Paths and concrete independent A/B hypotheses."""
from pathlib import Path

ROOT = Path('out/sparseworld_opus_gaussian_bias_ab_first_epoch_4096_1024_15e')
ANALYSIS = Path('analysis/sparseworld_opus_gaussian_bias_ab_first_epoch_4096_1024_15e')
BASE_CONFIG = Path('config/sparseworld_opus_gaussian_bias/sparseworld_opus_gaussian_bias_fullspace_4096_1024_15e.py')
ARMS = {
    'A_softmax_focal': Path('config/sparseworld_opus_gaussian_bias/sparseworld_opus_gaussian_bias_fullspace_focal_A_4096_1024_15e.py'),
    'B_ce_free_0_1': Path('config/sparseworld_opus_gaussian_bias/sparseworld_opus_gaussian_bias_fullspace_ce_free_B_4096_1024_15e.py'),
}
