"""Independent C experiment paths and preserved B reference."""
from pathlib import Path

NAME = 'C_conditional_semantic_ce'
ROOT = Path('out/sparseworld_opus_gaussian_bias_conditional_ce_C_4096_1024_15e')
WORK = ROOT / NAME
ANALYSIS = Path('analysis/sparseworld_opus_gaussian_bias_conditional_ce_C_4096_1024_15e')
CONFIG = Path('config/sparseworld_opus_gaussian_bias/sparseworld_opus_gaussian_bias_fullspace_conditional_ce_C_4096_1024_15e.py')
BASE_CONFIG = Path('config/sparseworld_opus_gaussian_bias/sparseworld_opus_gaussian_bias_fullspace_ce_free_B_4096_1024_15e.py')
BASE_WORK = Path('out/sparseworld_opus_gaussian_bias_ab_first_epoch_4096_1024_15e/B_ce_free_0_1')
BASE_ANALYSIS = Path('analysis/sparseworld_opus_gaussian_bias_ab_first_epoch_4096_1024_15e')
