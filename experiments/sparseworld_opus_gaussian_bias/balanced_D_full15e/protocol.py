from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / 'out/sparseworld_opus_gaussian_bias_balanced_D_full15e_4096_1024'
ANALYSIS = REPO / 'analysis/sparseworld_opus_gaussian_bias_balanced_D_full15e_4096_1024'
WORK = ROOT / 'D_balanced_conditional_ce'
SOURCE_ROOT = REPO / 'out/sparseworld_opus_gaussian_bias_semantic_occupancy_DE_4096_1024_15e'
SOURCE_WORK = SOURCE_ROOT / 'D_balanced_conditional_ce'
SOURCE_ANALYSIS = REPO / 'analysis/sparseworld_opus_gaussian_bias_semantic_occupancy_DE_4096_1024_15e'
SOURCE_CONFIG = REPO / 'config/sparseworld_opus_gaussian_bias/sparseworld_opus_gaussian_bias_balanced_conditional_D_4096_1024_15e.py'
CONFIG = REPO / 'config/sparseworld_opus_gaussian_bias/sparseworld_opus_gaussian_bias_balanced_conditional_D_continue_4096_1024_15e.py'
STRICT = REPO / 'out/sparseworld_opus_gaussian_bias_4096_1024_15e/sparseworld_strict'

