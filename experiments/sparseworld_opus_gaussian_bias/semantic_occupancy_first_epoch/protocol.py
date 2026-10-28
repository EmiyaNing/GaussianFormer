from pathlib import Path


ROOT = Path('out/sparseworld_opus_gaussian_bias_semantic_occupancy_DE_4096_1024_15e')
ANALYSIS = Path('analysis/sparseworld_opus_gaussian_bias_semantic_occupancy_DE_4096_1024_15e')
C_CONFIG = Path('config/sparseworld_opus_gaussian_bias/sparseworld_opus_gaussian_bias_fullspace_conditional_ce_C_4096_1024_15e.py')
C_WORK = Path('out/sparseworld_opus_gaussian_bias_conditional_ce_C_4096_1024_15e/C_conditional_semantic_ce')
C_ANALYSIS = Path('analysis/sparseworld_opus_gaussian_bias_conditional_ce_C_4096_1024_15e')
D = 'D_balanced_conditional_ce'
E = 'E_factorized_occupancy'
ARMS = {
    D: Path('config/sparseworld_opus_gaussian_bias/sparseworld_opus_gaussian_bias_balanced_conditional_D_4096_1024_15e.py'),
    E: Path('config/sparseworld_opus_gaussian_bias/sparseworld_opus_gaussian_bias_factorized_occupancy_E_4096_1024_15e.py'),
}
NEW_BRANCH_PREFIX = 'pts_bbox_head.transformer.encoder.occupancy_branches.'

