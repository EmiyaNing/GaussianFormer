"""Exact SparseWorld evaluation entry-point for :mod:`eval.py`.

The model graph is the released ``SparseWorld4DTraj`` graph in the sibling
checkout, not an approximation constructed from GaussianFormer OPUS modules.
``eval.py`` detects this flag before importing local MMEngine components and
delegates to ``tools/sparseworld_official_compat.py``.
"""
sparseworld_official_compat = True
official_root = '../SparseWorld'
official_config = 'configs/sparseworld/nuscenes-temporal/sparseworld-traj-finetune.py'
official_checkpoint = '../SparseWorld/ckpts/epoch_56.pth'
strict_checkpoint = 'ckpts/sparseworld_official_epoch_56_compat.pth'
# Optional additional arguments forwarded to SparseWorld/tools/test_temporal.py.
official_eval_args = []
