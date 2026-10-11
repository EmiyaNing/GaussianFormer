"""Full 16-epoch paired ablation: remove semantic KL only.

Use a new work directory. Geometry/feature losses, matcher, initialization,
freezing, optimizer, schedule, augmentation and future-born queries inherit
the original teacher-aligned recipe unchanged.
"""
_base_ = ['./sparseworld_teacher_aligned_r50_704x256_6h.py']

loss = dict(semantic_weight=0.)

# Metadata contract distinguishes this run from the completed full-KL run.
# It prevents accidentally resuming the original experiment's checkpoint.
alignment_contract_version = 3
