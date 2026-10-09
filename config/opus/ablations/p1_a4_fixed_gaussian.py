"""P1/A4: Gaussian kernel with FIXED isotropic scale and opacity.

Plan reference: ``plans/sparse_render_plans/P1_frozen_backbone_semantic_validation.md``
T1.1, which lists A4 as "高斯核，固定各向同性尺度/不透明度 — 用 A5 冻结参数即可".

A4 completes the ablation *chain* A2 -> A4 -> A5.  It is not a cell of a
2x2: the fourth combination (uniform weights that still use the geometry) does
not exist, because uniform weights make the geometry irrelevant.  The chain
separates two things that the
A5-vs-A2 comparison confounds:

* **A4 vs A2** — Gaussian distance weighting against uniform KNN-8 averaging,
  both with a *global* geometry.  Isolates the *kernel form* (confounded with the
  neighbour selection, which A4 keeps from A5).
* **A5 vs A4** — per-point, query-conditioned geometry against a **single global**
  per-axis scale + global opacity.  Isolates the *per-point* parameterisation.

It matters for the verdict, not just for completeness.  A5 finished last in the
control matrix, which is consistent with two very different stories: "the
Gaussian kernel adds nothing beyond uniform smoothing" (A4 ~ A2) or "the kernel
form helps and the learnable parameterisation was the mistake" (A4 best).  Only
A4 distinguishes them.

Implementation: ``freeze_geometry`` sets ``requires_grad_(False)`` on the two
heads' **weights**, which are zero-initialised.  That removes the input-dependent
term, so every point shares one geometry -- but the **biases stay trainable**, so
A4 still *learns* that shared geometry (measured at epoch 40: scale
0.403/0.405/0.407 per axis, opacity 0.0186).  A4 is therefore "global learned
isotropic", not "frozen at 0.35/0.02"; the exact numbers are pinned by
``tests/test_report_numbers.py::test_p1_7_a4_geometry_is_global_not_frozen``.
"""
_base_ = ['./p1_a5_gaussian.py']

model = dict(semantic_gaussian=dict(freeze_geometry=True))
