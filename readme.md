# GaussianFormer Clear Allocation

This branch supports training, evaluation, and visualization for only these
configurations:

- `config/nuscenes_gs25600_solid.py`
- `config/img_point_fusion/img_voxel_lite.py`
- `config/img_point_lite_react/img_voxel_react.py`
- `config/img_point_lite_react/img_voxel_react_occ3d.py`
- `config/adaptive_allocation/adaptive_allocationv6_lite.py`
- `config/opus/opusv1_t_r50_704x256_1f_occ3d.py`

Install the Python dependencies and the CUDA extensions in an environment
compatible with CUDA, PyTorch, `spconv`, and `frnn`:

```bash
pip install -r requirements.txt
pip install -r requirements-vis.txt  # needed for visualization
pip install -r requirements-opus.txt # optional accelerated OPUS KNN backends
pip install -e model/encoder/gaussian_encoder/ops
pip install -e model/head/localagg
pip install -e model/head/localagg_react
```

Place the nuScenes, SurroundOcc, Occ3D, and checkpoint data under the paths
configured by the selected config. Data, checkpoints, and outputs are ignored
by Git.

```bash
python train.py --py-config config/img_point_fusion/img_voxel_lite.py --work-dir out/img_voxel_lite
python eval.py --py-config config/img_point_fusion/img_voxel_lite.py --work-dir out/img_voxel_lite --resume-from out/img_voxel_lite/latest.pth
python visualize.py --py-config config/img_point_fusion/img_voxel_lite.py --work-dir out/img_voxel_lite --resume-from out/img_voxel_lite/latest.pth --vis-occ --vis-gaussian --num-samples 1
```

## OPUS

The OPUS baseline uses six current-frame cameras and predicts sparse occupied
points before rasterizing them to the same Occ3D grid used by evaluation and
visualization. It needs Occ3D labels at `data/occ3d/gts` and NuScenes camera
metadata at `data/nuscenes_cam`.

```bash
python train.py --py-config config/opus/opusv1_t_r50_704x256_1f_occ3d.py --work-dir out/opusv1_t_occ3d
python eval.py --py-config config/opus/opusv1_t_r50_704x256_1f_occ3d.py --work-dir out/opusv1_t_occ3d --resume-from out/opusv1_t_occ3d/latest.pth
python visualize.py --py-config config/opus/opusv1_t_r50_704x256_1f_occ3d.py --work-dir out/opusv1_t_occ3d --resume-from out/opusv1_t_occ3d/latest.pth --vis-occ --vis-opus-points --num-samples 1
```

`OPUSSetLoss` uses a deterministic 8,192-point matching cap with the built-in
PyTorch fallback to keep the first training run bounded. Install and wire a
GPU KNN backend from `requirements-opus.txt` before full 76,800-point matching
experiments.

`clarify_reop.md` documents the dependency boundaries and release verification
matrix for this branch.
