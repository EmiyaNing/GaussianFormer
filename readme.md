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

## SparseWorldStrict

`SparseWorldStrict` is a native port for this repository's Python 3.10 and
MMCV-2/MMDet-3 stack. It does not import or launch the incompatible official
runtime. Build the vendored MSMV extension before GPU evaluation:

```bash
python3.10 -m pip install -v -e model/ops/opus_msmv_sampling
```

The configuration expects the official AD-MLP and OccWorld artifacts at
`admlp/stp3_val/data_nuscene.pkl` and
`occworld/nuscenes_infos_{train,val}_temporal_v3_scene.pkl`, plus the local
nuScenes/Occ3D data paths declared in `config/opus/sparseworld_strict.py`.

Convert the released checkpoint. The tool builds the local model, audits all
keys/shapes/dtypes, performs `strict=True` before and after serialization, and
writes a per-tensor SHA-256 manifest:

```bash
python3.10 tools/convert_sparseworld_strict_checkpoint.py \
  --config config/opus/sparseworld_strict.py \
  --src ../SparseWorld/ckpts/epoch_56.pth \
  --dst ckpts/sparseworld_strict_epoch_56.pth
```

Evaluate or train with the repository's standard entry points:

```bash
python3.10 eval.py \
  --py-config config/opus/sparseworld_strict.py \
  --resume-from ckpts/sparseworld_strict_epoch_56.pth \
  --work-dir out/sparseworld_strict_eval

python3.10 train.py \
  --py-config config/opus/sparseworld_strict.py \
  --work-dir out/sparseworld_strict_train
```

The released target is Avg mIoU **13.20** and Avg occupied IoU **22.03** over
the 0/2/4/6 horizons, without a camera or lidar evaluation mask. The CPU MSMV
fallback is intended for contract tests; full precision reproduction should
use the compiled CUDA extension.
