# 目标配置的干净分支方案

## 1. 目标和边界

创建一个新的、可独立训练和评测的分支，仅服务于以下五个配置：

1. `config/nuscenes_gs25600_solid.py`（用户写作 `Nuscenes_gs25600_soild.py`，仓库实际文件名为 `solid`）。
2. `config/img_point_fusion/img_voxel_lite.py`。
3. `config/img_point_lite_react/img_voxel_react.py`。
4. `config/img_point_lite_react/img_voxel_react_occ3d.py`。
5. `config/adaptive_allocation/adaptive_allocationv6_lite.py`（用户写作 `adaptive_allocation6_lite.py`，仓库实际名含 `v6`）。

这里的“仅包含”应理解为只保留支持上述配置训练、评测、可视化、编译 CUDA 算子、安装和最小回归测试所必需的代码及文档；不承诺保留与目标配置无关的历史实验、GaussianFormer-2、LitePT 或其他配置。

不应从当前工作目录直接执行清理。当前分支为 `allocationv6`，工作区含未提交且与 V6 相关的文件和修改，也有无关的统计、计划、编辑器等改动。清理必须在新的 worktree 中进行，不能用 `git reset --hard` 或在原工作区批量删除。

## 2. 已确认的依赖结论

### 配置继承

所有五项配置都依赖：

- `config/_base_/misc.py`
- `config/_base_/model.py`
- `config/_base_/surroundocc.py`

`adaptive_allocationv6_lite.py` 还继承 `config/adaptive_allocation/adaptive_allocationv5_lite.py`。该父配置必须保留，即使运行时的 densify 类型最终被 V6 覆盖。

### 可视化路径

目标配置的 checkpoint 通过 `visualize.py` 进入可视化。该入口默认优先导入 `vis_open3d_voxel.py`，并回退到 `vis.py`；它依赖 `visualize_color_utils.py`、`open3d_vis_utils.py`，且其流式可视化模式依赖 `dataset/dataset_flow.py` 和 `fifo_eval.py`。这些不是训练/eval 的最小依赖，但属于本分支的交付范围，不能在裁剪时删除。

保留的可视化能力包括：occupancy、高斯、top-down、高斯点、预测误差、GT/高斯匹配和 Open3D 场景导出。`visualize.py` 中仅面向 `AdaptiveAllocationV5` 的统计 hook 在 V6 上会自动禁用；保留该代码不代表 V6 已有同等的 allocation-operation 可视化。

### 两条模型路径

- `solid` 使用 `GaussianLifter` 和普通 `local_aggregate` CUDA 扩展。
- `img_voxel_lite` 使用 `GaussianVoxelLearnear`、`TopkDensifyModule`、LiDAR 体素化和普通 `local_aggregate`。
- 两个 `react` 配置使用 `GaussianVoxelLearnear`、`DensifyOnly`、LiDAR 体素化和 `local_aggregate_react` CUDA 扩展；`img_voxel_react_occ3d.py` 还要求 Occ3D 数据加载与 `Metric_mIoU`。
- Allocation V6 使用 `GaussianVoxelLearnear`、`AdaptiveAllocationV6` 和普通 `local_aggregate`。其 V6 辅助损失依赖 `allocation_aux` 从 encoder 传递到 `GaussianHead`，再传给 `AdaptiveAllocationV6Loss`。

因此最小分支仍需要 `spconv`/`cumm`、`frnn`（React 的 `DensifyOnly` 顶层导入）以及两个本地 CUDA 聚合扩展。不要保留或编译 `localagg_prob*`，它们属于未纳入目标的 probabilistic 配置。

## 3. 建议分支建立流程

### 阶段 A：先固化 V6 的未提交必要改动

以当前 `HEAD` (`63a6440`) 为基准，先在独立 worktree 内整理一个“运行基线”提交。这个提交必须包含：

- `loss/adaptive_allocation_v6_loss.py`：目前为未跟踪文件，是 V6 配置中 `AdaptiveAllocationV6Loss` 的实现。
- `loss/__init__.py`：注册 `AdaptiveAllocationV6Loss`。
- `loss/multi_loss.py`：支持 V6 损失返回 `(loss, metrics)`，否则训练会把 tuple 当作标量相加而失败。
- `train.py` 当前的验证明细汇总改动：不是构建 V6 所必需，但用于记录 V6 指标；建议保留并在提交说明中标注为 observability。

`model/encoder/gaussian_encoder/gaussian_encoder.py`、`model/head/gaussian_head.py` 和 `model/encoder/gaussian_encoder/topk_module/adaptive_allocationv6.py` 已在 `HEAD` 中包含 V6 的 `allocation_aux` 链路，不能回退或遗漏。

建议命令顺序如下，实际执行前需确认上述未提交改动就是要交付的版本：

```bash
git worktree add ../GaussianFormer-cleanup-base HEAD
cd ../GaussianFormer-cleanup-base
git switch -c clear_allocation
# 以明确的 git add 路径加入“阶段 A”文件，再创建基线提交。
```

不要使用 `git stash -u` 作为唯一备份；它会混合用户当前所有无关修改。应以文件白名单方式复制/应用上述 V6 文件及 diff。

### 阶段 B：先做白名单副本，再删除

在 `clear_allocation` 内，不要从完整仓库直接 `rm -rf`。先把白名单复制到临时目录或用 `git archive` 建立候选树，运行验证后才替换分支内容。建议按下面的清单 `git add -f`/保留，并将其余已跟踪内容分批删除。

#### 入口、配置、训练和评测

```text
train.py
eval.py
visualize.py
fifo_eval.py
one_cycle_lr.py
config/_base_/misc.py
config/_base_/model.py
config/_base_/surroundocc.py
config/nuscenes_gs25600_solid.py
config/img_point_fusion/img_voxel_lite.py
config/img_point_lite_react/img_voxel_react.py
config/img_point_lite_react/img_voxel_react_occ3d.py
config/adaptive_allocation/adaptive_allocationv5_lite.py
config/adaptive_allocation/adaptive_allocationv6_lite.py
```

建议在新分支新增一个唯一的 `requirements.txt`，将 `docs/installation.md` 中的版本改写为可执行依赖，而非保留整份旧文档。至少锁定并在 CI 镜像中验证：Python 3.8、PyTorch 2.0/CUDA 11.8、mmcv 2.0.1、mmdet 3.0.0、mmsegmentation 1.0.0、mmdet3d 1.1.1、`spconv-cu117`、`timm`、`pyquaternion`、`tensorboard`、`frnn`。`frnn` 和 CUDA/PyTorch 的 wheel ABI 必须在目标机器上实测，而不能假定 pip 会自动匹配。

将可视化依赖拆为 `requirements-vis.txt` 或 `.[vis]` extra，至少包含 `open3d`、`matplotlib`、`Pillow`、`psutil`、`opencv-python`、`tqdm`；若保留 Mayavi 回退路径，还应包含 `mayavi`、`pyvirtualdisplay` 和与部署环境匹配的 Qt 运行时。Open3D 的 GUI 初始化在无显示环境可能失败，因此发布环境至少要验证离屏渲染或明确提供 X/Wayland 的运行说明。

#### 数据、变换、指标和日志

```text
dataset/__init__.py
dataset/dataset.py
dataset/dataset_flow.py               # visualize.py 的 --stream / --fifo 模式
dataset/transform_3d.py
dataset/utils.py
dataset/sampler.py                    # 仅 train.py 的 --iter-resume 路径需要
loss/__init__.py
loss/base_loss.py
loss/multi_loss.py
loss/occupancy_loss.py
loss/utils/lovasz_softmax.py
loss/adaptive_allocation_v6_loss.py
misc/checkpoint_util.py
misc/metric_util.py
misc/occ3d_nus_metrics.py
misc/tb_wrapper.py
```

#### 可视化模块和资源

```text
vis.py
vis_open3d.py
vis_open3d_voxel.py
vis_surround_dataset.py
open3d_vis_utils.py
visualize_color_utils.py
draw_nuscene_dataset.py
draw_sence.py
assets/
```

其中 `visualize.py`、`vis_open3d_voxel.py`、`vis.py` 和 `visualize_color_utils.py` 是目标 checkpoint 的主可视化链路，必须作为同一组验证。`open3d_vis_utils.py`、`vis_surround_dataset.py`、`draw_nuscene_dataset.py` 和 `draw_sence.py` 是数据/场景辅助可视化工具，也应保留；它们需要前述 Open3D、OpenCV 与 tqdm extra。`vis_open3d.py` 是独立 Open3D 导出工具，保留以支持无 Mayavi 环境。

`dataset/transform_3d.py` 当前把目标所需的 SurroundOcc 与 Occ3D transforms 和不需要的 transforms 放在同一文件中。第一轮可整体保留；第二轮再拆成 `transforms/common.py`、`surroundocc.py`、`occ3d.py`，只保留 `LoadMultiViewImageFromFiles`、`LoadOccupancySurroundOcc`、`LoadOccupancyOcc3D`、`ResizeCropFlipImage`、`PhotoMetricDistortionMultiViewImage`、`NormalizeMultiviewImage`、`DefaultFormatBundle`、`NuScenesAdaptor`。拆分后必须重复第 5 节验证。

#### 模型核心

```text
model/__init__.py
model/segmentor/__init__.py
model/segmentor/base_segmentor.py
model/segmentor/bev_segmentor.py
model/backbone/__init__.py
model/neck/__init__.py
model/utils/safe_ops.py
model/utils/utils.py
model/lifter/__init__.py
model/lifter/base_lifter.py
model/lifter/gaussian_lifter.py
model/lifter/voxel_gaussian_lifter.py
model/lifter/lidar_processor.py
model/lifter/spconv_voxelize.py
model/lifter/spconv_backbone.py
model/lifter/spconv_utils.py
model/encoder/__init__.py
model/encoder/base_encoder.py
model/encoder/gaussian_encoder/__init__.py
model/encoder/gaussian_encoder/utils.py
model/encoder/gaussian_encoder/anchor_encoder_module.py
model/encoder/gaussian_encoder/deformable_module.py
model/encoder/gaussian_encoder/refine_module.py
model/encoder/gaussian_encoder/ffn_module.py
model/encoder/gaussian_encoder/gaussian_encoder.py
model/encoder/gaussian_encoder/spconv3d_module.py
model/encoder/gaussian_encoder/topk_densify_module.py
model/encoder/gaussian_encoder/topk_module/topk_densify_v2.py
model/encoder/gaussian_encoder/topk_module/adaptive_allocationv6.py
model/head/__init__.py
model/head/base_head.py
model/head/gaussian_head.py
```

#### 必须保留的 CUDA 源码

保留以下目录的 `setup.py`、Python package、`ext.cpp`、`.cu`、`.h` 和 `src/`：

```text
model/encoder/gaussian_encoder/ops/
model/head/localagg/
model/head/localagg_react/
```

不要提交本机生成的 `build/`、`.so`、`*.egg-info/`。在干净环境中分别执行：

```bash
pip install -e model/encoder/gaussian_encoder/ops
pip install -e model/head/localagg
pip install -e model/head/localagg_react
```

### 阶段 C：收窄注册导入

现有的 `model/__init__.py` 和各子包 `__init__.py` 是“全量导入”模式。例如 `model/lifter/__init__.py` 会导入 LitePT、点云算子等所有 lifter；`model/encoder/gaussian_encoder/__init__.py` 会导入历史、查询、V4/V5 等实验模块。若删除实现文件而不改这些入口，`import model` 会先失败，甚至不会到达目标配置的构建阶段。

在删除非白名单文件的同一个提交中，将注册入口精简为下列类及其必要基类，保持导入顺序为“依赖组件先，`GaussianOccEncoder`/`BEVSegmentor` 后”：

```text
BEVSegmentor, CustomBaseSegmentor
GaussianLifter, GaussianVoxelLearnear, BaseLifter
SparseGaussian3DEncoder, SparseGaussian3DKeyPointsGenerator
DeformableFeatureAggregation, SparseGaussian3DRefinementModule
AsymmetricFFN, SparseConv3D, GaussianOccEncoder
TopkDensifyModule, DensifyOnly, AdaptiveAllocationV6
GaussianHead, BaseTaskHead
```

同样将 `loss/__init__.py` 仅保留 `MultiLoss`、`OccupancyLoss`、`AdaptiveAllocationV6Loss`。这样可以删除 `LitePT/`、`model/backbone_img/`、所有 `localagg_prob*`、其他 lifter/encoder/densify 试验模块及其配置，而不会造成注册时的旁路依赖。

### 阶段 D：提交组织

建议不要把所有删除压成一次提交：

1. `feat(v6): add auxiliary allocation loss and metrics compatibility`：阶段 A 的 V6 未提交必要内容。
2. `chore(config): retain five supported experiment configs`：配置与运行文档/requirements。
3. `refactor(registry): restrict imports to supported models`：先让精简注册表通过 import/build。
4. `chore(cleanup): remove unsupported experiments and extensions`：删除其它代码与资产。
5. `test: add config build and one-batch smoke coverage`：加入第 5 节测试。

这样每一步都可独立 `git bisect`，也能清楚地区分“新增 V6 功能”与“仓库裁剪”。

## 4. 明确排除项

在第 5 节全数通过后，可删除：

- 其它 `config/**`，包括 `prob/`、`gs_*`、`point_only/`、`dynamic_window/`、非目标 `img_point_*` 与 V4/V5 配置。
- `LitePT/` 及 pointops、pointrope、pointgroup_ops、sparsehash。
- `model/head/localagg_prob/`、`model/head/localagg_prob_fast/`、其他未列 lifter、encoder、backbone_img 与 Gaussian initializer。
- stream/history/entropy loader、`eval_stream.py`；但保留 `dataset_flow.py` 与 `fifo_eval.py`，因为 `visualize.py` 的流式可视化依赖它们。
- 统计、分析、旧计划和结果文件。可视化脚本、绘制工具和 `assets/` 不在删除范围内。
- `gaussian_statistic*`、`tests/test_gaussian_statistic_mixed.py`，除非另有统计功能交付要求。

数据集、checkpoint 和输出目录必须继续在 `.gitignore` 中排除：`data/`、`ckpts/`、`out/`，且不能为了让 smoke test 通过把它们提交到分支。

## 5. 验收矩阵

以下检查必须在新建的干净环境和独立 worktree 中完成。没有数据时可完成 1--4；有一份合法样本和 GPU/CUDA 环境后必须完成 5--8。

1. 静态清单：`git status --short` 只出现本方案预期文件；`git ls-files` 不含 `LitePT`、`localagg_prob`、未支持配置、`build` 或 `.so`。
2. 配置解析：对五个目标配置分别运行 `mmengine.Config.fromfile(...)`，确认 `_base_` 全部解析，V6 合并后 densify 类型为 `AdaptiveAllocationV6`。
3. 注册导入：在无数据环境运行 `python -c 'import model; import dataset; import loss'`，确保没有被已删除模块或 LitePT 旁路导入阻塞。
4. CUDA 安装：在新环境重新执行三个 `pip install -e`，再执行一次 `import local_aggregate` 与 `import local_aggregate_react`；验证 deformable aggregation extension 可被 `model.encoder.gaussian_encoder.ops` 导入。
5. 模型 build：五个配置分别执行 `build_segmentor(cfg.model)` 与 `init_weights()`。React/V6 的 build 同时验证 `spconv` 和 `cumm` 可用。
6. 数据一批：分别从 SurroundOcc 三个配置及 Occ3D 配置创建 dataloader，取一批并检查所有 `return_keys` 存在。React/V6 要检查 `lidar_points`；Occ3D 要检查四种 mask、`occ_xyz`、`occ_label`。
7. 前向和反向：每个配置至少一批执行 forward；训练配置执行 `loss.backward()` 和一次 optimizer step。V6 必须确认结果中有非空 `allocation_aux`，并记录 `AdaptiveAllocationV6Loss/*` 指标。
8. 端到端：每个配置至少以极小数据子集完成 `train.py` 的一个 epoch/若干 iteration，并以产生的 checkpoint 执行 `eval.py`。SurroundOcc 检查 `MeanIoU`，Occ3D 检查 `Metric_mIoU` 和 `occ3d_eval_mask=camera`。
9. 可视化：用每个目标配置的 checkpoint 至少执行一次 `visualize.py --vis-occ --vis-gaussian --num-samples 1`，确认输出目录生成 occupancy 与 Gaussian 图/点云，且没有导入回退后的未定义函数。对至少一个 SurroundOcc 配置再覆盖 `--vis-gaussian-point`、`--vis-occ-error` 和 `--vis-gaussian-match`；对至少一个配置覆盖 Open3D 离屏环境。若交付流式模式，再执行 `--stream` 与 `--fifo` 各一小段场景，确认 `NuScenesFlowDataset`、`FIFOQueue` 和输出的 `stream_vis_results.json` 正常工作。

建议将 2、3、5 写成 CPU/无数据 CI 测试；将 4、6、7、8、9 作为带 CUDA 和私有数据挂载的 nightly 或发布前 job。由于这些配置均依赖本地数据和 CUDA 扩展，单靠 Python import 测试不能证明“可以正常训练、eval 和可视化”。

## 6. 最终交付状态

分支名为 `clear_allocation`。合入前应包含：精简后的 README（列五个支持配置、三条扩展安装命令和可视化命令）、`requirements.txt`、`requirements-vis.txt` 或等价 extra、数据目录契约、上述验收脚本，以及五个配置实际路径。不要为了兼容用户输入而复制一份拼写错误的 `Nuscenes_gs25600_soild.py` 或 `adaptive_allocation6_lite.py`；README 可以说明它们对应的真实文件名，避免形成两个会漂移的配置入口。
