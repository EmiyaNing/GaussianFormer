# visualize.py 自适应 Gaussian 颜色可视化实现工作流

## 目标

为 `visualize.py` 增加一种新的 Gaussian 可视化模式：不再完全使用 NuScenes 语义 colormap，而是根据每个 Gaussian 的 opacity 自适应选择颜色。

该功能参考当前 `--vis-gaussian` 的渲染路径，保持 Gaussian 的几何过滤、opacity 处理、保存目录和普通/流式可视化流程不变，只替换最终渲染颜色。

## 当前代码路径

1. `visualize.py` 负责加载模型、跑验证集、拿到 `result_dict['gaussian']`。
2. 普通可视化入口：
   - `visualize.py` 中 `if args.vis_gaussian:` 调用 `save_gaussian(save_dir, result_dict['gaussian'], ...)`。
3. 流式可视化入口：
   - `main_stream()` 中同样通过 `args.vis_gaussian` 调用 `save_gaussian(...)`，如果启用 Gaussian FIFO 融合，还会对融合后的 Gaussian 再调用一次。
4. 实际渲染函数优先来自 `vis_open3d_voxel.py`：
   - `save_gaussian(...)`：Open3D 椭球网格渲染。
   - `save_gaussian_point(...)`：Open3D 点云式 Gaussian 渲染。
5. 如果 Open3D 导入失败，`visualize.py` 会 fallback 到 `vis.py` 中的 Mayavi/Matplotlib 版本 `save_gaussian(...)`。

因此建议把核心颜色选择逻辑放在可视化工具函数侧，并在 `visualize.py` 增加参数控制，这样普通模式、流式模式和 fallback 路径都能复用同一套规则。

## 新增 CLI 参数

在 `visualize.py` 的 argparse 中增加一个开关：

```python
parser.add_argument(
    '--vis-gaussian-adaptive-color',
    action='store_true',
    default=False,
    help='根据 Gaussian opacity 自适应选择红/蓝/灰颜色'
)
```

然后在构造 `draw_gaussian_params` 后追加：

```python
draw_gaussian_params['adaptive_color'] = args.vis_gaussian_adaptive_color
draw_gaussian_params['adaptive_color_seed'] = args.seed
```

如果后续也希望 `save_gaussian_point()` 支持这套颜色规则，需要在调用点同样透传 `adaptive_color` 和 `adaptive_color_seed`。

## 颜色规则定义

对每个有效 Gaussian：

1. 获取 opacity：
   - `opas = gaussian.opacities[0].squeeze()`
   - adaptive 模式下必须保留真实 opacity，不能因为 `ignore_opa=True` 把它改成 1。
2. 获取语义类别：
   - `pred = np.argmax(gaussian.semantics[0], axis=-1)`
   - 类别只用于过滤 empty label，不参与颜色决策。
3. 颜色常量建议使用 RGB float：
   - 灰色：`[0.5, 0.5, 0.5]`
   - 红色：`[1.0, 0.0, 0.0]`
   - 蓝色：`[0.0, 0.25, 1.0]`

规则：

```text
opacity > 0.7:
  随机一半红色，另一半蓝色

opacity <= 0.7:
  灰色
```

注意边界建议使用“严格大于”为高 opacity，其余进入低 opacity 分支，即：

```python
high_opacity = opas > 0.7
```

## 推荐实现函数

在 `vis_open3d_voxel.py` 增加一个纯 numpy 工具函数：

```python
def get_adaptive_gaussian_colors(pred, scales, opacities=None, seed=42):
    gray = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    red = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    blue = np.array([0.0, 0.25, 1.0], dtype=np.float32)

    pred = pred.astype(np.int64)
    opacities = np.ones(len(pred), dtype=np.float32) if opacities is None else opacities
    colors = np.tile(gray[None, :], (len(pred), 1))

    high_opacity_indices = np.where(opacities > 0.7)[0]
    rng = np.random.default_rng(seed)
    rng.shuffle(high_opacity_indices)
    red_count = len(high_opacity_indices) // 2
    colors[high_opacity_indices[:red_count]] = red
    colors[high_opacity_indices[red_count:]] = blue

    return colors
```

为了 fallback 一致，`vis.py` 中也应增加同名函数，或从一个轻量公共模块导入。更稳妥的做法是新增 `visualize_color_utils.py`，让 `vis_open3d_voxel.py` 和 `vis.py` 都从中导入，避免两份规则漂移。

## 修改 save_gaussian

在 `vis_open3d_voxel.py::save_gaussian()` 函数签名中增加参数：

```python
def save_gaussian(
    save_dir,
    gaussian_data,
    name,
    scalar=1.5,
    ignore_opa=False,
    filter_zsize=False,
    show_window=True,
    max_gaussians=25600,
    adaptive_color=False,
    adaptive_color_seed=42,
):
```

在完成 `means/scales/rotations/opas/pred` 的 mask 过滤之后、进入 for 循环之前计算：

```python
adaptive_colors = None
if adaptive_color:
    adaptive_colors = get_adaptive_gaussian_colors(
        pred,
        scales,
        opas,
        seed=adaptive_color_seed,
    )
```

循环内替换 base color 来源：

```python
if adaptive_colors is not None:
    base_color = adaptive_colors[idx].copy()
else:
    base_color = sem_cmap[pred[idx]][:3].copy()
```

adaptive 模式下不要使用原 `vis-gaussian` 中的 opacity 混白/透明度衰减逻辑：

```python
if adaptive_colors is not None:
    color = base_color
else:
    color = base_color * opa_val + np.array([1.0, 1.0, 1.0]) * (1.0 - opa_val)
```

`vis.py` fallback 中同理，adaptive 模式下 `alpha` 使用 `1.0`，非 adaptive 模式继续使用原来的 `opas[indx]`。

## 修改 save_gaussian_point

如果需要点云式 Gaussian 同步支持，给 `vis_open3d_voxel.py::save_gaussian_point()` 增加相同参数，并在采样 `max_gaussians` 之后计算 `adaptive_colors`。

注意：如果函数内部先随机采样高斯，再计算 adaptive colors，则随机红/蓝二分作用于采样后的可见集合。建议在采样后计算，保证屏幕上看到的高 opacity Gaussian 约一半红、一半蓝。

## 修改 vis.py fallback

`vis.py::save_gaussian()` 也应增加：

```python
adaptive_color=False,
adaptive_color_seed=42,
```

在 mask 过滤后计算 `adaptive_colors`。原函数最后 `ax.plot_surface(...)` 当前使用 colormap/语义颜色时，把颜色来源改成：

```python
if adaptive_colors is not None:
    face_color = adaptive_colors[indx]
else:
    face_color = sem_cmap[pred[indx]][:3]
```

然后传入 `color=face_color` 或 `facecolors`，保持 alpha 使用原来的 `opas[indx]`。

## visualize.py 调用点

普通模式中当前调用：

```python
save_gaussian(
    save_dir,
    result_dict['gaussian'],
    f'val_{i_iter_val}_gaussian',
    **draw_gaussian_params)
```

不需要额外改动，只要 `draw_gaussian_params` 里包含新参数即可。

`vis_gaussian_each_stage` 同理会自动拿到新参数。

流式模式中同样检查所有 `save_gaussian(...)` 调用是否都使用了 `**draw_gaussian_params`。若有单独手写参数的调用，需要补上：

```python
adaptive_color=args.vis_gaussian_adaptive_color,
adaptive_color_seed=args.seed,
```

## 随机二分的可复现策略

当前 adaptive color 会对 `opacity > 0.7` 的 Gaussian 做随机二分。为了便于对比实验，随机过程应由 `adaptive_color_seed` 控制。

建议：

1. 默认使用 `args.seed` 作为 `adaptive_color_seed`。
2. adaptive 模式下不要执行 `opas[:] = 1.0`，否则会破坏 `opacity > 0.7` 的判定。
3. adaptive 模式下不要使用 opacity 混白或 alpha 衰减，否则低 opacity 的灰色会被进一步淡化，无法准确表达规则。

## 验证流程

1. 语法检查：

```bash
python -m py_compile visualize.py vis_open3d_voxel.py vis.py visualize_color_utils.py
```

2. 小样本运行：

```bash
python visualize.py \
  --py-config <config> \
  --work-dir <work_dir> \
  --resume-from <ckpt> \
  --vis-gaussian \
  --vis-gaussian-adaptive-color \
  --vis-index 0 \
  --num-samples 1
```

3. 检查输出：
   - `opacity > 0.7` 的 Gaussian 应约一半红色、另一半蓝色。
   - `opacity <= 0.7` 的 Gaussian 应为灰色。
   - adaptive 模式下红/蓝/灰颜色不应再被 opacity 混白或透明化。

4. 可选调试统计：

在 `adaptive_color=True` 时打印每类颜色数量：

```python
print(
    '[adaptive_color]',
    'red=', int((adaptive_colors == red).all(axis=1).sum()),
    'blue=', int((adaptive_colors == blue).all(axis=1).sum()),
    'gray=', int((adaptive_colors == gray).all(axis=1).sum()),
)
```

正式版本可以去掉该打印，或只在 verbose 参数开启时输出。

## 风险点

1. adaptive 模式下 `ignore_opa=True` 不能覆盖真实 opacity，否则所有 Gaussian 都会被当成高 opacity。
2. `adaptive_color_seed` 会影响高 opacity Gaussian 的红/蓝随机分配，做可视化对比时应固定 seed。
3. Open3D 主路径和 `vis.py` fallback 都要支持新参数，否则 fallback 时会因为未知关键字报错。
4. 如果有 `save_gaussian_point()` 调用路径，也要同步签名，否则透传 `draw_gaussian_params` 时可能报未知关键字。
5. `max_gaussians` 随机采样会改变最终可见集合。建议采样后再做 adaptive color，让屏幕上看到的高 opacity 集合满足随机红/蓝二分。
