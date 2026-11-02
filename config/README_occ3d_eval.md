# Occ3D 评估范围

普通 Occ3D 配置默认 `occ3d_eval_mask = 'none'`，在整个有效标签网格计算指标。数据中含有 `occ_cam_mask` 不会自动启用过滤。

训练期间验证、独立 `eval.py` 和 `--eval-all` 使用相同的选择规则：**命令行 > 配置中的 `occ3d_eval_mask` > 显式旧开关 `eval_mask_flag` > 默认 none**。解析后的配置和日志会记录实际评估范围。

显式启用 camera mask：

```bash
python3.10 eval.py --py-config config/img_point_lite_react/img_voxel_react_occ3d_fast_v5_mem.py \
  --resume-from /path/to/epoch_1.pth --occ3d-eval-mask camera
```

训练期间验证同样支持在 `train.py` 命令中添加 `--occ3d-eval-mask camera`；也可以在配置中设置：

```python
occ3d_eval_mask = 'camera'
```

历史配置如果已经显式写了 `'camera'`，仍按该设置评估。对这些配置强制使用全网格，添加 `--occ3d-eval-mask none`。另外保留 `lidar`、`nonempty` 两种显式选项。

这是评估范围设置；训练损失中的 `occ_loss_mask` / `loss_input_convertion` 由原训练配置决定。历史实验的 camera 指标仍按原定义解释，不能将改变默认值当作历史结果已经重新评估。
