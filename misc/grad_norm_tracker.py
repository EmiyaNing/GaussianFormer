"""Gradient Norm Tracker —— 统计 my_model 各组件梯度范数并输出 Markdown 分析报告。

提供:
  - compute_param_grad_norm(): 计算一组参数的总 L2 梯度范数
  - get_model_grad_norms(): 从 raw_model 提取各组件梯度范数
  - get_model_param_counts(): 从 raw_model 提取各组件参数量
  - GradNormTracker: 收集迭代级数据，计算统计量，生成 grad_norm_analysis.md
"""

import os
import time
import torch
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def compute_param_grad_norm(parameters, norm_type: float = 2.0) -> float:
    """计算给定参数的总 L2 梯度范数（与 torch.nn.utils.clip_grad_norm_ 一致）。

    公式: total_norm = sqrt( sum( ||p.grad||_2^2 ) )

    Args:
        parameters: 参数迭代器或单个 Tensor
        norm_type: 范数类型，默认 2.0 (L2)
    Returns:
        float: 总梯度范数
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    if len(parameters) == 0:
        return 0.0
    device = parameters[0].grad.device
    total_norm = torch.norm(
        torch.stack([
            torch.norm(p.grad.detach(), norm_type).to(device)
            for p in parameters
        ]),
        norm_type
    )
    return total_norm.item()


def get_model_grad_norms(model) -> Dict[str, float]:
    """从 raw_model（DDP 解包后的 BEVSegmentor）提取所有组件的梯度范数。

    分组策略:
    - 顶层: img_backbone, img_neck, lifter, head
    - encoder: anchor_encoder + 按 operation_order 中 op 类型聚合的 layers
    - 可选: extra_img_backbone

    返回字典如:
    {
        'img_backbone': 1.23, 'img_neck': 0.45, 'lifter': 2.34,
        'encoder.anchor_encoder': 0.12,
        'encoder.spconv': 3.45, 'encoder.norm': 0.08, 'encoder.ffn': 4.56,
        'encoder.deformable': 2.78, 'encoder.refine': 1.90, 'encoder.densify': 0.67,
        'head': 5.01, 'total': 9.87
    }
    """
    norms: Dict[str, float] = {}

    # ---- 顶层组件 ----
    for comp_name in ['img_backbone', 'img_neck', 'lifter', 'head']:
        comp = getattr(model, comp_name, None)
        if comp is not None:
            norms[comp_name] = compute_param_grad_norm(comp.parameters())

    # 可选的第二个 backbone
    if hasattr(model, 'extra_img_backbone') and model.extra_img_backbone is not None:
        norms['extra_img_backbone'] = compute_param_grad_norm(
            model.extra_img_backbone.parameters())

    # ---- Encoder 组件 ----
    encoder = getattr(model, 'encoder', None)
    if encoder is not None:
        # anchor_encoder
        if hasattr(encoder, 'anchor_encoder') and encoder.anchor_encoder is not None:
            norms['encoder.anchor_encoder'] = compute_param_grad_norm(
                encoder.anchor_encoder.parameters())

        # layers 按 operation 类型聚合
        if hasattr(encoder, 'operation_order') and hasattr(encoder, 'layers'):
            op_params: Dict[str, List[torch.Tensor]] = defaultdict(list)
            for op, layer in zip(encoder.operation_order, encoder.layers):
                if layer is not None:
                    for p in layer.parameters():
                        if p.grad is not None:
                            op_params[op].append(p)

            for op, params in op_params.items():
                norms[f'encoder.{op}'] = compute_param_grad_norm(params)

    # ---- 总梯度范数 ----
    norms['total'] = compute_param_grad_norm(model.parameters())

    return norms


def get_model_param_counts(model) -> Dict[str, int]:
    """从 raw_model 提取各组件的参数量（与 get_model_grad_norms 使用相同的分组策略）。

    返回字典如:
    {
        'img_backbone': 23456789, 'img_neck': 123456, 'lifter': 456789,
        'encoder.anchor_encoder': 12345,
        'encoder.spconv': 234567, 'encoder.norm': 512, 'encoder.ffn': 456789,
        'encoder.deformable': 345678, 'encoder.refine': 234567, 'encoder.densify': 123456,
        'head': 567890, 'total': 45678901
    }
    """
    counts: Dict[str, int] = {}

    def _count_params(module) -> int:
        return sum(p.numel() for p in module.parameters())

    # ---- 顶层组件 ----
    for comp_name in ['img_backbone', 'img_neck', 'lifter', 'head']:
        comp = getattr(model, comp_name, None)
        if comp is not None:
            counts[comp_name] = _count_params(comp)

    # 可选的第二个 backbone
    if hasattr(model, 'extra_img_backbone') and model.extra_img_backbone is not None:
        counts['extra_img_backbone'] = _count_params(model.extra_img_backbone)

    # ---- Encoder 组件 ----
    encoder = getattr(model, 'encoder', None)
    if encoder is not None:
        # anchor_encoder
        if hasattr(encoder, 'anchor_encoder') and encoder.anchor_encoder is not None:
            counts['encoder.anchor_encoder'] = _count_params(encoder.anchor_encoder)

        # layers 按 operation 类型聚合
        if hasattr(encoder, 'operation_order') and hasattr(encoder, 'layers'):
            op_counts: Dict[str, int] = defaultdict(int)
            for op, layer in zip(encoder.operation_order, encoder.layers):
                if layer is not None:
                    op_counts[op] += _count_params(layer)

            for op, cnt in op_counts.items():
                counts[f'encoder.{op}'] = cnt

    # ---- 总参数量 ----
    counts['total'] = _count_params(model)

    return counts


# ---------------------------------------------------------------------------
# GradNormTracker
# ---------------------------------------------------------------------------

class GradNormTracker:
    """梯度范数追踪器。

    使用方法:
        tracker = GradNormTracker()
        tracker.set_param_counts(get_model_param_counts(raw_model))
        for iteration in training:
            ...
            loss.backward()
            comp_norms = get_model_grad_norms(raw_model)
            tracker.add(comp_norms)
            ...
        tracker.write_markdown_report('grad_norm_analysis.md')
    """

    def __init__(self, model_name: str = 'BEVSegmentor',
                 config_path: Optional[str] = None):
        self.model_name = model_name
        self.config_path = config_path
        self.history: Dict[str, List[float]] = defaultdict(list)  # 组件名 → grad_norm 列表
        self.param_counts: Dict[str, int] = {}  # 组件名 → 参数量
        self._init_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

    def set_param_counts(self, counts: Dict[str, int]) -> None:
        """设置各组件的参数量（用于梯度/参数平衡分析）。"""
        self.param_counts = counts

    def add(self, norms_dict: Dict[str, float]) -> None:
        """添加一次迭代的各组件梯度范数数据。"""
        for name, value in norms_dict.items():
            self.history[name].append(value)

    def compute_statistics(self) -> Dict:
        """计算所有组件的统计量。

        Returns:
            {
                'component_name': {
                    'count': int, 'mean': float, 'std': float,
                    'min': float, 'max': float,
                    'p50': float, 'p90': float, 'p95': float, 'p99': float,
                    'ratio': float  # 占总梯度的比例
                },
                ...
            }
        """
        if not self.history:
            return {}

        total_mean = np.mean(self.history['total']) if 'total' in self.history else 1.0

        stats = {}
        for name, values in self.history.items():
            if not values:
                continue
            arr = np.array(values, dtype=np.float64)
            comp_stats = {
                'count': len(arr),
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'min': float(np.min(arr)),
                'max': float(np.max(arr)),
                'p50': float(np.percentile(arr, 50)),
                'p90': float(np.percentile(arr, 90)),
                'p95': float(np.percentile(arr, 95)),
                'p99': float(np.percentile(arr, 99)),
            }
            # 计算占比（相对 total 的均值）
            if name != 'total' and total_mean > 0:
                comp_stats['ratio'] = comp_stats['mean'] / total_mean
            elif name == 'total':
                comp_stats['ratio'] = 1.0
            else:
                comp_stats['ratio'] = 0.0

            stats[name] = comp_stats

        return stats

    def write_markdown_report(self, filepath: str) -> None:
        """将统计结果写入 grad_norm_analysis.md。

        Args:
            filepath: Markdown 文件输出路径
        """
        stats = self.compute_statistics()
        if not stats:
            print("[GradNormTracker] No data to write.")
            return

        total_iterations = len(self.history.get('total', []))
        total_params = self.param_counts.get('total', 0)

        lines = []
        lines.append('# Gradient Norm Analysis\n')
        lines.append('## Overview\n')
        lines.append('| Item | Value |')
        lines.append('|------|-------|')
        lines.append(f'| Model | {self.model_name} |')
        lines.append(f'| Total Iterations Analyzed | {total_iterations} |')
        lines.append(f'| Total Parameters | {total_params:,} |')
        if self.config_path:
            lines.append(f'| Config | {self.config_path} |')
        lines.append(f'| Generated | {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} |')
        lines.append('')

        # ---- 去除 total 行，单独处理 ----
        total_stats = stats.pop('total', None)
        component_names = sorted(stats.keys())

        has_param_counts = bool(self.param_counts)

        # ---- 主表格 ----
        lines.append('## Component Gradient Norm Statistics\n')
        if has_param_counts:
            header = ('| Component | Count | Mean | Std | Min | Max | '
                      'P50 | P90 | P95 | P99 | Grad% | Params | Param% | G/P Ratio |')
            sep = ('|-----------|-------|------|-----|-----|-----|'
                   '-----|-----|-----|-----|-------|--------|--------|-----------|')
        else:
            header = ('| Component | Count | Mean | Std | Min | Max | '
                      'P50 | P90 | P95 | P99 | Grad% |')
            sep = ('|-----------|-------|------|-----|-----|-----|'
                   '-----|-----|-----|-----|-------|')
        lines.append(header)
        lines.append(sep)

        def _fmt(v):
            """格式化小数为 4 位有效数字。"""
            if abs(v) < 1e-8:
                return '0.0000'
            return f'{v:.4f}'

        for name in component_names:
            s = stats[name]
            row = (f'| {name} | {s["count"]} | {_fmt(s["mean"])} | {_fmt(s["std"])} | '
                   f'{_fmt(s["min"])} | {_fmt(s["max"])} | '
                   f'{_fmt(s["p50"])} | {_fmt(s["p90"])} | {_fmt(s["p95"])} | '
                   f'{_fmt(s["p99"])} | {s["ratio"]*100:.1f}% |')

            if has_param_counts:
                pc = self.param_counts.get(name, 0)
                param_ratio = pc / total_params if total_params > 0 else 0.0
                gp_ratio = s['ratio'] / param_ratio if param_ratio > 0 else float('inf')
                row += (f' {pc:,} | {param_ratio*100:.1f}% | '
                        f'{gp_ratio:.2f} |')
            lines.append(row)

        # total 行（加粗）
        if total_stats:
            s = total_stats
            row = (f'| **total** | **{s["count"]}** | **{_fmt(s["mean"])}** | '
                   f'**{_fmt(s["std"])}** | **{_fmt(s["min"])}** | '
                   f'**{_fmt(s["max"])}** | **{_fmt(s["p50"])}** | '
                   f'**{_fmt(s["p90"])}** | **{_fmt(s["p95"])}** | '
                   f'**{_fmt(s["p99"])}** | **100%** |')
            if has_param_counts:
                row += f' **{total_params:,}** | **100%** | **1.00** |'
            lines.append(row)
        lines.append('')

        # ---- 解释 G/P Ratio ----
        if has_param_counts:
            lines.append('> **G/P Ratio** = Grad% / Param%。'
                         ' 值为 1.0 表示梯度分配与参数规模成正比例。'
                         ' < 0.3 表示梯度信号偏弱（欠优化），'
                         ' > 3.0 表示梯度信号偏强（过优化）。\n')

        # ---- 梯度占比可视化 ----
        lines.append('## Gradient Norm Ratio Visualization\n')
        lines.append('```')
        max_bar_width = 50
        max_name_len = max(len(n) for n in component_names) if component_names else 20

        for name in component_names:
            ratio = stats[name]['ratio']
            bar_len = int(ratio * max_bar_width)
            bar = '=' * max(bar_len, 1) if ratio > 0 else ''
            pct = f'{ratio * 100:.1f}%'
            lines.append(
                f'{name:<{max_name_len}} [{bar:<{max_bar_width}}] {pct}'
            )
        lines.append('```\n')

        # ---- 梯度/参数平衡可视化 ----
        if has_param_counts:
            lines.append('## Gradient/Parameter Balance Visualization\n')
            lines.append('```')
            for name in component_names:
                param_ratio = self.param_counts.get(name, 0) / total_params if total_params > 0 else 0.0
                gp_ratio = stats[name]['ratio'] / param_ratio if param_ratio > 0 else 0.0
                # 用不同符号表示平衡状态
                if gp_ratio < 0.3:
                    marker = '◀◀ UNDER'
                elif gp_ratio < 0.7:
                    marker = '◀ LOW'
                elif gp_ratio <= 1.5:
                    marker = '● OK'
                elif gp_ratio <= 3.0:
                    marker = 'HIGH ▶'
                else:
                    marker = 'OVER ▶▶'
                pct = f'{gp_ratio:.2f}'
                lines.append(
                    f'{name:<{max_name_len}} G/P={pct}  {marker}'
                )
            lines.append('```\n')

        # ---- 观察与建议 ----
        lines.append('## Observations & Recommendations\n')

        if total_stats:
            observations = self._generate_observations(stats, total_stats)
            lines.extend(observations)

        # ---- 写入文件 ----
        content = '\n'.join(lines)
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f'[GradNormTracker] Report written to: {filepath}')

    def _generate_observations(self, stats: Dict, total_stats: Dict) -> List[str]:
        """根据统计数据自动生成观察与建议。"""
        observations = []

        if not stats:
            return observations

        has_param_counts = bool(self.param_counts)
        total_params = self.param_counts.get('total', 1)

        # ---- 潜在问题 ----
        observations.append('### 潜在问题\n')

        # 梯度消失：ratio < 2%
        vanishing = [(n, s) for n, s in stats.items() if s['ratio'] < 0.02]
        if vanishing:
            observations.append('1. **梯度消失组件**: ' +
                ', '.join(f'`{n}` ({s["ratio"]*100:.1f}%)' for n, s in vanishing) +
                ' — 梯度占比极低，可能学习不充分')

        # 梯度主导：ratio > 35%
        dominating = [(n, s) for n, s in stats.items() if s['ratio'] > 0.35]
        if dominating:
            observations.append('2. **梯度主导组件**: ' +
                ', '.join(f'`{n}` ({s["ratio"]*100:.1f}%)' for n, s in dominating) +
                ' — 梯度占比过高，可能挤压其他组件的学习信号')

        # 梯度波动大：std/mean > 0.5
        volatile = [(n, s) for n, s in stats.items()
                    if s['mean'] > 0 and s['std'] / s['mean'] > 0.5]
        if volatile:
            observations.append('3. **梯度波动较大**: ' +
                ', '.join(f'`{n}` (std/mean={s["std"]/s["mean"]:.2f})'
                          for n, s in volatile) +
                ' — 训练可能不稳定')

        # 梯度为 0 的组件
        zero_grad = [(n, s) for n, s in stats.items() if s['mean'] < 1e-8]
        if zero_grad:
            observations.append('4. **零梯度组件**: ' +
                ', '.join(f'`{n}`' for n, s in zero_grad) +
                ' — 可能被 freeze 或配置错误')

        # ---- 梯度/参数不平衡诊断 ----
        if has_param_counts:
            under_optimized = []
            over_optimized = []
            for name, s in stats.items():
                pc = self.param_counts.get(name, 0)
                param_ratio = pc / total_params if total_params > 0 else 0.0
                if param_ratio > 0 and s['ratio'] > 0:
                    gp_ratio = s['ratio'] / param_ratio
                    if gp_ratio < 0.3:
                        under_optimized.append((name, gp_ratio, param_ratio, s['ratio']))
                    elif gp_ratio > 3.0:
                        over_optimized.append((name, gp_ratio, param_ratio, s['ratio']))

            if under_optimized:
                items = ', '.join(
                    f'`{n}` (G/P={gp:.2f}, Grad%={gr*100:.1f}%, Param%={pr*100:.1f}%)'
                    for n, gp, pr, gr in under_optimized)
                observations.append(
                    f'5. **欠优化组件**（梯度信号相对参数量偏弱）: {items}'
                    ' — 这些组件参数多但梯度小，可能学习不充分')

            if over_optimized:
                items = ', '.join(
                    f'`{n}` (G/P={gp:.2f}, Grad%={gr*100:.1f}%, Param%={pr*100:.1f}%)'
                    for n, gp, pr, gr in over_optimized)
                observations.append(
                    f'6. **过优化组件**（梯度信号相对参数量偏强）: {items}'
                    ' — 这些组件参数少但梯度大，可能主导训练')

        observations.append('')

        # ---- 建议 ----
        observations.append('### 建议\n')

        if vanishing:
            observations.append('- 考虑为梯度占比较低的组件（' +
                ', '.join(f'`{n}`' for n, s in vanishing[:3]) +
                '）使用更大的学习率（通过 `paramwise_cfg` 的 `custom_keys`）')

        if dominating:
            observations.append('- 监控 ' +
                ', '.join(f'`{n}`' for n, s in dominating[:3]) +
                ' 的梯度是否出现爆炸，必要时降低其学习率')

        if volatile:
            observations.append('- ' +
                ', '.join(f'`{n}`' for n, s in volatile[:3]) +
                ' 梯度波动大，检查是否受益于梯度裁剪')

        if has_param_counts and under_optimized:
            observations.append('- **欠优化组件**（' +
                ', '.join(f'`{n}`' for n, gp, pr, gr in under_optimized[:3]) +
                '）梯度信号与参数规模不匹配，建议：'
                ' (1) 增大该组件学习率；'
                ' (2) 检查上游是否有梯度阻断；'
                ' (3) 考虑添加辅助损失直接监督该组件')

        if has_param_counts and over_optimized:
            observations.append('- **过优化组件**（' +
                ', '.join(f'`{n}`' for n, gp, pr, gr in over_optimized[:3]) +
                '）梯度信号过强，建议降低其学习率以避免训练不稳定')

        # 检查 total 梯度范围
        if total_stats and total_stats['max'] > 100:
            observations.append(
                f'- 总梯度范数最大值 {total_stats["max"]:.1f}，'
                f'当前 `grad_max_norm` 截断阈值为 {total_stats.get("grad_max_norm", "?")}，'
                f'可能需要调整')

        if len(observations) == 2:  # 只有标题没有实际建议
            observations.append('- 梯度分布看起来合理，继续监控即可')

        return observations
