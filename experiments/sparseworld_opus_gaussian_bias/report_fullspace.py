"""Full-space comparison with verified diagnostics for all fifteen epochs."""
import argparse
from pathlib import Path

import numpy as np

from . import report as original_report
from .queue_fullspace import ROOT, ANALYSIS
from .train import atomic_json


def gather_diagnostics(work, output):
    diagnostics = []
    for epoch in range(1, 16):
        directory = output / 'epoch_diagnostics'
        record = original_report.read(directory / f'epoch_{epoch:02d}_diagnostics.json')
        formal = original_report.read(work / f'epoch_{epoch:02d}_metrics.json')
        if (record['epoch'] != epoch or record['num_samples'] != 1024 or
                record['is_smoke'] or record['voxel_supervision_mask'] != 'none'):
            raise ValueError(f'epoch {epoch} diagnostic is incomplete or not full-space')
        if record['selection_mean_miou'] != formal['selection_mean_miou']:
            raise ValueError('diagnostic and formal scores differ')
        if len(record['attribute_sample_indices']) != 16 or len(record['attributes']) != 16 * 7:
            raise ValueError('fixed diagnostic samples are incomplete')
        with np.load(directory / f'epoch_{epoch:02d}_visibility_confusion.npz') as visibility:
            with np.load(work / f'epoch_{epoch:02d}_confusion.npz') as confusion:
                if (not np.array_equal(visibility['full'], confusion['confusion']) or
                        not np.array_equal(visibility['full'], visibility['camera'] + visibility['noncamera'])):
                    raise ValueError('visibility diagnostic differs from formal evaluation')
        diagnostics.append(record)
    return diagnostics


def create_report(root, output):
    work = root / 'sparseworld_opus_gaussian_bias'
    diagnostics = gather_diagnostics(work, output)
    original_report.create_report(root, output)
    path = output / 'report.md'
    text = path.read_text().replace(
        '# SparseWorld OPUS Gaussian bias：4096 / 1024，15 epoch 对比',
        '# SparseWorld OPUS Gaussian bias全空间监督：4096 / 1024，15 epoch 对比')
    text = text.replace('camera-mask Gaussian 体素监督',
                        '全空间 Gaussian 体素监督（mask=none，CE包含可见/不可见区域和空体素）')
    protocol = ('- Strict复用原实验完整15轮及其最优checkpoint独立复核；新Gaussian在独立目录从头训练15轮。\n'
                '- 相对原camera-mask Gaussian配置，训练语义只改变体素监督mask；预训练时域、未来课程、模型、优化器、seed与子集不变。\n'
                '- 诊断复用每轮1024样本的正式前向；固定16样本记录属性与渲染变体，前2样本比较实际监督输出层梯度。诊断探针不参与checkpoint选择。\n')
    text = text.replace('## 实验协议与解释范围\n\n', '## 实验协议与解释范围\n\n' + protocol)
    lines = ['', '## 每轮诊断趋势', '',
             '以下属性饱和率为固定16样本的平均值；FP来自完整1024样本。', '',
             '| epoch | 平均mIoU (%) | 当前不可见FP | 当前尺度饱和 (%) | 当前opacity饱和 (%) | 3秒尺度饱和 (%) | 3秒opacity饱和 (%) | empty bias | 诊断 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    summary = []
    for d in diagnostics:
        epoch = d['epoch']
        def mean_attribute(h, key):
            return float(np.mean([v[key] for v in d['attributes'] if v['horizon'] == h]))
        row = dict(epoch=epoch, selection_mean_miou=d['selection_mean_miou'],
                   current_noncamera_fp=d['visibility']['h0/noncamera']['occupied_fp'],
                   current_scale_saturation=mean_attribute(0, 'scale_upper_saturation_percent'),
                   current_opacity_saturation=mean_attribute(0, 'opacity_upper_saturation_percent'),
                   three_second_scale_saturation=mean_attribute(6, 'scale_upper_saturation_percent'),
                   three_second_opacity_saturation=mean_attribute(6, 'opacity_upper_saturation_percent'),
                   empty_bias=mean_attribute(0, 'empty_bias'),
                   report=f'epoch_diagnostics/epoch_{epoch:02d}_diagnostics.md')
        summary.append(row)
        lines.append(f"| {epoch} | {row['selection_mean_miou']:.4f} | {row['current_noncamera_fp']} | {row['current_scale_saturation']:.3f} | {row['current_opacity_saturation']:.3f} | {row['three_second_scale_saturation']:.3f} | {row['three_second_opacity_saturation']:.3f} | {row['empty_bias']:.4f} | [诊断]({row['report']}) |")
    atomic_json(output / 'diagnostics_summary.json', dict(epochs=summary, verified_epochs=15))
    path.write_text(text + '\n'.join(lines) + '\n')
    status = original_report.read(output / 'status.json')
    status.update(voxel_supervision_mask='none', diagnostics_verified_epochs=15,
                  strict_baseline_reused=True)
    atomic_json(output / 'status.json', status)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=ANALYSIS)
    args = parser.parse_args()
    create_report(args.root, args.output)


if __name__ == '__main__':
    main()
