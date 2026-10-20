"""Compare five-epoch mini Occ3D logs and name each class's winning model.

The primary per-class comparison uses each model's best-mIoU checkpoint.
An additional oracle comparison selects each class's best epoch separately.
Binary IoU also reports its own best epoch.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLASSES = (
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation')
MODELS = ('opusv1', 'opusv2', 'opusv1_gaussian_sparse_render', 'gaussianformer')
EPOCH = re.compile(r'\[EVAL\] Epoch\s+(\d+)\s+Iter')
PER_CLASS = re.compile(r'\b(' + '|'.join(CLASSES) + r')\s+:\s+([0-9.]+)%')
SCORES = re.compile(r'\bmIoU:\s*([0-9.]+),\s*iou2:\s*([0-9.]+)')


def parse_log(paths):
    epochs = {}
    for path in paths:
        current_epoch = None
        current_classes = {}
        for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
            match = EPOCH.search(line)
            if match:
                current_epoch = int(match.group(1))
                continue
            match = PER_CLASS.search(line)
            if match:
                current_classes[match.group(1)] = float(match.group(2))
                continue
            match = SCORES.search(line)
            if match and current_epoch is not None:
                if len(current_classes) != len(CLASSES):
                    raise ValueError(f'{path}: epoch {current_epoch} has '
                                     f'{len(current_classes)} per-class scores')
                epochs[current_epoch] = {
                    'epoch': current_epoch,
                    'mIoU': float(match.group(1)),
                    'IoU': float(match.group(2)),
                    'per_class': current_classes,
                }
                current_classes = {}
    return [epochs[key] for key in sorted(epochs)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path,
                        default=ROOT / 'work_dirs/mini_occ3d_4096_1024')
    parser.add_argument('--out', type=Path,
                        default=ROOT / 'work_dirs/mini_occ3d_4096_1024/summary.json')
    parser.add_argument('--markdown-out', type=Path,
                        default=ROOT / 'work_dirs/mini_occ3d_4096_1024/comparison.md')
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    summary = {'models': {}, 'per_class_best': {}, 'per_class_oracle_best': {}}
    for name in MODELS:
        logs = sorted((args.run_root / name).glob('*.log'))
        if not logs:
            if args.allow_partial:
                continue
            raise FileNotFoundError(f'No training log for {name}')
        results = parse_log(logs)
        if len(results) != 5 and not args.allow_partial:
            raise ValueError(f'{name} has {len(results)} complete epochs, expected 5')
        if not results:
            if args.allow_partial:
                continue
            raise ValueError(f'{name} has no complete evaluation')
        best_miou = max(results, key=lambda item: (item['mIoU'], -item['epoch']))
        best_iou = max(results, key=lambda item: (item['IoU'], -item['epoch']))
        best_per_class = {
            cls: max(results, key=lambda item: (item['per_class'][cls], -item['epoch']))
            for cls in CLASSES
        }
        summary['models'][name] = {
            'logs': [str(path) for path in logs], 'epochs': results,
            'best_mIoU_epoch': best_miou['epoch'],
            'best_mIoU': best_miou['mIoU'],
            'best_IoU_epoch': best_iou['epoch'],
            'best_IoU': best_iou['IoU'],
            'per_class_at_best_mIoU': best_miou['per_class'],
            'best_per_class': {
                cls: {'IoU': item['per_class'][cls], 'epoch': item['epoch']}
                for cls, item in best_per_class.items()
            },
        }
    if summary['models']:
        for cls in CLASSES:
            scores = {name: data['per_class_at_best_mIoU'][cls]
                      for name, data in summary['models'].items()}
            top = max(scores.values())
            summary['per_class_best'][cls] = {
                'winners': sorted(name for name, value in scores.items() if value == top),
                'IoU': top, 'all_models': scores,
            }
            oracle = {
                name: data['best_per_class'][cls]
                for name, data in summary['models'].items()
            }
            oracle_top = max(value['IoU'] for value in oracle.values())
            summary['per_class_oracle_best'][cls] = {
                'winners': sorted(name for name, value in oracle.items()
                                  if value['IoU'] == oracle_top),
                'IoU': oracle_top, 'all_models': oracle,
            }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n',
                        encoding='utf-8')
    manifest_path = ROOT / 'data/mini_occ3d_4096_1024/manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    lines = [
        '# Occ3D mini：四模型五轮比较', '',
        '## 协议', '',
        '- 训练 4096 帧、验证 1024 帧；从官方 train/val 分别以固定 seed 无放回抽样。',
        f'- 训练样本 SHA256：`{manifest["train"]["sample_pairs_sha256"]}`。',
        f'- 验证样本 SHA256：`{manifest["val"]["sample_pairs_sha256"]}`。',
        '- 六相机单帧、704×256、Occ3D 相机可见 mask；每模型训练 5 epoch。',
        '- micro-batch=1，梯度累积 4；mIoU 为 17 个非空语义类平均，IoU 为二值占用 IoU。',
        '- 固定随机种子 928；四模型均沿用各自配置的 ResNet-50 主干初始化 checkpoint `ckpts/raydn_r50_flash_704_bs2_seq_428q_nui_60e.pth`。',
        '- 每个指标在该模型的五个验证 epoch 内独立选最优；逐类比较取该模型最佳 mIoU epoch。',
        '- OPUS-V2 是按官方 PFN + sparse-conv 离散化结构移植的单帧版本，复用本仓库 OPUS 图像采样接口；不是官方八帧 checkpoint。',
        '- OPUSv1 与 Gaussian 变体共用 600 query、最终每 query 128 child 的候选预算；GaussianFormer 沿用 25,600 Gaussian anchor 的原配置。',
        '- 运行命令：`python3.10 train.py --py-config config/mini_occ3d_4096_1024/<模型名>.py --work-dir work_dirs/mini_occ3d_4096_1024/<模型名> --gradient-accumulation 4 --seed 928`。',
        '- 分割生成命令：`python3.10 tools/make_occ3d_mini_split.py --seed 928 --train-count 4096 --val-count 1024`。',
        '- 本表是单一随机种子、短训练预算的 mini 实验，不表示完整 Occ3D 验证集或多种子统计结论。',
        '', '## 总体指标', '',
        '| 模型 | 最佳 mIoU (%) | epoch | 最佳 IoU (%) | epoch |',
        '| --- | ---: | ---: | ---: | ---: |',
    ]
    for name in MODELS:
        if name not in summary['models']:
            continue
        data = summary['models'][name]
        lines.append(f'| {name} | {data["best_mIoU"]:.2f} | {data["best_mIoU_epoch"]} | '
                     f'{data["best_IoU"]:.2f} | {data["best_IoU_epoch"]} |')
    lines.extend(['', '## 逐类 IoU：各模型最佳 mIoU checkpoint', '',
                  '| 类别 | ' + ' | '.join(MODELS) + ' | 最佳模型 |',
                  '| --- | ' + ' | '.join(['---:'] * len(MODELS)) + ' | --- |'])
    for cls in CLASSES:
        info = summary['per_class_best'].get(cls)
        if info is None:
            continue
        values = [f'{info["all_models"][name]:.2f}' if name in info['all_models'] else '—'
                  for name in MODELS]
        lines.append(f'| {cls} | ' + ' | '.join(values) + ' | '
                     + ', '.join(info['winners']) + ' |')
    lines.extend(['', '## 逐类 IoU：每个类别各自选择最佳 epoch', '',
                  '此表允许同一模型的不同类别来自不同 checkpoint，仅作为五轮内的上界。', '',
                  '| 类别 | 最佳模型 | 该类 IoU (%) | epoch |',
                  '| --- | --- | ---: | ---: |'])
    for cls in CLASSES:
        info = summary['per_class_oracle_best'].get(cls)
        if info is None:
            continue
        winner_epochs = ', '.join(
            f'{name}: {info["all_models"][name]["epoch"]}'
            for name in info['winners'])
        lines.append(f'| {cls} | {", ".join(info["winners"])} | '
                     f'{info["IoU"]:.2f} | {winner_epochs} |')
    lines.extend(['', '数据来源：各模型训练日志的逐 epoch 验证输出；完整数值见 `summary.json`。', ''])
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.write_text('\n'.join(lines), encoding='utf-8')
    print(f'{len(summary["models"])} models summarized: {args.out}')
    for name, data in summary['models'].items():
        print(f'{name}: best mIoU {data["best_mIoU"]:.2f} (epoch {data["best_mIoU_epoch"]}), '
              f'best IoU {data["best_IoU"]:.2f} (epoch {data["best_IoU_epoch"]})')
    for cls, info in summary['per_class_best'].items():
        print(f'{cls}: {", ".join(info["winners"])} {info["IoU"]:.2f}')


if __name__ == '__main__':
    main()
