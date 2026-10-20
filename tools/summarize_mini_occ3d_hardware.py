"""Summarize single-4090 model profiles into the mini comparison report."""
from __future__ import annotations

import json
from pathlib import Path

from profile_mini_occ3d_models import MODELS

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'work_dirs/mini_occ3d_4096_1024/hardware_profile'
DOC = ROOT / 'plans/sparse_render_offical_9_28'
START = '<!-- hardware_profile_start -->'
END = '<!-- hardware_profile_end -->'
LABELS = {
    'opusv1': 'OPUSv1',
    'opusv2': 'OPUSv2',
    'opusv1_gaussian_sparse_render': 'OPUSv1 Gaussian 原版',
    'gaussianformer': 'GaussianFormer',
    'opusv1_gaussian_margin': 'Gaussian + margin',
    'opusv1_gaussian_softplus': 'Gaussian + softplus',
    'opusv1_gaussian_margin_softplus': 'Gaussian + margin + softplus',
    'opusv1_gaussian_softplus_camera_mask': 'Gaussian + 相机 mask',
    'opusv1_gaussian_softplus_learned_bias': 'Gaussian + 相机 mask + 学习偏置',
    'opusv1_gaussian_softplus_class_margin': 'Gaussian + 相机 mask + 类别 margin',
}


def load_profiles():
    records = {}
    for name in MODELS:
        records[name] = {}
        for phase in ('train', 'val', 'infer'):
            path = RAW / f'full_{name}_{phase}.json'
            if not path.is_file():
                raise FileNotFoundError(path)
            record = json.loads(path.read_text())
            if record['model'] != name or record['phase'] != phase:
                raise ValueError(f'Profile identity mismatch: {path}')
            if record['device'] != 'NVIDIA GeForce RTX 4090':
                raise ValueError(f'Unexpected GPU: {path}: {record["device"]}')
            expected = {'train': 4096, 'val': 1024, 'infer': 1024}[phase]
            if record['measured_samples'] != expected:
                raise ValueError(f'Unexpected sample count: {path}')
            records[name][phase] = record
    return records


def load_warm_window():
    records = {}
    for name in MODELS:
        records[name] = {}
        for phase, expected in (('train', 128), ('val', 128), ('infer', 64)):
            path = RAW / f'{name}_{phase}.json'
            record = json.loads(path.read_text())
            if record['model'] != name or record['phase'] != phase or record['measured_samples'] != expected:
                raise ValueError(f'Unexpected warm-window record: {path}')
            records[name][phase] = record
    return records


def gib(mib):
    return mib / 1024


def main():
    records = load_profiles()
    warm = load_warm_window()
    output = RAW / 'summary.json'
    output.write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n')
    lines = [
        '# Occ3D mini：RTX 4090 显存与单样本推理开销', '',
        '测量方案见[设计文档](mini_occ3d_4090_profile_design.md)。'
        '所有模型均使用最佳 mIoU 的 epoch 5 checkpoint、batch 1、六相机 704×256 输入。'
        '训练阶段按原配置执行完整损失、反向与每 4 帧一次的优化器更新。'
        '训练峰值来自完整 4096 帧训练集的一次重放，验证峰值来自完整 1024 帧验证集；'
        '推理延迟与显存也覆盖完整 1024 帧验证集。'
        '环境为 PyTorch 2.12.0+cu126、MMCV 2.2.0、NVIDIA 驱动 595.91.07。', '',
        '## 训练与验证显存峰值', '',
        '单位 GiB。此表包含进程首帧的 cuDNN 算法选择开销。allocated 是张量占用峰值，'
        'reserved 是 PyTorch 缓存分配器向驱动保留的峰值；'
        '这两者均非整卡 `nvidia-smi` 已用显存。', '',
        '| 模型 | 训练 allocated | 训练 reserved | 验证 allocated | 验证 reserved |',
        '| --- | ---: | ---: | ---: | ---: |',
    ]
    for name in MODELS:
        train = records[name]['train']['measurement']
        val = records[name]['val']['measurement']
        lines.append(f'| {LABELS[name]} | '
                     f'{gib(train["peak_allocated_mib"]):.2f} | '
                     f'{gib(train["peak_reserved_mib"]):.2f} | '
                     f'{gib(val["peak_allocated_mib"]):.2f} | '
                     f'{gib(val["peak_reserved_mib"]):.2f} |')
    lines += ['', '### 预热后的短窗口参考', '',
              '先运行 8 帧训练/验证或 16 帧推理，再在后续 128/128/64 帧上统计 allocated 峰值。'
              '这不是完整划分预热后的严格最大值，只用于说明冷启动与通常运行状态的差异。', '',
              '| 模型 | 训练 128 帧 (GiB) | 验证 128 帧 (GiB) | 推理 64 帧 (GiB) |',
              '| --- | ---: | ---: | ---: |']
    for name in MODELS:
        row = warm[name]
        lines.append(f'| {LABELS[name]} | '
                     f'{gib(row["train"]["measurement"]["peak_allocated_mib"]):.2f} | '
                     f'{gib(row["val"]["measurement"]["peak_allocated_mib"]):.2f} | '
                     f'{gib(row["infer"]["measurement"]["peak_allocated_mib"]):.2f} |')
    lines += ['', '## 单样本模型推理', '',
              '单位毫秒；从 GPU 输入已经就绪到 occupancy 输出完成，计入 Python 调用与 CUDA 同步，'
              '沿用验证 DataLoader 的 metas（包括标签和 mask），只运行模型前向；'
              '不计数据读取、主机到 GPU 拷贝、损失和指标统计。'
              'p50 为 1024 帧延迟中位数，p95 为第 95 百分位；'
              '最大值展示少数异常帧的代价，不代表典型延迟。', '',
              '| 模型 | p50 (ms) | p95 (ms) | 最大 (ms) | 单帧吞吐 (帧/s) | 推理 allocated 峰值 (GiB) |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name in MODELS:
        record = records[name]['infer']
        latency = record['latency_ms']
        mem = record['measurement']
        lines.append(f'| {LABELS[name]} | {latency["median"]:.2f} | '
                     f'{latency["p95"]:.2f} | {latency["max"]:.2f} | '
                     f'{1000/latency["median"]:.1f} | '
                     f'{gib(mem["peak_allocated_mib"]):.2f} |')
    lines += ['', '最慢帧与显存峰值的帧序号从验证集 0 开始计数；两者相同可作为定位线索，'
              '但仅凭序号不能确定内部算子的因果归属。', '',
              '| 模型 | 最慢帧序号 | 推理显存峰值帧序号 |',
              '| --- | ---: | ---: |']
    for name in MODELS:
        record = records[name]['infer']
        latency = record['latency_ms']['raw']
        lines.append(f'| {LABELS[name]} | {latency.index(max(latency))} | '
                     f'{record["peak_sample_index"]} |')
    if all(records[name]['infer']['peak_sample_index'] == 0 and
           records[name]['infer']['latency_ms']['raw'].index(
               records[name]['infer']['latency_ms']['max']) == 0
           for name in MODELS):
        lines += ['', '十组模型的最慢帧与推理显存峰值均在首帧。'
                  '结合所有配置启用 `cudnn.benchmark=True`，这符合首帧卷积算法搜索的冷启动特征；'
                  '这是根据观测与配置作出的归因，而不是逐算子内存追踪。']
    control_names = (
        ('opusv1', 'control_opusv1_infer_no_cudnn_benchmark.json'),
        ('opusv1_gaussian_softplus_learned_bias',
         'control_learned_bias_infer_no_cudnn_benchmark.json'),
    )
    lines += ['', '### 冷启动归因对照', '',
              '对同一验证集首帧仅关闭 `cudnn.benchmark`，其他模型、checkpoint 与输入保持一致。'
              '这是首帧控制，不用于正式模型排序。', '',
              '| 模型 | benchmark=True 首帧显存 (GiB) | False 首帧显存 (GiB) | '
              'True 首帧延迟 (ms) | False 首帧延迟 (ms) |',
              '| --- | ---: | ---: | ---: | ---: |']
    for name, filename in control_names:
        control = json.loads((RAW / filename).read_text())
        full = records[name]['infer']
        lines.append(f'| {LABELS[name]} | '
                     f'{gib(full["measurement"]["peak_allocated_mib"]):.2f} | '
                     f'{gib(control["measurement"]["peak_allocated_mib"]):.2f} | '
                     f'{full["latency_ms"]["raw"][0]:.2f} | '
                     f'{control["latency_ms"]["raw"][0]:.2f} |')
    lines += ['', '这项控制支持首帧高峰主要由 cuDNN 算法选择引起；'
              '正式训练、验证和推理表仍遵循原配置 `cudnn.benchmark=True`。']
    best = records['opusv1_gaussian_softplus_learned_bias']
    gf = records['gaussianformer']
    best_train = best['train']['measurement']['peak_allocated_mib']
    gf_train = gf['train']['measurement']['peak_allocated_mib']
    best_latency = best['infer']['latency_ms']['median']
    gf_latency = gf['infer']['latency_ms']['median']
    lines += ['', '## 与 GaussianFormer 的资源权衡', '',
              f'最高精度的可学习偏置组训练 allocated 峰值是 GaussianFormer 的 '
              f'{best_train/gf_train:.2f} 倍；单帧推理 p50 为 {best_latency:.2f} ms，'
              f'相对 GaussianFormer 的 {gf_latency:.2f} ms 缩短 '
              f'{(1-best_latency/gf_latency)*100:.1f}%。'
              '该结论仅针对本机 batch 1 模型前向与这次完整划分显存重放。']
    lines += ['', '## 解释与边界', '',
              '训练数字是从 epoch 5 checkpoint 与优化器状态出发，重放全部 4096 个训练样本一次的峰值；'
              '它不是历史五轮训练进程的回溯实测最大值。验证数字覆盖全部 1024 个验证样本。'
              '模型阶段在独立进程中运行；推理不保留训练优化器状态。'
              '推理表中的帧率是 p50 延迟的倒数，并非包含读取的实测系统吞吐。'
              '该表用于当前 RTX 4090、软件环境及相同输入分辨率下的相对开销比较。', '',
              f'原始逐模型记录：[{output.relative_to(ROOT)}](../../{output.relative_to(ROOT)})。', '']
    report = DOC / 'mini_occ3d_4090_profile_results.md'
    report.write_text('\n'.join(lines))

    compact = [START, '## RTX 4090 单卡开销', '',
               '固定 batch 1；训练/验证分别为完整 4096/1024 帧重放的 PyTorch allocated 峰值，'
               '推理为完整 1024 帧的单样本前向延迟中位数。完整口径、reserved 峰值、p95 和最慢帧见'
               '[硬件开销报告](mini_occ3d_4090_profile_results.md)。', '',
               '| 模型 | 训练峰值 (GiB) | 验证峰值 (GiB) | 推理 p50 (ms) |',
               '| --- | ---: | ---: | ---: |']
    for name in MODELS:
        row = records[name]
        compact.append(f'| {LABELS[name]} | '
                       f'{gib(row["train"]["measurement"]["peak_allocated_mib"]):.2f} | '
                       f'{gib(row["val"]["measurement"]["peak_allocated_mib"]):.2f} | '
                       f'{row["infer"]["latency_ms"]["median"]:.2f} |')
    compact += [END]
    comparison = DOC / 'mini_occ3d_4096_1024_results.md'
    existing = comparison.read_text()
    section = '\n'.join(compact)
    if START in existing and END in existing:
        before = existing[:existing.index(START)].rstrip()
        after = existing[existing.index(END) + len(END):].strip()
        updated = before + '\n\n' + section + ('\n\n' + after if after else '\n')
    else:
        updated = existing.rstrip() + '\n\n' + section + '\n'
    comparison.write_text(updated)
    print(f'Wrote {report} and updated {comparison}')


if __name__ == '__main__':
    main()
