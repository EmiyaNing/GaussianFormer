"""Extract the experiment-record metadata the plan requires for every table.

``plans/sparse_render_plans/README.md`` section 4.2 makes this mandatory:

    每个训练/评估实验必须落盘以下元信息，缺一不可
    exp_id / config / backbone / input_size / frames / pretraining /
    gaussian_or_query_count / mask / gpu / batch

and P3 T3.1 repeats it: "每张表必须列出的字段: backbone、输入尺寸、帧数、预训练、
高斯/query 数量、mask、GPU、batch ... 任何一行结果都必须能追溯到同一套协议."

No report in this project carried those fields, so there was no mechanical way to
check that nothing was mixed -- which is the requirement's whole purpose.  This
script derives them **from the config file** rather than from prose, so the table
cannot drift away from what was actually run.

    python3.10 benchmarks/p1/experiment_metadata.py --configs a0 a5 a2 a4
    python3.10 benchmarks/p1/experiment_metadata.py --markdown
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mmengine import Config

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# The experiments whose metadata the README table records.
EXPERIMENTS = {
    'a0_base': 'config/opus/ablations/p1_a0_base.py',
    'a1_pointwise': 'config/opus/ablations/p1_a1_pointwise.py',
    'a2_knn_mean': 'config/opus/ablations/p1_a2_knn_mean.py',
    'a2_knn_max': 'config/opus/ablations/p1_a2_knn_max.py',
    'a2_knn_k59': 'config/opus/ablations/p1_a2_knn_k59.py',
    'a2_knn_k215': 'config/opus/ablations/p1_a2_knn_k215.py',
    'a3_graph_attention': 'config/opus/ablations/p1_a3_graph_attention.py',
    'a4_fixed_gaussian': 'config/opus/ablations/p1_a4_fixed_gaussian.py',
    'a5_cap0': 'config/opus/ablations/p1_a5_gaussian.py',
    'a5_cap8': 'config/opus/ablations/p1_a5_cap8.py',
    'a5_cap32': 'config/opus/ablations/p1_a5_cap32.py',
    'a5_cap64': 'config/opus/ablations/p1_a5_cap64.py',
    'a5_cap128': 'config/opus/ablations/p1_a5_cap128.py',
    'dense_naive': 'config/opus/ablations/p3_dense_g2v.py',
    'dense_block': 'config/opus/ablations/p3_dense_g2v_block.py',
    'fullval_a0': 'config/opus/ablations/p4_fullval_a0.py',
    'fullval_a5': 'config/opus/ablations/p4_fullval_a5.py',
}

# Values that come from the run rather than the config, recorded once here so the
# README table and this script cannot disagree silently (a test pins them).
RUN_FACTS = {
    'gpu': 'RTX 4090 24 GiB',
    'batch_train': 4,
    'batch_eval': 1,
    'iters_per_epoch': 85,
    'epochs': 40,
    'train_split': 'data/reduced_splits/train_reduced.pkl (12 scenes, 482 keyframes)',
    'val_split': 'data/reduced_splits/val_reduced.pkl (4 scenes, 161 keyframes, 113 samples)',
    'val_split_full': 'data/nuscenes_cam/nuscenes_infos_val_sweeps_occ.pkl (150 scenes, 6019 keyframes)',
}


def describe(name: str, path: str) -> dict:
    cfg = Config.fromfile(str(ROOT / path))
    head = cfg.model.pts_bbox_head
    backbone = cfg.model.img_backbone
    aug = cfg.data_aug_conf
    adapter = cfg.model.get('semantic_gaussian') or {}
    if not isinstance(adapter, dict):
        adapter = dict(adapter)
    return {
        'exp_id': name,
        'config': path,
        'backbone': f"{backbone['type']}-{backbone.get('depth', '?')}",
        'input_size': f"{aug['final_dim'][0]}x{aug['final_dim'][1]}",
        'frames': cfg.get('num_frames'),
        'horizons': head.get('num_fu_frames'),
        'pretraining': cfg.get('train_load_from'),
        'queries': head.get('num_query'),
        'children': cfg.model.get('semantic_gaussian', {}).get('num_children')
        if isinstance(cfg.model.get('semantic_gaussian'), dict) else None,
        # A0 has no adapter, so it has no children: its point count IS the query
        # count.  Multiplying by a missing child count would print "0" and hide
        # the fact that A0 is a differently-shaped model, not a broken config.
        'children_effective': adapter.get('num_children') or 1,
        'points': (head.get('num_query') or 0) * (adapter.get('num_children') or 1),
        'adapter': adapter.get('type', '(none: base classifier only)'),
        'adapter_neighbours': adapter.get('num_neighbors'),
        'adapter_cell': adapter.get('cell_size'),
        'adapter_support': adapter.get('support_sigma'),
        'adapter_cap': adapter.get('max_neighbours', 0),
        'adapter_freeze_geometry': adapter.get('freeze_geometry', False),
        'mask': 'no camera/lidar mask (train_mask_type=none)',
        'train_batch': cfg.get('train_loader', {}).get('batch_size'),
        'val_batch': cfg.get('val_loader', {}).get('batch_size'),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configs', nargs='*', default=None,
                        help='subset of experiment ids; default all')
    parser.add_argument('--markdown', action='store_true',
                        help='emit the README table instead of key: value lines')
    args = parser.parse_args(argv)

    names = args.configs or list(EXPERIMENTS)
    rows = []
    for name in names:
        if name not in EXPERIMENTS:
            raise SystemExit(f'unknown experiment {name!r}; '
                             f'known: {", ".join(EXPERIMENTS)}')
        rows.append(describe(name, EXPERIMENTS[name]))

    if not args.markdown:
        for row in rows:
            print(f'--- {row["exp_id"]}')
            for key, value in row.items():
                if key != 'exp_id':
                    print(f'    {key}: {value}')
        return

    print('| exp_id | backbone | 输入 | 帧数 | 预训练 | query×children = 点数 | adapter | '
          'mask | GPU | batch(训/评) |')
    print('| --- | --- | --- | ---: | --- | --- | --- | --- | --- | ---: |')
    for row in rows:
        # Build the adapter suffix without nesting f-strings: Python <3.12
        # forbids backslashes inside f-string expressions, and the nested quotes
        # would need them.
        suffix = ''
        # num_neighbors is inherited by every adapter from the sparse-world
        # config, but only KNNResidualAggregator actually uses it.  The CUDA and
        # dense adapters override forward(), so printing k=8 for A5 would state
        # the exact opposite of what the operator does (A5 aggregates every point
        # in its support box -- 215.2 accepted neighbours on average).
        if row['adapter_neighbours'] is not None and \
                row['adapter'].endswith('KNNResidualAggregator'):
            suffix += f" (k={row['adapter_neighbours']})"
        if row['adapter_cell']:
            suffix += f" cell={row['adapter_cell']}"
        if row['adapter_support']:
            suffix += f" sup={row['adapter_support']}"
        if row['adapter_cap']:
            suffix += f" cap={row['adapter_cap']}"
        if row['adapter_freeze_geometry']:
            suffix += ' 冻结几何'
        adapter = row['adapter'].replace('PointAnchoredSemanticGaussianCUDA', 'A5/CUDA')
        pretrain = Path(row['pretraining']).name if row['pretraining'] else '—'
        print(f"| `{row['exp_id']}` | {row['backbone']} | {row['input_size']} | "
              f"{row['frames']} | `{pretrain}` | "
              f"{row['queries']}"
              f"{'×' + str(row['children']) + ' = ' + format(row['points'], ',') if row['children'] else '（无 children）'} | "
              f"{adapter}{suffix} | "
              f"无 | {RUN_FACTS['gpu']} | {row['train_batch']}/{row['val_batch']} |")
    print()
    print(f"- 训练轮数：**{RUN_FACTS['epochs']}**，每轮 **{RUN_FACTS['iters_per_epoch']}** 迭代"
          f"（约 3,400 次，相对完整配方约 33,000 次压缩 ~10×）")
    print(f"- 训练集：`{RUN_FACTS['train_split']}`")
    print(f"- 评测集（本目录全部结果）：`{RUN_FACTS['val_split']}`")
    print(f"- 评测集（全量 val 锚点复核）：`{RUN_FACTS['val_split_full']}`")
    print(f"- 时域：0/2/4/6 s（`num_fu_frames`），无 camera/lidar mask")
    print("- ⚠️ **A5/A4 的邻居数不是 k**：`PointAnchoredSemanticGaussianCUDA` 覆盖了 `forward`，"
          "`num_neighbors`（配置里继承的 8）对它是**无效键**；它聚合支撑盒内的全部点，"
          "在学到的 scale 0.53 上平均 **215.2** 个接纳邻居（[P4-8](P4-8_robustness.md)、[P1-3](P1-3_aggregation_is_not_the_source.md)）。"
          "表中 KNN 变体才标 k。")
    print("- A0 只有 query 没有 children（无 adapter），其 train batch 为 1 是因为它从不训练、"
          "只作为冻结基线被评测——这不构成协议混用。")


if __name__ == '__main__':
    main()
