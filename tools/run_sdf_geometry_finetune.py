"""Full-data one-epoch SDF geometry replacement, evaluation, and reporting."""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import pickle
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from mmengine import Config
from tools.prepare_sdf_geometry_cache import RUN, manifest, sha, write
from tools.sdf_geometry.field import load, volume
from loss.gaussian_sdf_geometry_loss import sample_geometry

CONFIG = ROOT/'config/sdf_geometry_replacement_full/opusv1_gaussian_softplus_tiny_8f_sdf_1ep.py'
INITIAL = ROOT/'ckpts/opus_gaussian_8f_tiny.pth'
REPORT = ROOT/'analysis/sdf_geometry_replacement_full_results.md'
SDF_NAMES = ('gaussians', 'center_valid', 'shape_valid', 'sum_signed_sdf',
    'sum_positive_sdf', 'outside_count', 'inside_count', 'sum_normal_scale',
    'shape_violation_count', 'sum_center_loss', 'sum_shape_loss',
    'center_camera_foreground', 'center_camera_empty', 'center_unknown',
    'center_foreground_same_class')


def stamp(): return datetime.now(timezone.utc).isoformat()


def progress_report():
    """Keep the human-readable report honest while the queue is running."""
    if (RUN/'summary.json').exists(): return
    status = json.loads((RUN/'status.json').read_text()) if (RUN/'status.json').exists() else {}
    training = json.loads((RUN/'train/progress.json').read_text()) if (RUN/'train/progress.json').exists() else {}
    cache = json.loads((RUN/'cache_progress_all.json').read_text()) if (RUN/'cache_progress_all.json').exists() else {}
    before = (RUN/'eval/before/complete.json').exists()
    after = (RUN/'eval/after/complete.json').exists()
    text = '# SDF 替换 Set 几何监督：完整数据微调结果\n\n'
    text += '**实验尚未全部完成；下面仅报告真实执行进度，性能与几何改善结论等待完整验证。**\n\n'
    text += f"更新时间：{stamp()}。当前阶段：`{status.get('phase', '准备')}`，状态：`{status.get('status', 'running')}`。\n\n"
    text += '| 项目 | 当前状态 |\n| --- | --- |\n'
    text += f"| GT 距离场缓存 | {cache.get('completed', 0):,} / {cache.get('total', 34149):,} 帧 |\n"
    text += f"| 原始 tiny 权重完整验证 | {'已完成，150 场景／6,019 帧' if before else '待完成'} |\n"
    text += f"| 1 epoch 微调 | {training.get('microbatches', 0):,} / 28,130 帧，{training.get('optimizer_updates', 0):,} 次更新 |\n"
    text += f"| 微调后完整验证 | {'已完成，150 场景／6,019 帧' if after else '待完成'} |\n"
    if training:
        text += f"\n平均训练耗时：{training.get('seconds_per_microbatch', 0):.3f} 秒／帧；预计剩余训练时间：{training.get('remaining_hours', 0):.2f} 小时。估计随实际吞吐更新。\n"
    if status.get('error'): text += f"\n执行错误：`{status['error']}`。检查对应日志后恢复，未把失败步骤记为完成。\n"
    text += '\n完整范围为训练 700 场景／28,130 帧、验证 150 场景／6,019 帧。Set 分类和 Occupancy CE／Lovasz 保留；移除所有阶段的双向 Set 几何损失，最后一层使用中心与每步固定法线的形状损失。\n'
    text += '\n[实验设计](sdf_geometry_replacement_full_design.md) · [运行状态](../work_dirs/sdf_geometry_replacement_full/status.json) · [配置](../config/sdf_geometry_replacement_full/opusv1_gaussian_softplus_tiny_8f_sdf_1ep.py)\n'
    REPORT.parent.mkdir(parents=True, exist_ok=True); REPORT.write_text(text)


class CachedFrames(Dataset):
    def __init__(self, base, rows, deterministic_augmentation=False):
        self.dataset = base; self.rows = rows
        self.deterministic_augmentation = deterministic_augmentation
        assert len(base) == len(rows)
        assert all(tuple(base.keyframes[i]) == (r['scene'], r['frame']) for i, r in enumerate(rows))

    def __getattr__(self, name):
        if name == 'dataset': raise AttributeError(name)
        return getattr(self.dataset, name)

    def __len__(self): return len(self.dataset)

    def __getitem__(self, index):
        if self.deterministic_augmentation:
            np.random.seed(928000000+index); random.seed(928000000+index)
        cache = ROOT/self.rows[index]['cache']
        started = time.monotonic()
        while not cache.exists():
            if time.monotonic()-started > 600: raise TimeoutError(f'SDF cache not ready: {cache}')
            time.sleep(.2)
        row = self.dataset[index]
        sdf, valid, normals = load(cache, return_normals=True)
        # This recipe has image-only augmentation; GT and GS remain in ego axes.
        with np.load(ROOT/self.rows[index]['gt']) as source:
            if not np.array_equal(row['occ_label'], source['semantics']):
                raise ValueError('GT/cache coordinate or index mismatch')
        row['sdf_fields'] = volume(sdf, valid, normals=normals)
        return row


def configure():
    import tools.gaussian_geometry_causal as core
    core.CONFIG = CONFIG
    return core


def training(pilot_steps=0):
    core = configure(); core.seed_all(928); torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    c = Config.fromfile(CONFIG)
    m = json.loads((RUN/'manifest.json').read_text())
    folder = RUN/('pilot' if pilot_steps else 'train'); folder.mkdir(exist_ok=True)
    if not pilot_steps and (folder/'complete.json').exists(): return
    resume = folder/'latest.pth'
    source = resume if resume.exists() and not pilot_steps else INITIAL
    c, net, opt, objective, state = core.build(source, freeze_empty=False)
    completed = int(state.get('finetune_microbatches', 0)) if source == resume else 0
    updates = int(state.get('finetune_optimizer_updates', 0)) if source == resume else 0
    from dataset import OPENOCC_DATASET
    from dataset.utils import custom_collate_fn_temporal
    dataset = CachedFrames(OPENOCC_DATASET.build(c.train_dataset_config), m['splits']['train']['rows'], True)
    order = np.random.default_rng(928).permutation(len(dataset)).tolist()
    total = pilot_steps or len(dataset)
    order = order[:total]
    class FixedSampler(Sampler):
        def __iter__(self): return iter(order[completed:])
        def __len__(self): return total-completed
    data = DataLoader(dataset, batch_size=1, sampler=FixedSampler(), num_workers=4,
        collate_fn=custom_collate_fn_temporal, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    if source == resume:
        torch.set_rng_state(state['rng_torch']); torch.cuda.set_rng_state_all(state['rng_cuda'])
        np.random.set_state(state['rng_numpy']); random.setstate(state['rng_random'])
    else: core.seed_all(928200000)
    net.train(); opt.zero_grad()
    accumulation = c.gradient_accumulation
    total_updates = math.ceil(total/accumulation)
    group_rates = [g.get('initial_lr', g['lr']) for g in opt.param_groups]
    start = time.perf_counter(); beginning = completed
    direct_sdf_gradients = []
    def checkpoint(path, count):
        value = dict(state_dict=net.state_dict(), optimizer=opt.state_dict(), epoch=31,
            finetune_microbatches=count, finetune_optimizer_updates=updates,
            initial=str(INITIAL), initial_sha256=sha(INITIAL), recipe=str(CONFIG),
            rng_torch=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state_all(),
            rng_numpy=np.random.get_state(), rng_random=random.getstate(),
            data_order_sha256=hashlib.sha256(np.asarray(order, dtype=np.int64).tobytes()).hexdigest())
        tmp = path.with_suffix('.tmp'); torch.save(value, tmp); tmp.replace(path)
    with (folder/'trace.jsonl').open('a', buffering=1) as trace:
        for local, batch in enumerate(data, completed):
            at_group_start = local % accumulation == 0
            if at_group_start:
                warmup = min(1., (updates+1)/min(c.finetune_warmup_updates, total_updates))
                cosine = .5*(1+math.cos(math.pi*updates/max(total_updates-1, 1)))
                factor = c.finetune_base_lr/2e-4 * (.1+.9*warmup) * (c.finetune_min_ratio+(1-c.finetune_min_ratio)*cosine)
                for group, base in zip(opt.param_groups, group_rates): group['lr'] = base*factor
            batch = core.to_cuda(batch)
            out = core.forward(net, batch)
            value, details = objective(core.loss_input(c, out, batch))
            if not torch.isfinite(value): raise FloatingPointError(f'Nonfinite loss at {local}')
            if pilot_steps:
                auxiliary, _ = objective.losses[-1]({'gaussian': out['gaussian'], 'metas': batch})
                attributes = (out['gaussian'].means, out['gaussian'].scales, out['gaussian'].rotations)
                gradients = torch.autograd.grad(auxiliary, attributes, retain_graph=True, allow_unused=True)
                direct_sdf_gradients.append([float(g.norm()) if g is not None else 0. for g in gradients])
            group_length = min(accumulation, total-(local//accumulation)*accumulation)
            (value/group_length).backward()
            row = dict(microbatch=local+1, index=order[local], total=float(value.detach()),
                       losses={k:float(v) for k,v in details.items()})
            if (local+1) % accumulation == 0 or local+1 == total:
                norm = torch.nn.utils.clip_grad_norm_(net.parameters(), c.grad_max_norm, error_if_nonfinite=True)
                opt.step(); opt.zero_grad(); updates += 1
                row.update(optimizer_update=updates, grad_norm=float(norm), base_lr=c.finetune_base_lr)
            trace.write(json.dumps(row)+'\n')
            count = local+1
            if count % 32 == 0 or count in (beginning+1, total):
                elapsed = time.perf_counter()-start; rate = elapsed/(count-beginning)
                g = out['gaussian']
                status = dict(status='running', microbatches=count, total=total,
                    optimizer_updates=updates, seconds=elapsed, seconds_per_microbatch=rate,
                    remaining_hours=(total-count)*rate/3600, loss=row['total'],
                    sdf_center=row['losses']['GaussianSDFGeometryLoss/center'],
                    sdf_shape=row['losses']['GaussianSDFGeometryLoss/shape'],
                    valid_centers=row['losses']['GaussianSDFGeometryLoss/valid_center'],
                    valid_shapes=row['losses']['GaussianSDFGeometryLoss/valid_shape'],
                    mean_scale=float(g.scales.detach().mean()),
                    maxscale_saturated=float((g.scales.detach().max(-1).values >= .3999).float().mean()),
                    peak_cuda_MiB=torch.cuda.max_memory_allocated()/2**20)
                write(folder/'progress.json', status); print(json.dumps(status), flush=True)
                progress_report()
            if (count % (500*accumulation) == 0 or count == total) and not pilot_steps:
                checkpoint(folder/'latest.pth', count)
            del value, out, batch
    if pilot_steps:
        audit = np.asarray(direct_sdf_gradients)
        if not np.isfinite(audit).all() or not (audit.sum(0) > 0).all():
            raise RuntimeError(f'SDF pilot has no finite/nonzero route to all geometric attributes: {audit.tolist()}')
        write(folder/'direct_gradient_audit.json', dict(status='passed', names=['means','scales','rotations'],
              per_frame_norms=direct_sdf_gradients))
    if not pilot_steps:
        # The final file is a hard link to the atomic latest checkpoint.
        final = folder/'final.pth'
        if final.exists(): final.unlink()
        os.link(folder/'latest.pth', final)
    write(folder/'complete.json', dict(status='complete', frames=total, optimizer_updates=updates,
        epochs=1 if not pilot_steps else 0, unique_frames=len(set(order)),
        seconds=time.perf_counter()-start, initial_sha256=sha(INITIAL),
        checkpoint_sha256=sha(folder/'final.pth') if not pilot_steps else None,
        data_order_sha256=hashlib.sha256(np.asarray(order, dtype=np.int64).tobytes()).hexdigest()))
    net.flush_memory_phase()


def evaluate(name):
    configure()
    import tools.eval_gaussian_geometry_causal as evaluation
    from tools.boundary_gaussian_stats.data import EvaluationFrames
    evaluation.RUN = RUN
    original_accumulator = evaluation.extra_accumulator
    original_add = evaluation.add_extra
    m = json.loads((RUN/'manifest.json').read_text())
    def accumulator():
        a = original_accumulator()
        a['sdf_geometry'] = torch.zeros(len(SDF_NAMES), dtype=torch.float64, device='cuda')
        return a
    def add(a, net, out, batch, values):
        original_add(a, net, out, batch, values)
        g = out['gaussian']
        value, sigma, cv, sv = sample_geometry(g, batch['sdf_fields'])
        value, sigma, cv, sv = value[0], sigma[0], cv[0], sv[0]
        positive = value[cv].clamp_min(0)
        violation = value[sv]+1.2815515655446004*sigma[sv]
        means = g.means[0]
        cells = ((means-means.new_tensor([-40., -40., -1.]))/.4).long()
        cells = torch.minimum(torch.maximum(cells, torch.zeros_like(cells)), cells.new_tensor([199, 199, 15]))
        ids = (cells[:, 0]*200+cells[:, 1])*16+cells[:, 2]
        y = batch['occ_label'].flatten()[ids]; camera = batch['occ_cam_mask'].flatten()[ids].bool()
        foreground = camera & (y >= 0) & (y < 17)
        rows = [value.new_tensor(value.numel()), cv.sum(), sv.sum(), value[cv].sum(),
                positive.sum(), (value[cv] > 0).sum(), (value[cv] <= 0).sum(),
                sigma[sv].sum(), (violation > 0).sum(), positive.square().sum(),
                violation.clamp_min(0).square().sum(), foreground.sum(),
                (camera & (y == 17)).sum(), (~camera).sum(),
                (foreground & (g.semantics[0].argmax(-1) == y)).sum()]
        a['sdf_geometry'] += torch.stack(rows).double()
    evaluation.extra_accumulator = accumulator; evaluation.add_extra = add
    evaluation.EvaluationFrames = lambda c, split: CachedFrames(EvaluationFrames(c, split), m['splits'][split]['rows'])
    checkpoint = INITIAL if name == 'before' else RUN/'train/final.pth'
    evaluation.evaluate(name, checkpoint, workers=4, splits=('val',))


def wait_for(predicate, label):
    start = time.monotonic()
    while not predicate():
        pid = json.loads((RUN/'cache_pid.json').read_text())['pid']
        try: os.kill(pid, 0)
        except ProcessLookupError: raise RuntimeError(f'Cache process stopped during {label}')
        progress_report()
        time.sleep(15)
    print(f'{label}: ready after {time.monotonic()-start:.1f}s', flush=True)


def audit_data():
    from concurrent.futures import ThreadPoolExecutor
    m = json.loads((RUN/'manifest.json').read_text())
    if (RUN/'availability.json').exists():
        previous = json.loads((RUN/'availability.json').read_text())
        if all(split in previous and previous[split]['missing'] == 0 and
               previous[split]['frames'] == m['splits'][split]['frames'] for split in ('train','val')):
            print('Reusing completed full-file availability audit from this experiment.',flush=True)
            return
    cameras = ('CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT')
    result = {}
    for split in ('train', 'val'):
        annotation = ROOT/f'data/nuscenes_cam/nuscenes_infos_{split}_sweeps_occ.pkl'
        with annotation.open('rb') as stream: data = pickle.load(stream)
        paths = {ROOT/r['gt'] for r in m['splits'][split]['rows']}
        image_paths = set(); lidar_paths = set()
        for row in m['splits'][split]['rows']:
            frames = data['infos'][row['scene']]; current = row['frame']
            previous = [i for i in range(current-1, -1, -1) if all(cam in frames[i].get('data', {}) for cam in cameras)]
            selected = {current}
            if split == 'train':
                for stride in range(4, 9):
                    selected.update(previous[k*stride-1] for k in range(1, 8) if k*stride <= len(previous))
            else:
                selected.update([i for i in previous if frames[i].get('token') and 'LIDAR_TOP' in frames[i].get('data', {})][:7])
            for index in selected:
                image_paths.update(ROOT/'data/nuscenes'/frames[index]['data'][cam]['filename'] for cam in cameras)
            lidar_paths.add(ROOT/'data/nuscenes'/frames[current]['data']['LIDAR_TOP']['filename'])
        paths.update(image_paths); paths.update(lidar_paths)
        with ThreadPoolExecutor(max_workers=16) as pool:
            missing = [str(p) for p, exists in zip(paths, pool.map(Path.is_file, paths)) if not exists]
        result[split] = dict(frames=len(m['splits'][split]['rows']), scenes=m['splits'][split]['scenes'],
            required_files=len(paths), images=len(image_paths), lidar=len(lidar_paths), missing=len(missing), examples=missing[:20])
        print(split, result[split], flush=True)
        write(RUN/'availability.json', result)
        if missing: raise FileNotFoundError(f'{split}: {len(missing)} required files missing: {missing[:3]}')


def queue():
    RUN.mkdir(exist_ok=True); m = manifest(); c = Config.fromfile(CONFIG)
    files = [CONFIG, Path(__file__), ROOT/'loss/gaussian_sdf_geometry_loss.py',
             ROOT/'tools/sdf_geometry/field.py', ROOT/'tools/prepare_sdf_geometry_cache.py',
             ROOT/'tools/report_sdf_geometry_finetune.py', ROOT/'tools/gaussian_geometry_causal.py',
             ROOT/'tools/eval_gaussian_geometry_causal.py', ROOT/'tools/stat_gaussian_voxel_coverage.py',
             ROOT/'tools/boundary_gaussian_stats/data.py', ROOT/'loss/opus_set_loss.py',
             ROOT/'model/head/gaussian_opus_head.py', ROOT/'model/head/opus_v1_gaussian_trainable_bias_head.py']
    protocol = dict(status='registered', initial_sha256=sha(INITIAL), seed=928,
        train_frames=28130, train_scenes=700, val_frames=6019, val_scenes=150,
        optimizer='restore AdamW moments; reset group-ratio warmup/cosine for one epoch',
        accumulation=c.gradient_accumulation, base_lr=c.finetune_base_lr,
        center_weight=.5, shape_weight=.5, delta_m=0., rho=.1,
        classification='all stages, original nearest-target focal loss',
        geometry='remove BOTH original Set geometry directions at ALL stages; SDF center/shape on final stage only',
        source_sha256={str(f.relative_to(ROOT)):sha(f) for f in files})
    if (RUN/'protocol.json').exists():
        assert json.loads((RUN/'protocol.json').read_text()) == protocol, 'Protocol/source changed'
    else:
        write(RUN/'protocol.json', protocol); c.dump(RUN/'resolved_config.py')
        from shutil import copy2
        for f in files:
            target = RUN/'sources'/f.relative_to(ROOT); target.parent.mkdir(parents=True, exist_ok=True); copy2(f, target)
    def phase(name, action=None, *args):
        write(RUN/'status.json', dict(status='running', phase=name, updated=stamp(), pid=os.getpid()))
        progress_report()
        if action:
            with (RUN/(action+('_'+args[0] if args else '')+'_console.log')).open('ab', buffering=0) as log:
                command = [sys.executable, '-u', str(Path(__file__)), action, *args]
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    try:
        phase('核对完整训练／验证及可能使用的历史图像', 'audit')
        phase('等待完整验证集 SDF 缓存')
        wait_for(lambda: (RUN/'cache_complete_val.json').exists() or (RUN/'cache_complete_all.json').exists(), 'validation cache')
        phase('原始 tiny 权重：完整验证与几何统计', 'eval', 'before')
        phase('等待训练缓存首批样本')
        order = np.random.default_rng(928).permutation(28130)[:128]
        wait_for(lambda: all((ROOT/m['splits']['train']['rows'][i]['cache']).exists() for i in order), 'training cache')
        phase('训练入口与 CUDA 梯度检查', 'pilot')
        phase('完整训练集微调 1 epoch', 'train')
        phase('微调后权重：完整验证与几何统计', 'eval', 'after')
        phase('汇总性能与几何分布', 'report')
        write(RUN/'status.json', dict(status='complete', phase='完整实验与报告已完成', updated=stamp(), pid=os.getpid()))
    except BaseException as error:
        write(RUN/'status.json', dict(status='failed', phase='执行中断', error=repr(error), updated=stamp()))
        progress_report(); raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['queue', 'train', 'pilot', 'eval', 'report', 'audit'])
    p.add_argument('name', nargs='?', choices=['before', 'after'])
    args = p.parse_args()
    if args.action == 'audit': audit_data()
    elif args.action == 'queue': queue()
    elif args.action == 'train': training()
    elif args.action == 'pilot': training(pilot_steps=8)
    elif args.action == 'eval': evaluate(args.name)
    else:
        from tools.report_sdf_geometry_finetune import report
        report()


if __name__ == '__main__': main()
