"""Prepare, check, and launch one full epoch of Gaussian budget fine-tuning."""
from __future__ import annotations
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUN = ROOT / 'work_dirs/small_query_budget_camera_full_1ep'
CONFIG = ROOT / 'config/full_occ3d_query_budget/opusv1_gaussian_small_8f_query_budget_1ep.py'
REPORT = ROOT / 'analysis/query_importance_camera_set_results.md'


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def status(phase, **extra):
    obj = dict(status=phase, pid=os.getpid(), updated_at=datetime.now().astimezone().isoformat(), **extra)
    write(RUN / 'status.json', obj)
    print(json.dumps(obj, ensure_ascii=False), flush=True)


def progress_report(phase, details=''):
    check_path = RUN / 'preflight.json'
    check = json.loads(check_path.read_text()) if check_path.exists() else None
    lines = ['# Query 重要性分支：完整 Occ3D 微调结果', '',
        f'**当前状态：{phase}。**', '',
        '本轮使用原始 small 预训练权重初始化，完整训练集为 700 场景、28,130 帧。'
        '最后一级生成 115,200 个候选 GS，训练软门控总量为 76,800，推理硬选 76,800。'
        '六个阶段的 Set GT 均在最近邻匹配前按 camera mask 过滤。', '']
    if details:
        lines += [details, '']
    if check:
        lines += ['## 已完成的运行检查', '',
            f"- 在真实完整训练数据管线上执行了 {check['optimizer_steps']} 次独立检查更新；这些更新没有写入正式初始化权重。",
            f"- Occupancy loss 到重要性分支的直接梯度范数为 {check['occupancy_importance_gradient_norm']:.8g}，有限且非零。",
            '- Set loss 对重要性分支没有经由离散选择索引的直接梯度。',
            '- 六阶段 camera GT 数量一致；最终 Set 只监督 76,800 个被选 GS。',
            '- 硬推理保留 76,800 个 GS，所保留 GS 使用原始 opacity。',
            f"- 检查期间峰值 allocated 显存为 {check['peak_allocated_mib']:.1f} MiB；该数不是完整训练的峰值保证。", '']
    lines += ['## 正式训练与验证', '',
        '训练尚未完成时，不填写最终 mIoU、IoU 或方法收益。正式训练结束后会在完整 150 场景、6,019 帧上进行硬选择验证。', '',
        '## 文件', '',
        '- [实验设计](query_importance_camera_set_design.md)',
        '- [运行状态](../work_dirs/small_query_budget_camera_full_1ep/status.json)',
        '- [严格权重初始化校验](../work_dirs/small_query_budget_camera_full_1ep/initialization_audit.json)',
        '- 正式训练日志：`work_dirs/small_query_budget_camera_full_1ep/train_console.log`', '']
    if check:
        lines += ['- [真实数据运行检查](../work_dirs/small_query_budget_camera_full_1ep/preflight.json)', '']
    REPORT.write_text('\n'.join(lines), encoding='utf-8')


def preflight(steps=3):
    import torch
    from mmengine import Config
    from mmengine.runner import set_random_seed
    from mmengine.optim import build_optim_wrapper
    from mmseg.models import build_segmentor
    from dataset import OPENOCC_DATASET
    from dataset.utils import custom_collate_fn_temporal
    from loss import OPENOCC_LOSS
    from model.encoder.gaussian_budget_gate import gather_candidates
    cfg = Config.fromfile(CONFIG)
    set_random_seed(cfg.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    network = build_segmentor(cfg.model)
    checkpoint = torch.load(ROOT / cfg.train_load_from, map_location='cpu', weights_only=False)
    network.load_state_dict(checkpoint['state_dict'], strict=True)
    del checkpoint
    network.cuda().train()
    objective = OPENOCC_LOSS.build(cfg.loss).cuda()
    optimizer = build_optim_wrapper(network, cfg.optimizer)
    training = OPENOCC_DATASET.build(cfg.train_dataset_config)
    validation = OPENOCC_DATASET.build(cfg.val_dataset_config)
    assert len(training) == 28130 and len(validation) == 6019
    assert len({s for s, _ in training.keyframes}) == 700
    assert len({s for s, _ in validation.keyframes}) == 150
    importance = list(network.encoder.importance_branch.parameters())
    initial_weight = network.encoder.importance_branch[-1].weight.detach().clone()
    records = []
    importance_norm = None
    torch.cuda.reset_peak_memory_stats()
    for step in range(steps):
        batch = custom_collate_fn_temporal([training[step]])
        batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
        iteration = 0 if step == 0 else 100 if step == 1 else 14065
        optimizer.zero_grad()
        output = network(imgs=batch['img'], metas={k: v for k, v in batch.items() if k != 'img'}, global_iter=iteration)
        inputs = dict(metas=batch, global_iter=iteration)
        inputs.update({k: output[v] for k, v in cfg.loss_input_convertion.items()})
        value, metrics = objective(inputs)
        assert torch.isfinite(value)
        target_count = int(((batch['occ_label'] >= 0) & (batch['occ_label'] < 17) & batch['occ_cam_mask'].bool()).sum())
        for stage in range(6):
            assert metrics[f'CenterSupportDiagnosticSetLoss/stage_{stage}/gt_count'] == target_count
        assert metrics['CenterSupportDiagnosticSetLoss/stage_5/pred_count'] == 76800
        assert output['gaussian'].means.shape == (1, 115200, 3)
        p = output['allocation']['gates']
        assert float((p.sum(-1) - 76800).abs().max()) < .05
        if step == 0:
            occ = objective.losses[1](inputs)
            if isinstance(occ, tuple):
                occ = occ[0]
            gradients = torch.autograd.grad(occ, importance, retain_graph=True, allow_unused=True)
            importance_norm = float(sum(g.detach().square().sum() for g in gradients if g is not None).sqrt())
            assert importance_norm > 0 and all(torch.isfinite(g).all() for g in gradients if g is not None)
            set_value = objective.losses[0](inputs)[0]
            set_gradients = torch.autograd.grad(set_value, importance, retain_graph=True, allow_unused=True)
            assert all(g is None or not g.any() for g in set_gradients)
        value.backward()
        norm = torch.nn.utils.clip_grad_norm_(network.parameters(), cfg.grad_max_norm, error_if_nonfinite=True)
        optimizer.step()
        records.append(dict(step=step, schedule_iteration=iteration, loss=float(value.detach()),
            gradient_norm=float(norm), camera_foreground=target_count,
            soft_count=float(p.detach().sum()),
            gate_temperature=output['allocation']['temperature'],
            binary_weight=metrics['GaussianBudgetLoss/binary_weight']))
        del output, inputs, value
    assert not torch.equal(initial_weight, network.encoder.importance_branch[-1].weight.detach())
    network.eval()
    with torch.no_grad():
        output = network(imgs=batch['img'], metas={k: v for k, v in batch.items() if k != 'img'})
        assert output['gaussian'].means.shape == (1, 76800, 3)
        candidates = output['representation'][-1]['gaussian']
        expected_alpha = gather_candidates(candidates.opacities, output['allocation']['selected_indices']).flatten(1, 2)
        torch.testing.assert_close(output['gaussian'].opacities, expected_alpha, rtol=0, atol=0)
        assert torch.isfinite(output['pred_occ'][-1]).all()
        scores = output['allocation']['query_scores']
    result = dict(status='passed', optimizer_steps=steps, records=records,
        train_frames=len(training), train_scenes=700, validation_frames=len(validation), validation_scenes=150,
        occupancy_importance_gradient_norm=importance_norm,
        set_has_no_direct_importance_gradient=True, strict_initial_checkpoint_load=True,
        importance_weights_changed=True, inference_gaussians=76800,
        final_query_score_std=float(scores.std(unbiased=False)),
        hard_inference_preserves_physical_opacity=True,
        peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
        final_set_predictions=76800, candidates=115200)
    write(RUN / 'preflight.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def finish_report():
    import torch
    from tools.prepare_gaussian_budget_checkpoint import sha
    logs = sorted(RUN.glob('*.log'))
    text = '\n'.join(p.read_text(errors='replace') for p in logs if p.name != 'pipeline_console.log')
    values = re.findall(r'mIoU:\s*([\d.]+), iou2:\s*([\d.]+)', text)
    if not values:
        raise RuntimeError('Full validation did not produce final mIoU/IoU')
    state = torch.load(RUN / 'epoch_1.pth', map_location='cpu', weights_only=False)
    assert state['epoch'] == 1 and state['global_iter'] == 28130
    miou, iou = map(float, values[-1])
    result = dict(status='complete', epochs=1, training_frames=28130, training_scenes=700,
        optimizer_updates=14065, validation_frames=6019, validation_scenes=150,
        mIoU=miou, IoU=iou, checkpoint_sha256=sha(RUN / 'epoch_1.pth'),
        preflight=json.loads((RUN / 'preflight.json').read_text()))
    write(RUN / 'complete.json', result)
    baseline = json.loads((ROOT / 'work_dirs/small_camera_set_full_1ep/diagnostics/aggregate/comparison_summary.json').read_text())
    old = baseline['splits']['val']['models']['after']
    lines = ['# Query 重要性分支：完整 Occ3D 微调结果', '',
        '**状态：完整训练和完整验证已完成。**', '',
        '从原始 small 预训练权重初始化，在完整 700 场景、28,130 帧上微调 1 epoch，'
        '梯度累积为 2，共 14,065 次参数更新。验证覆盖完整 150 场景、6,019 帧，使用 camera mask。', '',
        '| 方法 | 训练候选 GS | 推理 GS | mIoU（%） | 二值 IoU（%） |',
        '| --- | ---: | ---: | ---: | ---: |',
        f"| 仅 camera-filtered Set 微调 | 76,800 | 76,800 | {old['mIoU']:.5f} | {old['IoU']:.5f} |",
        f'| 加入 query 重要性与软预算 | 115,200 | 76,800 | {miou:.5f} | {iou:.5f} |', '',
        f"相对上一组微调，mIoU 变化 {miou-old['mIoU']:+.5f} 个百分点，二值 IoU 变化 {iou-old['IoU']:+.5f} 个百分点。"
        '两组从相同原始权重出发；本轮同时改变了候选规模与选择方式，不能仅凭这两组拆分二者贡献。', '',
        '真实数据检查确认：Occupancy loss 能直接更新重要性分支；预算保持固定；'
        '最终 Set 监督与推理均为 76,800 个 GS。正式训练对非有限 loss 和梯度执行失败检查。', '',
        '本报告验证可训练性和完整验证性能。尚未重新统计训练后模型的完整几何覆盖，'
        '不能把性能变化直接解释为覆盖改善。', '',
        '## 复核文件', '',
        '- [实验设计](query_importance_camera_set_design.md)',
        '- [真实数据检查](../work_dirs/small_query_budget_camera_full_1ep/preflight.json)',
        '- [完成记录](../work_dirs/small_query_budget_camera_full_1ep/complete.json)',
        '- [初始化校验](../work_dirs/small_query_budget_camera_full_1ep/initialization_audit.json)',
        '- 日志：`work_dirs/small_query_budget_camera_full_1ep/train_console.log`', '']
    REPORT.write_text('\n'.join(lines), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'preflight', 'run'), nargs='?', default='run')
    parser.add_argument('--wait-pid', type=int)
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / 'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            from tools.prepare_gaussian_budget_checkpoint import prepare, sha
            status('preparing')
            prepare(CONFIG)
            if args.action == 'prepare':
                return
            if args.wait_pid:
                status('waiting_for_previous_diagnostics', wait_pid=args.wait_pid)
                progress_report('等待上一轮完整统计结束，随后执行 GPU 检查并训练')
                while True:
                    try:
                        os.kill(args.wait_pid, 0)
                    except ProcessLookupError:
                        break
                    time.sleep(10)
            status('preflight')
            progress_report('执行真实数据前向、反向和硬选择检查')
            # An independent child releases all preflight CUDA memory before
            # train.py starts, and its optimizer steps cannot leak into training.
            if args.action == 'run':
                command = [sys.executable, '-u', str(Path(__file__)), 'preflight']
                # This child does not take the parent's pipeline lock.
                environment = dict(os.environ, GAUSSIAN_BUDGET_PREFLIGHT_CHILD='1')
                with (RUN / 'preflight_console.log').open('w') as stream:
                    child = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
                if child.returncode:
                    raise RuntimeError(f'GPU preflight failed with exit code {child.returncode}; see preflight_console.log')
            else:
                preflight()
                status('preflight_passed')
                progress_report('真实数据检查通过，尚未启动正式训练')
                return
            files = [CONFIG, Path(__file__), ROOT / 'model/encoder/gaussian_budget_gate.py',
                ROOT / 'model/encoder/opus_v1_gaussian_budget_encoder.py',
                ROOT / 'model/head/opus_v1_gaussian_budget_head.py',
                ROOT / 'loss/gaussian_budget_loss.py', ROOT / 'tools/prepare_gaussian_budget_checkpoint.py']
            write(RUN / 'protocol.json', dict(source_sha256={str(p.relative_to(ROOT)): sha(p) for p in files},
                source_checkpoint_sha256=sha(ROOT / 'ckpts/opus_gaussian_8f_small.pth'),
                initialization_sha256=sha(RUN / 'initial_model.pth'),
                epochs=1, train_frames=28130, train_scenes=700, val_frames=6019, val_scenes=150))
            command = [sys.executable, '-u', 'train.py', '--py-config', str(CONFIG.relative_to(ROOT)),
                '--work-dir', str(RUN.relative_to(ROOT)), '--gradient-accumulation', '2']
            progress_report('真实数据检查通过，正式完整训练进行中')
            with (RUN / 'train_console.log').open('a') as stream:
                child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                status('training', training_pid=child.pid, command=command)
                while child.poll() is None:
                    with (RUN / 'train_console.log').open('rb') as reader:
                        reader.seek(0, 2)
                        size = reader.tell()
                        reader.seek(max(0, size - (128 << 10)))
                        tail = reader.read().decode(errors='replace')
                    matches = re.findall(r'\[TRAIN\] Epoch (\d+) Iter\s+(\d+)/(\d+)', tail)
                    latest = dict(epoch=int(matches[-1][0]), iteration=int(matches[-1][1]),
                                  total_iterations=int(matches[-1][2])) if matches else {}
                    phase = 'validation' if tail.rfind('[EVAL]') > tail.rfind('[TRAIN]') else 'training'
                    write(RUN / 'status.json', dict(status=phase, pid=os.getpid(), training_pid=child.pid,
                        updated_at=datetime.now().astimezone().isoformat(), **latest))
                    time.sleep(30)
                code = child.returncode
            if code:
                raise RuntimeError(f'Training failed with exit code {code}; see train_console.log')
            result = finish_report()
            status('complete', **{k: result[k] for k in ('mIoU', 'IoU', 'epochs')})
        except BaseException as error:
            status('failed', error=repr(error))
            progress_report('运行中断，尚无完整训练结论', str(error))
            raise


if __name__ == '__main__':
    if os.environ.get('GAUSSIAN_BUDGET_PREFLIGHT_CHILD') == '1':
        preflight()
    else:
        main()
