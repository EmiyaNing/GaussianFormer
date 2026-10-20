"""Full capacity, real eight-frame train/backward/AdamW and validation on CUDA."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from mmengine import Config
from mmengine.optim import build_optim_wrapper
from mmseg.models import build_segmentor
import model  # noqa: F401
from dataset import OPENOCC_DATASET
from dataset.utils import custom_collate_fn_temporal
from loss import OPENOCC_LOSS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', default='tiny', choices=['tiny', 'small', 'middle', 'large'])
    parser.add_argument('--py-config', help='Override the full-data size configuration')
    parser.add_argument('--out', type=Path, help='Independent preflight result path')
    parser.add_argument('--steps', type=int, default=2)
    args = parser.parse_args()
    cfg = Config.fromfile(args.py_config or f'config/full_occ3d_oups_guassian/'
                          f'opusv1_gaussian_softplus_learned_bias_{args.size}_8f.py')
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.backends.cudnn.benchmark = False
    dataset = OPENOCC_DATASET.build(cfg.train_dataset_config)
    full_train_size = len(dataset)
    # Find a sample deep enough into a scene to have seven real sweeps.
    index = next(i for i, (_, frame) in enumerate(dataset.keyframes) if frame >= 90)
    scene, frame = dataset.keyframes[index]
    selection = dataset.pipeline[1]
    ctx = dict(scene_infos=dataset.scene_infos, scene_token=scene,
               frame_index=frame, sensor_types=dataset.sensor_types)
    history, count = selection.select_indices(ctx)
    assert count == 7 and all(i < frame for i in history), (history, frame)
    # Test scene-start padding and deterministic validation selection.
    start = dict(ctx, frame_index=0)
    assert selection.select_indices(start)[0] == [0] * 7
    val_selection = type(selection)(test_mode=True)
    key_history, _ = val_selection.select_indices(ctx)
    assert all(i <= frame and (i == frame or selection.is_keyframe(
        dataset.scene_infos[scene][i])) for i in key_history)
    # Reset so audited indices are the actual loaded training frames.
    np.random.seed(cfg.seed)
    dataset.return_keys = list(dataset.return_keys) + ['history_frame_indices']
    item = dataset[index]
    loaded_indices = item.pop('history_frame_indices').tolist()
    batch = custom_collate_fn_temporal([item])
    unmasked = cfg.get('occ3d_eval_mask') == 'none'
    if unmasked:
        assert 'occ_cam_mask' not in batch
        assert batch['occ_loss_mask'].bool().all()
        assert cfg.loss_input_convertion.occ_mask == 'occ_loss_mask'
    validation = OPENOCC_DATASET.build(cfg.val_dataset_config)
    val_samples = len(validation)
    val_batch = custom_collate_fn_temporal([validation[0]])
    if unmasked:
        from train import select_occ3d_eval_mask
        assert 'occ_cam_mask' not in val_batch
        assert val_batch['occ_loss_mask'].bool().all()
        assert select_occ3d_eval_mask(val_batch, 0, cfg) is None
    del validation
    del dataset, item
    batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    images = batch.pop('img')
    assert images.shape == (1, 48, 3, 256, 704), images.shape
    network = build_segmentor(cfg.model).cuda().train()
    state = torch.load(cfg.load_from, map_location='cpu', weights_only=False)['state_dict']
    backbone = {k.removeprefix('backbone.'): v for k, v in state.items()
                if k.startswith('backbone.')}
    report = network.img_backbone.load_state_dict(backbone, strict=False)
    assert not report.missing_keys and not report.unexpected_keys, report
    del state, backbone
    optimizer = build_optim_wrapper(network, cfg.optimizer)
    criterion = OPENOCC_LOSS.build(cfg.loss).cuda()
    frame_gradients = []
    handle = network.img_neck.register_full_backward_hook(
        lambda module, grad_in, grad_out: frame_gradients.append(
            sum(float(g.float().abs().sum()) for g in grad_out if g is not None)))
    torch.cuda.reset_peak_memory_stats()
    timings, losses = [], []
    for step in range(args.steps):
        torch.cuda.synchronize()
        started = time.perf_counter()
        output = network(imgs=images, metas=batch)
        assert output['pred_occ'][-1].shape == (1, 18, 640000)
        assert output['opus_pred_points'][-1].shape == (1, 76800, 3)
        inputs = {'metas': batch}
        inputs.update({name: output[key] for name, key in cfg.loss_input_convertion.items()})
        value, components = criterion(inputs)
        assert torch.isfinite(value), components
        (value / cfg.gradient_accumulation).backward()
        names = ['encoder.attribute_branches.5.net.2.weight',
                 'encoder.layers.5.reg_branch.4.weight',
                 'encoder.layers.5.cls_branch.6.weight', 'head.empty_bias']
        params = dict(network.named_parameters())
        for name in names:
            grad = params[name].grad
            assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, name
        bad = [name for name, p in network.named_parameters()
               if p.grad is not None and not torch.isfinite(p.grad).all()]
        assert not bad, bad[:10]
        torch.nn.utils.clip_grad_norm_(network.parameters(), cfg.grad_max_norm,
                                       error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - started)
        losses.append(float(value.detach()))
        print('step', step, 'loss', losses[-1], 'seconds', timings[-1], flush=True)
        del output, inputs, value, params, grad
    handle.remove()
    assert len(frame_gradients) == 8 * args.steps and all(v > 0 for v in frame_gradients), frame_gradients
    train_peak = dict(allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                      reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
    optimizer.zero_grad(set_to_none=True)
    network.eval()
    torch.cuda.reset_peak_memory_stats()
    val_batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in val_batch.items()}
    val_images = val_batch.pop('img')
    with torch.no_grad():
        evaluation = network(imgs=val_images, metas=val_batch)
        assert torch.isfinite(evaluation['pred_occ'][-1]).all()
        val_inputs = {'metas': val_batch}
        val_inputs.update({name: evaluation[key] for name, key in cfg.loss_input_convertion.items()})
        val_loss, _ = criterion(val_inputs)
        assert torch.isfinite(val_loss)
    val_peak = dict(allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                    reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
    result = dict(size=args.size, passed=True, train_samples=full_train_size,
                  val_samples=val_samples, config=args.py_config, camera_mask_disabled=unmasked,
                  validation_loss=float(val_loss),
                  image_shape=list(images.shape), history_frame_indices=loaded_indices,
                  validation_keyframe_indices=key_history, loss=losses,
                  full_train_step_seconds=timings, train_peak=train_peak, val_peak=val_peak,
                  all_frame_feature_gradient_l1=frame_gradients,
                  num_parameters=sum(p.numel() for p in network.parameters()))
    path = args.out or Path('work_dirs/full_occ3d_gaussian_8f_30e/preflight') / f'{args.size}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
