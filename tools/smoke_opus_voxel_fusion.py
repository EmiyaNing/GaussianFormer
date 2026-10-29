"""Run one real sample through fixed-600 voxel OPUS fusion, loss and backward."""

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from mmengine import Config
from mmseg.models import build_segmentor

import model  # noqa: F401
from dataset import OPENOCC_DATASET
from dataset.utils import custom_collate_fn_temporal
from loss import OPENOCC_LOSS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--py-config', default=
                        'config/opus/opusv1_voxel_fusion_t_r50_704x256_1f_occ3d.py')
    parser.add_argument('--sample-index', type=int, default=128)
    parser.add_argument('--memory-fraction', type=float, default=0.5)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('This integration smoke requires a CUDA device')
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    torch.manual_seed(928)
    config = Config.fromfile(args.py_config)
    dataset = OPENOCC_DATASET.build(config.val_dataset_config)
    sample_index = args.sample_index % len(dataset)
    sample = dataset[sample_index]
    batch = custom_collate_fn_temporal([sample])
    batch = {key: value.cuda() if isinstance(value, torch.Tensor) else value
             for key, value in batch.items()}
    images = batch.pop('img')
    network = build_segmentor(config.model).cuda().train()
    loss_module = OPENOCC_LOSS.build(config.loss).cuda()
    # Keep the configured Q=600, six decoders and complete GT sets. No
    # pretrained weights are needed to verify the integration's gradients.
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = network(imgs=images, metas=batch)
    torch.cuda.synchronize()
    forward_seconds = time.perf_counter() - start
    memory = result['voxel_memory']
    with torch.no_grad():
        repeat = network.lifter.initialize_queries(memory)
    torch.testing.assert_close(result['query_points'], repeat['query_points'], rtol=0, atol=0)
    torch.testing.assert_close(result['query_features'], repeat['query_features'], rtol=0, atol=0)
    inputs = {'metas': batch}
    for name, key in config.loss_input_convertion.items():
        inputs[name] = result[key]
    loss, details = loss_module(inputs)
    assert torch.isfinite(loss), details
    loss.backward()
    torch.cuda.synchronize()
    gradient_norms = {}
    for name, module in [('image_backbone', network.img_backbone),
                         ('lidar_backbone', network.lifter.lidar_backbone),
                         ('memory_projection', network.lifter.memory_projection),
                         ('fusion', network.encoder.layers[0].mixing),
                         ('sampling_offsets', network.encoder.layers[0].sampling.sampling_offset),
                         ('final_classifier', network.encoder.layers[-1].cls_branch),
                         ('final_regressor', network.encoder.layers[-1].reg_branch)]:
        gradients = [p.grad for p in module.parameters() if p.requires_grad and p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients), name
        value = sum(float(g.abs().sum()) for g in gradients)
        assert value > 0, name
        gradient_norms[name] = value
    query_count = config.model.lifter.num_queries
    assert query_count == 600
    assert result['opus_pred_points'][-1].shape == (1, query_count * 128, 3)
    row = dict(config=args.py_config, sample_index=sample_index,
               camera_images=images.shape[1], lidar_points=len(sample['lidar_points']),
               requested_lidar_history=config.val_dataset_config.num_lidar_history,
               memory_voxels=memory['features'].shape[0],
               memory_shape=list(memory['spatial_shape']),
               voxel_initialized_queries=int(result['query_voxel_valid_mask'].sum()),
               query_count=query_count, final_points=result['opus_pred_points'][-1].shape[1],
               forward_seconds=forward_seconds,
               total_seconds=time.perf_counter() - start, loss=float(loss.detach()),
               losses={key: float(value) for key, value in details.items()},
               gradient_l1=gradient_norms,
               lidar_valid_fraction=[float(state['lidar_sample_valid'].float().mean())
                                     for state in result['representation']],
               peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(row, indent=2) + '\n')
    print(json.dumps(row, indent=2))


if __name__ == '__main__':
    main()
