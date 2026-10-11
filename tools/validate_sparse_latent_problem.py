"""Validate sparse-state sufficiency and stage-one baselines on real data."""
import argparse
import json
import os
import sys

import torch
from mmengine import Config
from mmseg.models import build_segmentor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model  # noqa: F401,E402
from dataset import get_dataloader  # noqa: E402
from model.sparse_latent_dynamics import transform_points  # noqa: E402


def scores(prediction, target, empty=17, classes=17):
    prediction, target = prediction.reshape(-1), target.reshape(-1)
    ious = []
    for class_index in range(classes):
        union = ((prediction == class_index) |
                 (target == class_index)).sum()
        if union:
            intersection = ((prediction == class_index) &
                            (target == class_index)).sum()
            ious.append(float(intersection.float() / union))
    pred_occ, target_occ = prediction != empty, target != empty
    union = (pred_occ | target_occ).sum()
    occ_iou = float((pred_occ & target_occ).sum().float() / union)
    return dict(miou=100. * sum(ious) / max(len(ious), 1),
                occ_iou=100. * occ_iou)


def mean_metrics(records, key):
    values = [record[key] for record in records]
    return {name: sum(value[name] for value in values) / len(values)
            for name in ('miou', 'occ_iou')}


def canonical_sparse(output, voxel_num):
    logits = output['all_cls_scores'][-1]
    coords = output['all_voxel_coors'][-1].long()
    yz = int(voxel_num[1] * voxel_num[2])
    xyz = int(voxel_num.prod())
    linear = (coords[:, 0] * xyz + coords[:, 1] * yz +
              coords[:, 2] * int(voxel_num[2]) + coords[:, 3])
    order = linear.argsort()
    return linear[order], logits[order]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--max-batches', type=int, default=4)
    parser.add_argument('--distinct-scenes', action='store_true')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    network = build_segmentor(cfg.model)
    network.init_weights()
    # SparseLatentWorld.init_weights performs the audited 8-frame -> 1-frame
    # OPUS mixing conversion.  Keep the CLI checkpoint explicit and verify it
    # is the same source configured on the model.
    if os.path.abspath(args.checkpoint) != os.path.abspath(
            network.current_occupancy_checkpoint):
        raise ValueError('--checkpoint must match model.current_occupancy_checkpoint')
    network = network.cuda().eval()
    _, loader = get_dataloader(
        cfg.train_dataset_config, cfg.val_dataset_config,
        cfg.train_loader, cfg.val_loader, val_only=True)
    if args.distinct_scenes:
        selected, seen = [], set()
        for keyframe in loader.dataset.keyframes:
            if keyframe[0] in seen:
                continue
            selected.append(keyframe)
            seen.add(keyframe[0])
            if len(selected) >= args.max_batches:
                break
        loader.dataset.keyframes = selected

    records = []
    exact_voxels = total_voxels = 0
    max_state_logit_error = 0.
    for batch_index, batch in enumerate(loader):
        if batch_index >= args.max_batches:
            break
        images = batch.pop('img').cuda(non_blocking=True)
        for key, value in list(batch.items()):
            if isinstance(value, torch.Tensor):
                batch[key] = value.cuda(non_blocking=True)
        frames = images.reshape(
            images.shape[0], -1, network.num_cams, *images.shape[2:])
        with torch.no_grad():
            current = network._encode_frame(
                frames[:, 0], network._frame_metas(
                    batch, slice(0, network.num_cams)))
            current_state = network._state(current)
            current_score = scores(current['final_occ_grid'], batch['occ_label'])
            for horizon in range(network.num_future):
                frame_index = network.num_history + 1 + horizon
                start = frame_index * network.num_cams
                teacher = network._encode_frame(
                    frames[:, frame_index], network._frame_metas(
                        batch, slice(start, start + network.num_cams)))
                teacher_state = network._state(teacher)
                roundtrip = network.head.decode_sparse_state(
                    teacher_state['points'], teacher_state['point_feats'],
                    teacher_state['query_feats'])
                equality = (roundtrip['final_occ_grid'] ==
                            teacher['final_occ_grid'])
                exact_voxels += int(equality.sum())
                total_voxels += equality.numel()
                teacher_coord, teacher_logits = canonical_sparse(
                    teacher, network.head.voxel_num)
                state_coord, state_logits = canonical_sparse(
                    roundtrip, network.head.voxel_num)
                if not torch.equal(teacher_coord, state_coord):
                    raise AssertionError('state re-decode changed sparse coordinates')
                max_state_logit_error = max(
                    max_state_logit_error,
                    float((state_logits - teacher_logits).abs().max()))

                current_to_future = torch.linalg.inv(
                    batch['future_ego_to_current'][:, horizon].float())
                warped_points = transform_points(
                    current_state['points_metric'], current_to_future)
                warped = network.head.decode_sparse_state(
                    network._encode_points(warped_points).clamp(0., 1.),
                    current_state['point_feats'], current_state['query_feats'])
                target = batch['future_occ_labels'][:, horizon]
                records.append(dict(
                    batch=batch_index,
                    scene_token=(batch.get('scene_token', ['unknown'])[0]
                                 if isinstance(batch.get('scene_token'), list)
                                 else str(batch.get('scene_token', 'unknown'))),
                    horizon=horizon + 1,
                    current=current_score,
                    oracle=scores(teacher['final_occ_grid'], target),
                    roundtrip=scores(roundtrip['final_occ_grid'], target),
                    ego_warped_current=scores(warped['final_occ_grid'], target),
                    center_shift_m=float((
                        teacher_state['points_metric'].mean(2) -
                        transform_points(
                            current_state['points_metric'],
                            current_to_future).mean(2)
                    ).norm(dim=-1).mean())))

    result = dict(
        config=args.config, checkpoint=args.checkpoint,
        batches=len(set(record['batch'] for record in records)),
        autoencoding=dict(
            voxel_agreement=exact_voxels / max(total_voxels, 1),
            max_sparse_logit_abs_error=max_state_logit_error),
        aggregate=dict(
            current=mean_metrics(records, 'current'),
            oracle=mean_metrics(records, 'oracle'),
            roundtrip=mean_metrics(records, 'roundtrip'),
            ego_warped_current=mean_metrics(records, 'ego_warped_current'),
            oracle_gain_over_warped_miou=sum(
                record['oracle']['miou'] -
                record['ego_warped_current']['miou']
                for record in records) / len(records),
            mean_teacher_warped_center_gap_m=sum(
                record['center_shift_m'] for record in records) / len(records)),
        records=records)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result['autoencoding'] | result['aggregate'], indent=2))


if __name__ == '__main__':
    main()
