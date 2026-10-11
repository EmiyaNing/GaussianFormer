"""Evaluate forecast predictions or the real-feature oracle with direct IoU."""
import argparse
import json
import os
import sys

import torch
from mmengine import Config
from mmseg.models import build_segmentor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model  # noqa: F401,E402
import loss  # noqa: F401,E402
from dataset import get_dataloader  # noqa: E402
from loss import OPENOCC_LOSS  # noqa: E402


def scores(prediction, target, empty=17, classes=17):
    prediction, target = prediction.reshape(-1), target.reshape(-1)
    ious = []
    for index in range(classes):
        union = ((prediction == index) | (target == index)).sum()
        if union:
            intersection = ((prediction == index) & (target == index)).sum()
            ious.append(float(intersection.float() / union))
    pred_occ, target_occ = prediction != empty, target != empty
    occ_union = (pred_occ | target_occ).sum()
    occ_iou = float((pred_occ & target_occ).sum().float() / occ_union)
    return dict(miou=100. * sum(ious) / max(len(ious), 1),
                occ_iou=100. * occ_iou, present_classes=len(ious))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--oracle', action='store_true')
    parser.add_argument('--max-batches', type=int, default=8)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    network = build_segmentor(cfg.model)
    network.init_weights()
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        network.load_state_dict(state_dict, strict=True)
    network = network.cuda().eval()
    objective = OPENOCC_LOSS.build(cfg.loss).cuda().eval()
    _, loader = get_dataloader(
        cfg.train_dataset_config, cfg.val_dataset_config,
        cfg.train_loader, cfg.val_loader, val_only=True)
    records = []
    for batch_index, batch in enumerate(loader):
        if batch_index >= args.max_batches:
            break
        images = batch.pop('img').cuda()
        for key, value in list(batch.items()):
            if isinstance(value, torch.Tensor):
                batch[key] = value.cuda()
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=cfg.get('amp', False)):
            output = network(imgs=images, metas=batch,
                             oracle_future_features=args.oracle)
            _, metrics = objective(dict(
                forecasting_output=output['forecasting_output'], metas=batch))
        current = output['forecasting_output']['current_opus']['final_occ_grid']
        record = dict(batch=batch_index,
                      current=scores(current, batch['occ_label']), futures=[])
        for future in output['forecasting_output']['future_predictions']:
            horizon = int(future['horizon'])
            record['futures'].append(dict(
                horizon=horizon + 1,
                **scores(future['opus']['final_occ_grid'],
                         batch['future_occ_labels'][:, horizon])))
        record['loss'] = {
            name: float(metrics[name]) for name in
            ('current_total', 'future_total', 'latent_alignment',
             'pyramid_alignment')}
        records.append(record)
    aggregate = {}
    for section in ('current',):
        aggregate[section] = {
            key: sum(record[section][key] for record in records) / len(records)
            for key in ('miou', 'occ_iou')}
    aggregate['futures'] = []
    for index in range(cfg.model.num_future):
        values = [record['futures'][index] for record in records]
        aggregate['futures'].append(dict(
            horizon=index + 1,
            miou=sum(value['miou'] for value in values) / len(values),
            occ_iou=sum(value['occ_iou'] for value in values) / len(values)))
    result = dict(config=args.config, checkpoint=args.checkpoint,
                  oracle=args.oracle, batches=len(records),
                  aggregate=aggregate, records=records)
    with open(args.output, 'w') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(aggregate, indent=2))


if __name__ == '__main__':
    main()
