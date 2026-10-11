"""Shared native-head utilities for isolated causal experiments."""
from pathlib import Path
import copy
import torch
import torch.nn.functional as F
from mmengine import Config

from tools.train_future_predictable_full import build_full_student, historical_config, image_loader
from tools.diagnose_teacher_alignment import loader_for, tree_to, add_metric, seed_all
from tools.experiment_future_response_distillation import prepare_student
from tools.experiment_future2current import teacher_at_student
from tools.evaluate_future_predictable_native import summarized_rows, relative_to_baseline
from tools.evaluate_future2current_native import save_rows
from tools.experiment_future_predictable_information import save_json
from tools.diagnose_future_image_joint import metric_row
from eval_teacher_aligned import _metric
from loss import OPENOCC_LOSS

HORIZONS = (2, 4, 6)
FORECAST = ('ego_attention', 'position', 'semantic', 'reg', 'velocity', 'traj', 'future_adapter')


def deterministic():
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def model_for(cfg, checkpoint=None):
    strict = Config.fromfile(cfg.student_model_config)
    original = torch.load(cfg.student_checkpoint, map_location='cpu', weights_only=False, mmap=True)['state_dict']
    instance = build_full_student(cfg, strict, original).eval()
    if checkpoint:
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
        instance.load_state_dict(saved['state_dict'], strict=True)
    return instance, strict


def selected_images(cfg, split, count):
    return loader_for(historical_config(cfg), split, count, cfg.causal_workers, cfg.identity_sample_seed)


def teacher_record(cfg, split, index, scene):
    path = Path(cfg.causal_reference_root) / 'teacher_features' / split / f'{index:06d}.pth'
    row = torch.load(path, map_location='cpu', weights_only=False)
    if (row['index'], row['scene'], row['checkpoint_sha256']) != (index, scene, cfg.student_checkpoint_sha256):
        raise RuntimeError('Teacher sample/checkpoint mismatch')
    return row


def configure_forecast(instance, heads_trainable=True):
    instance.requires_grad_(False)
    for name in FORECAST:
        if not heads_trainable and name in ('semantic', 'reg', 'velocity'):
            continue
        getattr(instance, name).requires_grad_(True)
    # Frozen image features; only the future module is trained in this screen.
    instance.eval()
    instance.ego_attention.train()


def cached_rollout(instance, source, geometry_grad=True, detach_geometry=False, ego_override=None):
    encoded = tree_to(source['encoded'], 'cuda')
    stamps = source['stamps'].cuda()
    features = encoded['query_features'][:, stamps == 0]
    positions = encoded['query_points'][:, stamps == 0].detach()
    timestamp = positions.new_zeros(*positions.shape[:-1], 1)
    ego = source['ego'].cuda() if ego_override is None else ego_override
    points, logits, states, trajectories = [], [], [], []
    for step in range(6):
        fused = instance.ego_attention(ego, features.detach(), positions.detach())
        trajectories.append(instance.traj(fused))
        born = stamps == step + 1
        features = torch.cat([features, encoded['query_features'][:, born]], 1)
        positions = torch.cat([positions, encoded['query_points'][:, born]], 1).detach()
        timestamp = torch.cat([timestamp, timestamp.new_full((1, int(born.sum()), 48, 1), .5)], 1)
        features = features + fused.expand(-1, features.shape[1], -1) + instance.position(
            torch.cat([positions, timestamp], -1).flatten(2))
        adapted = instance.future_adapter(features)
        decoded, semantic = instance.decode(adapted, positions)
        output_positions = decoded if geometry_grad and not detach_geometry else decoded.detach()
        points.append(instance._world(output_positions).flatten(1, 2).float())
        logits.append(semantic.flatten(1, 2).float())
        states.append(dict(features=adapted, positions=decoded.detach(),
                           interaction_positions=positions.detach(), baseline_features=features))
        with torch.no_grad():
            positions, _ = instance.decode(features, positions)
    return dict(future_pred_points=points, future_pred_logits=logits, future_query_states=states,
                pred_traj=torch.cat(trajectories, 1).float(), metas=tree_to(source['metas'], 'cuda'))


def future_loss(objective, output):
    total = output['pred_traj'].sum() * 0.
    for step, (points, logits) in enumerate(zip(output['future_pred_points'], output['future_pred_logits'])):
        cls, pts = objective._set_loss(points[0], logits[0], output['metas']['future_occ_labels'][0, step])
        total = total + objective.lambda_future_cls * cls + objective.lambda_future_pts * pts
    trajectory = F.mse_loss(output['pred_traj'], output['metas']['temporal_trajs'].to(output['pred_traj']))
    return total + objective.lambda_traj * trajectory


def matched_targets(instance, output, teacher):
    return {h: teacher_at_student(instance, teacher['states'][h],
                                  output['future_query_states'][h - 1]['positions'], 4, 4.)
            for h in HORIZONS}


def feature_loss(output, targets):
    values = []
    for h, (target, valid) in targets.items():
        if valid.any():
            features = output['future_query_states'][h - 1]['features'][:, valid].float()
            target = target[:, valid].float()
            values.append(F.smooth_l1_loss(F.layer_norm(features, (256,)), F.layer_norm(target, (256,))) +
                          (1 - F.cosine_similarity(features, target, dim=-1)).mean())
    return torch.stack(values).mean() if values else output['pred_traj'].sum() * 0.


def prediction_rows(instance, output, cfg):
    rows = {}
    for h in HORIZONS:
        metric = _metric(cfg)
        add_metric(metric, instance, output['future_pred_points'][h - 1],
                   output['future_pred_logits'][h - 1], output['metas'], h)
        rows[h] = metric_row(metric)
    return rows


def prepare_sources(cfg, root):
    instance, _ = model_for(cfg)
    instance.requires_grad_(False)
    manifests = {}
    for split, count in [('train', cfg.causal_train_samples), ('val', cfg.causal_val_samples)]:
        images, manifest = selected_images(cfg, split, count)
        prepare_student(instance, images, manifest, root / 'source_cache', split)
        manifests[split] = manifest
        for index, scene in zip(manifest['indices'], manifest['scenes']):
            teacher_record(cfg, split, index, scene)
    if set(manifests['train']['scenes']) & set(manifests['val']['scenes']):
        raise RuntimeError('Train/val scenes overlap')
    save_json(root / 'manifest.json', dict(checkpoint_sha256=cfg.student_checkpoint_sha256,
              train=manifests['train'], val=manifests['val'],
              native_heads=True, masks='none', future_born_preserved=True,
              training_scope='future modules only for mechanism screens; existing all-parameter run unchanged'))
    del instance
    torch.cuda.empty_cache()
    return manifests


def source_records(root, manifest, split, limit=None):
    count = min(limit or len(manifest['indices']), len(manifest['indices']))
    for index, scene in zip(manifest['indices'][:count], manifest['scenes'][:count]):
        source = torch.load(root / 'source_cache' / split / f'{index:06d}.pth', map_location='cpu', weights_only=False)
        if (source['index'], source['scene']) != (index, scene):
            raise RuntimeError('Source sample identity differs')
        yield source


@torch.no_grad()
def evaluate_cached(instance, cfg, root, manifest, destination, limit=None):
    instance.eval()
    rows = [prediction_rows(instance, cached_rollout(instance, source), cfg)
            for source in source_records(root, manifest, 'val', limit)]
    save_rows(destination, rows)
    result = summarized_rows(rows, cfg)
    result['rows'] = str(destination)
    return result
