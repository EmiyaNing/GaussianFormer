"""Strictly expand only final output rows of a trained OPUS Gaussian model."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from mmengine import Config


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(config, destination=None):
    from mmengine.runner import set_random_seed
    from mmseg.models import build_segmentor
    import model
    cfg = Config.fromfile(config)
    set_random_seed(cfg.get('seed', 928))
    network = build_segmentor(cfg.model)
    target = network.state_dict()
    source_path = ROOT / cfg.budget_source_checkpoint
    source_ckpt = torch.load(source_path, map_location='cpu', weights_only=False)
    source = source_ckpt['state_dict']
    old = int(cfg.budget_source_children)
    new = network.encoder.layers[-1].num_refine
    if not old < new <= 2 * old:
        raise ValueError('Initializer supports an expansion between 1x and 2x')
    extra = new - old
    copied_children = torch.arange(extra) * old // extra
    mapping = torch.cat((torch.arange(old), copied_children))
    layer = network.encoder.layers[-1]
    branches = [(layer.cls_branch[-1], 17, 'class'),
                (layer.reg_branch[-1], 3, 'position'),
                (network.encoder.attribute_branches[str(len(network.encoder.layers)-1)].net[-1], 8, 'attributes')]
    names = {id(module): name for name, module in network.named_modules()}
    expanded = {}
    for module, channels, kind in branches:
        for suffix in ('weight', 'bias'):
            expanded[names[id(module)] + '.' + suffix] = (channels, kind, suffix)
    added = []
    unchanged = []
    for name, value in target.items():
        if name in expanded:
            channels, kind, suffix = expanded[name]
            before = source[name]
            assert before.shape[0] == old * channels
            row = before.reshape(old, channels, *before.shape[1:])[mapping].clone()
            if kind == 'position' and suffix == 'bias':
                # Decoder offsets are in metres after multiplication by scale.
                angle = torch.arange(extra, dtype=row.dtype)[:, None]
                direction = torch.cat((torch.sin(angle + .7), torch.cos(angle + .7),
                                       torch.sin(angle * 1.7 + .3)), dim=1)
                direction = torch.nn.functional.normalize(direction, dim=-1)
                row[old:] += float(cfg.budget_extra_offset_m) * direction / float(layer.scale)
            if kind == 'attributes' and suffix == 'bias':
                row[old:, 7] += float(cfg.budget_extra_opacity_logit_shift)
            target[name] = row.reshape_as(value)
            assert torch.equal(target[name][:old * channels], before)
        elif name in source:
            assert source[name].shape == value.shape, name
            target[name] = source[name].clone()
            unchanged.append(name)
        elif name.startswith('encoder.importance_branch.'):
            added.append(name)
        else:
            raise ValueError('Unexpected parameter absent from pretrained model: ' + name)
    assert set(source) <= set(target), set(source) - set(target)
    network.load_state_dict(target, strict=True)
    probe = torch.randn(2, 11, network.encoder.importance_branch[0].in_features)
    assert torch.equal(network.encoder.importance_branch(probe), torch.zeros(2, 11, 1))
    report = dict(status='passed', source_checkpoint=str(source_path.relative_to(ROOT)),
        source_sha256=sha(source_path), source_epoch=source_ckpt.get('epoch'),
        source_global_iter=source_ckpt.get('global_iter'), source_children=old,
        candidate_children=new, original_rows_exact=True, strict_load=True,
        expanded_tensors=sorted(expanded), added_parameters=added,
        unchanged_tensors=len(unchanged), extra_source_children=copied_children.tolist(),
        extra_position_offset_m=float(cfg.budget_extra_offset_m),
        extra_opacity_logit_shift=float(cfg.budget_extra_opacity_logit_shift),
        importance_initial_score=.5,
        config_sha256=hashlib.sha256(cfg.pretty_text.encode()).hexdigest())
    destination = Path(destination) if destination else ROOT / cfg.train_load_from
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        existing = torch.load(destination, map_location='cpu', weights_only=False)
        assert existing.get('allocation_initialization') == report, 'Existing initial checkpoint differs'
        assert existing['state_dict'].keys() == target.keys()
        assert all(torch.equal(existing['state_dict'][k], v) for k, v in target.items())
    else:
        temporary = destination.with_suffix('.tmp')
        torch.save(dict(state_dict=target, epoch=0, global_iter=0,
                        allocation_initialization=report), temporary)
        temporary.replace(destination)
    report['checkpoint_sha256'] = sha(destination)
    (destination.parent / 'initialization_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--py-config', required=True)
    parser.add_argument('--out')
    args = parser.parse_args()
    print(json.dumps(prepare(args.py_config, args.out), indent=2), flush=True)
