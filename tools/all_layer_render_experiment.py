"""Shared, isolated Small all-layer rendering experiment utilities."""
from pathlib import Path
import gc
import hashlib
import os
import random
import sys
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from mmengine import Config
from tools.gaussian_geometry_causal import write, sha, seed_all, to_cuda, forward, loss_input
from tools.train_dcd_mini import numpy_rng_checkpoint, restore_numpy_rng

RUN = ROOT / 'work_dirs/all_layer_render_small_mini_v1'
INITIAL = ROOT / 'ckpts/opus_gaussian_8f_small.pth'
ARMS = ('baseline', 'all_layers')


def prepare_annotation():
    import json
    import pickle
    source = ROOT / 'data/mini_occ3d_4096_1024/train.pkl'
    target = RUN / 'annotations/train_4090.pkl'
    manifest_path = target.with_suffix('.json')
    source_sha = sha(source)
    if manifest_path.exists():
        record = json.loads(manifest_path.read_text())
        assert record['source_sha256'] == source_sha and record['frames'] == 4090
        assert record['annotation_sha256'] == sha(target)
        return record
    with source.open('rb') as stream:
        data = pickle.load(stream)
    assert len(data['metadata']) == 4096
    selected = np.sort(np.random.default_rng(928).choice(4096, 4090, replace=False))
    removed = np.setdiff1d(np.arange(4096), selected)
    data['metadata'] = [data['metadata'][i] for i in selected]
    scenes = {row[0] for row in data['metadata']}
    data['infos'] = {s: values for s, values in data['infos'].items() if s in scenes}
    assert len(scenes) == 698
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.with_suffix('.tmp').open('wb') as stream:
        pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    target.with_suffix('.tmp').replace(target)
    record = dict(frames=4090, scenes=len(scenes), seed=928,
        source=str(source.relative_to(ROOT)), source_sha256=source_sha,
        selected_source_indices=selected.tolist(), omitted_source_indices=removed.tolist(),
        annotation_sha256=sha(target))
    write(manifest_path, record)
    return record


def config(arm):
    return Config.fromfile(ROOT / f'config/mini_all_layer_render/small_{arm}_1ep.py')


def register():
    import model
    import loss.all_layer_occupancy_loss
    import model.head.opus_all_layer_experiment_head


def build(arm, checkpoint=None, training=False):
    register()
    from mmseg.models import build_segmentor
    from mmengine.optim import build_optim_wrapper
    from loss import OPENOCC_LOSS
    cfg = config(arm)
    seed_all(cfg.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    net = build_segmentor(cfg.model)
    state = torch.load(checkpoint or INITIAL, map_location='cpu', weights_only=False)
    weights = state.get('state_dict', state)
    loaded = net.load_state_dict(weights, strict=False)
    expected = {f'encoder.attribute_branches.{i}.net.{layer}.{kind}'
                for i in range(5) for layer in (0, 2) for kind in ('weight', 'bias')}
    if arm == 'all_layers' and checkpoint is None:
        assert set(loaded.missing_keys) == expected and not loaded.unexpected_keys, loaded
    else:
        assert not loaded.missing_keys and not loaded.unexpected_keys, loaded
    net = net.cuda()
    if not training:
        net.eval().requires_grad_(False)
        net.head.render_all_eval = True
        return cfg, net
    opt = build_optim_wrapper(net, cfg.optimizer)
    if 'experiment_microbatches' in state:
        assert checkpoint is not None
        opt.load_state_dict(state['optimizer'])
    objective = OPENOCC_LOSS.build(cfg.loss).cuda()
    return cfg, net.train(), opt, objective, state


class TrainingFrames:
    def __init__(self, dataset, seed):
        self.dataset, self.seed = dataset, seed
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        np.random.seed(self.seed * 1000000 + index)
        random.seed(self.seed * 1000000 + index)
        return self.dataset[index]


def fingerprint(batch):
    """Native transformed current and history images, projections and GT."""
    digest = hashlib.sha256()
    for key in ('img', 'projection_mat', 'occ_label', 'occ_cam_mask'):
        a = batch[key].detach().cpu().contiguous().numpy()
        digest.update(a.tobytes())
    return digest.hexdigest()


def source_files():
    files = [ROOT / x for x in (
        'tools/all_layer_render_experiment.py', 'tools/train_all_layer_render_mini.py',
        'tools/run_all_layer_render_mini.py', 'tools/pilot_all_layer_render_mini.py',
        'tools/eval_all_layer_render_mini.py', 'loss/all_layer_occupancy_loss.py',
        'tests/test_all_layer_occupancy.py', 'model/head/opus_all_layer_experiment_head.py',
        'model/encoder/opus_v1_gaussian_encoder.py', 'model/encoder/opus_v1_gaussian_softplus_encoder.py',
        'model/encoder/opus_v1_gaussian_temporal_encoder.py', 'loss/occupancy_loss.py',
        'loss/opus_set_loss.py', 'loss/multi_loss.py', 'loss/utils/lovasz_softmax.py',
        'tools/train_dcd_mini.py', 'tools/gaussian_geometry_causal.py',
        'tools/stat_decoder_layers.py', 'tools/decoder_layer_stats/kernels.cu',
        'tools/decoder_layer_stats/__init__.py', 'tools/decoder_layer_stats/common.py',
        'tools/query_foreground_distance/kernels.cu', 'tools/query_foreground_distance/__init__.py')]
    import json
    previous = json.loads((ROOT / 'work_dirs/gaussian_sizes_full_val/small/protocol.json').read_text())
    files += [ROOT / f for f in previous['source_sha256']]
    files += sorted((ROOT / 'config/mini_all_layer_render').glob('*.py'))
    return list(dict.fromkeys(p for p in files if p.is_file()))


def assert_protocol():
    import json
    protocol = json.loads((RUN / 'protocol.json').read_text())
    for relative, digest in protocol['sources'].items():
        assert sha(ROOT / relative) == digest, relative
    assert sha(INITIAL) == protocol['initial_sha256']
    for phase, digest in protocol['annotations'].items():
        assert sha(config('baseline')[phase + '_dataset_config'].imageset) == digest
    return protocol


def release(net):
    net.flush_memory_phase()
    if hasattr(net, '_decoder_raw_points'):
        del net._decoder_raw_points
    gc.collect()
    torch.cuda.empty_cache()
