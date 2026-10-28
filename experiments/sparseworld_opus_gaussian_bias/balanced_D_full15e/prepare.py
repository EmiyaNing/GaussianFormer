"""Copy D's boundary into a new run; only change the output configuration hash."""
import hashlib
import json
import shutil

import torch
from mmengine.config import Config

from ..train import atomic_json, atomic_save, digest
from .protocol import ANALYSIS, CONFIG, ROOT, SOURCE_ANALYSIS, SOURCE_CONFIG, SOURCE_WORK, STRICT, WORK


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal_tree(left, right):
    if torch.is_tensor(left):
        return torch.is_tensor(right) and left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if hasattr(left, 'dtype') and hasattr(left, 'shape'):
        import numpy as np
        return isinstance(right, np.ndarray) and left.dtype == right.dtype and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(equal_tree(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(equal_tree(a, b) for a, b in zip(left, right))
    return left == right


def migrate(source, target, old_hash, new_hash):
    saved = torch.load(source, map_location='cpu', weights_only=False)
    require(saved['config_sha256'] == old_hash, 'source checkpoint config hash differs')
    saved['config_sha256'] = new_hash
    atomic_save(target, saved)
    recovered = torch.load(target, map_location='cpu', weights_only=False)
    require(equal_tree(saved, recovered), 'checkpoint values changed during output-only migration')
    return saved


def main():
    require(not (ANALYSIS / 'preparation.json').exists(), 'continuation already prepared')
    require(not WORK.exists(), 'continuation requires a new work directory')
    source_cfg, cfg = Config.fromfile(str(SOURCE_CONFIG)), Config.fromfile(str(CONFIG))
    old_values, new_values = source_cfg.to_dict(), cfg.to_dict()
    require(new_values.pop('report_directory') == str(ANALYSIS), 'new output directory differs')
    old_values.pop('report_directory')
    require(old_values == new_values, 'continuation must not alter any training configuration')
    require('_base_' not in CONFIG.read_text() and 'Config.fromfile' not in CONFIG.read_text(), 'config must be a static standalone file')
    boundary = read(SOURCE_WORK / 'epoch_01_boundary.json')
    original = SOURCE_WORK / 'epoch_01.pth'
    require(digest(original) == boundary['checkpoint_sha256'], 'D epoch-one checkpoint changed')
    require(digest(SOURCE_CONFIG) == boundary['config_sha256'], 'D original config changed')
    launch = read(SOURCE_ANALYSIS / 'launch.json')
    for path, expected in launch['source_sha256'].items():
        require(digest(path) == expected, f'validated source changed: {path}')
    require(read(SOURCE_ANALYSIS / 'final_verification.json')['status'] == 'passed', 'first-epoch audit failed')
    require(read(STRICT / 'completed.json')['epochs'] == 15, 'completed Strict baseline required')
    ROOT.mkdir(parents=True, exist_ok=True)
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    WORK.mkdir()
    copied = []
    for name in ('epoch_01_metrics.json', 'epoch_01_confusion.npz', 'epoch_01_scene_confusion.npz',
                 'epoch_01_boundary.json', 'first_epoch_completed.json', 'initialization.json', 'best.json'):
        shutil.copyfile(SOURCE_WORK / name, WORK / name)
        copied.append(WORK / name)
    diagnostic_source = SOURCE_ANALYSIS / 'D_balanced_conditional_ce/epoch_diagnostics'
    shutil.copytree(diagnostic_source, ANALYSIS / 'epoch_diagnostics')
    old_hash, new_hash = digest(SOURCE_CONFIG), digest(CONFIG)
    saved = migrate(original, WORK / 'epoch_01.pth', old_hash, new_hash)
    require((saved['epoch'], saved['next_iter'], saved['phase'], saved['updates']) == (1, 0, 'train', 1024), 'expected a complete first-epoch boundary')
    require(saved['optimizer']['state'] and all(k in saved['rng'] for k in ('python', 'numpy', 'torch', 'cuda')), 'optimizer/RNG state incomplete')
    require(saved['manifest_sha256'] == boundary['manifest_sha256'], 'checkpoint subset differs')
    shutil.copyfile(WORK / 'epoch_01.pth', WORK / 'latest.pth')
    migrate(SOURCE_WORK / 'best_mIoU.pth', WORK / 'best_mIoU.pth', old_hash, new_hash)
    (ROOT / 'sparseworld_strict').symlink_to(STRICT, target_is_directory=True)
    preserved = {
        str(SOURCE_CONFIG): old_hash, str(original): boundary['checkpoint_sha256'],
        str(SOURCE_WORK / 'best_mIoU.pth'): digest(SOURCE_WORK / 'best_mIoU.pth'),
        str(SOURCE_ANALYSIS / 'status.json'): digest(SOURCE_ANALYSIS / 'status.json'),
        str(SOURCE_ANALYSIS / 'D_balanced_conditional_ce/status.json'): digest(SOURCE_ANALYSIS / 'D_balanced_conditional_ce/status.json'),
        str(STRICT / 'best_mIoU.pth'): digest(STRICT / 'best_mIoU.pth'),
        str(STRICT / 'completed.json'): digest(STRICT / 'completed.json'),
    }
    preserved.update(launch['source_sha256'])
    atomic_json(ANALYSIS / 'preparation.json', dict(
        status='passed', source_checkpoint=str(original), source_checkpoint_sha256=boundary['checkpoint_sha256'],
        source_config=str(SOURCE_CONFIG), config=str(CONFIG), source_config_sha256=old_hash, config_sha256=new_hash,
        configuration_changes=['report_directory'], checkpoint_metadata_changes=['config_sha256'],
        all_checkpoint_values_reverified=True, model_optimizer_rng_query_stamps_preserved=True,
        start_epoch=2, target_total_epochs=15, updates_at_resume=1024, planned_total_updates=15360,
        manifest_sha256=boundary['manifest_sha256'], fullspace=True, per_epoch_diagnostics=True,
        original_first_epoch_diagnostics_copied=True, original_results_untouched=True,
        preserved_source_sha256=preserved, source_files_copied=[str(path) for path in copied],
        epoch_boundary_semantics='At a completed-epoch boundary, initialize the next epoch normally; restore within-epoch query statistics only for an interrupted epoch.'))
    atomic_json(ANALYSIS / 'status.json', dict(status='prepared', completed_training_epochs=1, planned_epochs=15, next_epoch=2))
    print(f'D continuation prepared: {WORK}', flush=True)


if __name__ == '__main__':
    main()
