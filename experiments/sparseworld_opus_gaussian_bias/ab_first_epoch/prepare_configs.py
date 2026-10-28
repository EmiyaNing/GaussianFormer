"""Generate literal standalone configs and audit each single-factor change."""
from pathlib import Path

from mmengine.config import Config

from ..train import atomic_json, digest
from .protocol import ARMS, ANALYSIS, BASE_CONFIG


def main():
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    source = Config.fromfile(str(BASE_CONFIG))
    audits = {}
    for arm, path in ARMS.items():
        if path.exists():
            raise ValueError(f'refusing to regenerate an existing experiment config: {path}')
        cfg = Config(source.to_dict())
        cfg.experiment_name = f'sparseworld_opus_gaussian_bias_{arm}'
        cfg.report_directory = str(ANALYSIS / arm)
        cfg.custom_imports.imports.append('experiments.sparseworld_opus_gaussian_bias.ab_first_epoch')
        if arm == 'A_softmax_focal':
            cfg.loss.occupancy_loss.type = 'FullspaceSoftmaxFocalOccupancyLoss'
            cfg.loss.occupancy_loss.focal_gamma = 2.
            change = 'voxel weighted CE -> weighted softmax focal gamma=2; same fixed label-weight normalization'
        else:
            cfg.loss.occupancy_loss.manual_class_weight[-1] = .1
            change = 'voxel CE free class relative weight: 0.5 -> 0.1'
        cfg.dump(str(path))
        if '_base_' in path.read_text():
            raise ValueError('runtime config inheritance is forbidden')
        audits[arm] = dict(config=str(path), config_sha256=digest(path),
                          only_training_change=change, max_epochs=15,
                          total_optimizer_updates=15360, first_epoch_updates=1024,
                          supervision_mask='none', free_relative_weight=cfg.loss.occupancy_loss.manual_class_weight[-1],
                          focal_gamma=cfg.loss.occupancy_loss.get('focal_gamma', 0.),
                          diagnostics_epoch=1, report_directory=cfg.report_directory)
        Path(cfg.report_directory).mkdir(parents=True, exist_ok=True)
    atomic_json(ANALYSIS / 'configuration_audit.json', dict(
        source_config=str(BASE_CONFIG), source_sha256=digest(BASE_CONFIG), arms=audits,
        initialization='from scratch world model; shared existing image backbone/FPN initialization',
        seed=17, train_samples=4096, validation_samples=1024,
        early_stop='runtime boundary after first complete evaluation; config and 15-epoch schedule unchanged',
        B_choice='lower free weight only; no group normalization and no focal change'))
    print(f'Independent configs written: {list(map(str, ARMS.values()))}', flush=True)


if __name__ == '__main__':
    main()
