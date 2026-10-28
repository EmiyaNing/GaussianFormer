"""Write a literal standalone snapshot and prove the one-factor B->C change."""
from mmengine.config import Config

from ..train import atomic_json, digest
from .protocol import ANALYSIS, BASE_ANALYSIS, BASE_CONFIG, CONFIG, NAME


def main():
    if CONFIG.exists():
        raise ValueError('refusing to overwrite an existing validated C configuration')
    audit = Config.fromfile(str(BASE_CONFIG))
    import json
    baseline = json.loads((BASE_ANALYSIS / 'configuration_audit.json').read_text())
    if digest(BASE_CONFIG) != baseline['arms']['B_ce_free_0_1']['config_sha256']:
        raise ValueError('B configuration changed after its experiment')
    cfg = Config(audit.to_dict())
    cfg.experiment_name = f'sparseworld_opus_gaussian_bias_{NAME}'
    cfg.report_directory = str(ANALYSIS)
    cfg.custom_imports.imports.append('experiments.sparseworld_opus_gaussian_bias.conditional_first_epoch')
    cfg.loss.occupancy_loss.type = 'FullspaceConditionalSemanticOccupancyLoss'
    cfg.loss.occupancy_loss.conditional_semantic_weight = 1.
    expected = cfg.to_dict()
    for key in ('experiment_name', 'report_directory', 'custom_imports'):
        expected[key] = audit.to_dict()[key]
    expected['loss']['occupancy_loss']['type'] = 'OccupancyLoss'
    expected['loss']['occupancy_loss'].pop('conditional_semantic_weight')
    if expected != audit.to_dict():
        raise ValueError('C changed an additional B training setting')
    cfg.dump(str(CONFIG))
    if '_base_' in CONFIG.read_text():
        raise ValueError('runtime config inheritance is forbidden')
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    atomic_json(ANALYSIS / 'configuration_audit.json', dict(
        config=str(CONFIG), config_sha256=digest(CONFIG), source_config=str(BASE_CONFIG), source_sha256=digest(BASE_CONFIG),
        training_change='Add weighted 17-class conditional semantic CE on all GT-occupied voxels, lambda=1.0.',
        baseline='B weighted 18-class full-space CE free=0.1 plus original Lovasz/set losses',
        conditional_semantic_weight=1., free_relative_weight=.1,
        max_epochs=15, total_optimizer_updates=15360, first_epoch_updates=1024,
        runtime_stop_after_epoch=1, diagnostic_epoch=1, supervision_mask='none', seed=17,
        train_samples=4096, validation_samples=1024, config_inheritance=False,
        model_initialization='same scratch world initialization as B; same image backbone/FPN checkpoint',
        comparison='first complete epoch only; never presented as a 15-epoch optimum'))
    print(f'Standalone C configuration written: {CONFIG}', flush=True)


if __name__ == '__main__':
    main()
