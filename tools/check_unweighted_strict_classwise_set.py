"""Check that the only model-loss change is uniform class aggregation."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.unweighted_strict_classwise_set_experiment import RUN, config, write


def main():
    from tools.strict_classwise_set_experiment import config as strict_config
    old, new = strict_config('strict_classwise'), config()
    assert old.model == new.model
    assert old.optimizer == new.optimizer
    assert old.train_dataset_config == new.train_dataset_config
    assert old.val_dataset_config == new.val_dataset_config
    old_losses, new_losses = old.loss.loss_cfgs, new.loss.loss_cfgs
    assert len(old_losses) == len(new_losses) == 2
    assert old_losses[1] == new_losses[1]
    a, b = dict(old_losses[0]), dict(new_losses[0])
    for key in ('class_weights', 'geometry_class_weights'):
        weights = list(b.pop(key))
        assert len(weights) == 17 and weights == [1] * 17
        a.pop(key)
    assert a == b, 'Unexpected Set-loss or model change beyond class weights'
    result = dict(status='passed', changed_fields=['class_weights', 'geometry_class_weights'],
                  class_count=17, weights=[1] * 17,
                  model_optimizer_data_occupancy_and_other_set_settings_identical=True)
    RUN.mkdir(parents=True, exist_ok=True)
    write(RUN / 'numerical_checks.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
