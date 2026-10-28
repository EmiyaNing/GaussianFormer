"""Reuse the verified 15-epoch scheduler and saved epoch-one pause boundary."""
import sys
import json

from mmengine.config import Config

from ..ab_first_epoch import train as boundary_runner
from .. import train as runner
from .diagnostics import evaluate
from . import loss  # Register the new objective in this opt-in entry point.
from .protocol import BASE_WORK


def main():
    cfg = Config.fromfile(sys.argv[sys.argv.index('--config') + 1])
    if (cfg.loss.occupancy_loss.type != 'FullspaceConditionalSemanticOccupancyLoss' or
            cfg.loss.occupancy_loss.conditional_semantic_weight != 1. or
            cfg.loss.occupancy_loss.manual_class_weight[-1] != .1):
        raise ValueError('C must retain B free=0.1 and add only lambda=1 conditional semantic CE')
    original = boundary_runner.evaluate
    original_initialize = runner.load_image_initialization
    def checked_initialize(model, path):
        audit = original_initialize(model, path)
        expected_fingerprint = json.loads((BASE_WORK / 'initialization.json').read_text())['initial_model_sha256']
        if boundary_runner.model_fingerprint(model) != expected_fingerprint:
            raise ValueError('formal C initialization differs from the preserved B initialization')
        return audit

    boundary_runner.evaluate = evaluate
    runner.load_image_initialization = checked_initialize
    try:
        boundary_runner.main()
    finally:
        boundary_runner.evaluate = original
        runner.load_image_initialization = original_initialize


if __name__ == '__main__':
    main()
