import sys

from mmengine.config import Config

from .. import train as runner
from ..ab_first_epoch import train as boundary_runner
from ..ab_first_epoch.report import read
from .diagnostics import evaluate
from .protocol import C_WORK, D, E, NEW_BRANCH_PREFIX


def initial_fingerprint(model):
    if not any(name.startswith(NEW_BRANCH_PREFIX) for name in model.state_dict()):
        return boundary_runner.model_fingerprint(model)
    import hashlib
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if name.startswith(NEW_BRANCH_PREFIX):
            continue
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    cfg = Config.fromfile(sys.argv[sys.argv.index('--config') + 1])
    if (cfg.route_arm not in (D, E) or cfg.loss.occupancy_loss.manual_class_weight[-1] != .1 or
            cfg.loss.occupancy_loss.conditional_semantic_weight != 1.):
        raise ValueError('D/E retain C free=.1 and lambda=1')
    old_evaluate, old_initialize = boundary_runner.evaluate, runner.load_image_initialization
    expected = read(C_WORK / 'initialization.json')['initial_model_sha256']

    def initialize(model, path):
        audit = old_initialize(model, path)
        shared = initial_fingerprint(model)
        if shared != expected:
            raise ValueError('unchanged D/E model tensors differ from C scratch initialization')
        audit.update(shared_initial_model_sha256=shared, same_unchanged_C_tensors=True,
                     initial_model_sha256=boundary_runner.model_fingerprint(model),
                     new_occupancy_parameters=sum(p.numel() for name, p in model.named_parameters() if name.startswith(NEW_BRANCH_PREFIX)))
        return audit

    boundary_runner.evaluate = evaluate
    runner.load_image_initialization = initialize
    try:
        boundary_runner.main()
    finally:
        boundary_runner.evaluate = old_evaluate
        runner.load_image_initialization = old_initialize


if __name__ == '__main__':
    main()
