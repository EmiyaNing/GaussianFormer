"""One original occupancy objective per decoder layer, summed at full weight."""
from . import OPENOCC_LOSS
from .occupancy_loss import OccupancyLoss


@OPENOCC_LOSS.register_module()
class AllLayerOccupancyLoss(OccupancyLoss):
    def __init__(self, stage_weights=(1., 1., 1., 1., 1., 1.), **kwargs):
        super().__init__(**kwargs)
        self.stage_weights = tuple(float(x) for x in stage_weights)

    def forward(self, inputs):
        arguments = {key: inputs[value] for key, value in self.input_dict.items()}
        predictions = arguments.pop('pred_occ')
        if len(predictions) != len(self.stage_weights):
            raise ValueError('Occupancy stage weights must match rendered decoder layers')
        values = [super(AllLayerOccupancyLoss, self).loss_voxel(
            pred_occ=[prediction], **arguments) for prediction in predictions]
        total = self.weight * sum(w * v for w, v in zip(self.stage_weights, values))
        metrics = {f'layer_{i+1}': v.detach() for i, v in enumerate(values)}
        return total, metrics
