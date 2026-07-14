import torch.nn as nn
from . import OPENOCC_LOSS
from misc.tb_wrapper import WrappedTBWriter
if 'selfocc' in WrappedTBWriter._instance_dict:
    writer = WrappedTBWriter.get_instance('selfocc')
else:
    writer = None

@OPENOCC_LOSS.register_module()
class MultiLoss(nn.Module):

    def __init__(self, loss_cfgs):
        super().__init__()
        
        assert isinstance(loss_cfgs, list)
        self.num_losses = len(loss_cfgs)
        
        losses = []
        for loss_cfg in loss_cfgs:
            losses.append(OPENOCC_LOSS.build(loss_cfg))
        self.losses = nn.ModuleList(losses)
        self.iter_counter = 0

    def forward(self, inputs):
        
        loss_dict = {}
        tot_loss = 0.
        for loss_func in self.losses:
            loss_output = loss_func(inputs)
            if isinstance(loss_output, tuple):
                loss, metrics = loss_output
            else:
                loss = loss_output
                metrics = {}
            tot_loss += loss
            loss_dict.update({
                loss_func.__class__.__name__: \
                loss.detach().item()
            })
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, nn.Parameter):
                    metric_value = metric_value.detach()
                if hasattr(metric_value, 'item'):
                    metric_value = metric_value.item()
                loss_dict.update({
                    f'{loss_func.__class__.__name__}/{metric_name}': metric_value
                })
            if writer and self.iter_counter % 10 == 0:
                writer.add_scalar(
                    f'loss/{loss_func.__class__.__name__}', 
                    loss.detach().item(), self.iter_counter)
                for metric_name, metric_value in metrics.items():
                    if hasattr(metric_value, 'item'):
                        metric_value = metric_value.item()
                    writer.add_scalar(
                        f'allocation/{metric_name}',
                        metric_value,
                        self.iter_counter)
        if writer and self.iter_counter % 10 == 0:
            writer.add_scalar(
                'loss/total', tot_loss.detach().item(), self.iter_counter)
        self.iter_counter += 1
        
        return tot_loss, loss_dict
