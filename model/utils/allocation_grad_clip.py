"""Keep a detached allocation scorer out of the task gradient clip norm."""

import torch


def clip_model_grad_norm_(
    model, max_norm, separate_allocation_score=False, error_if_nonfinite=False
):
    if not separate_allocation_score:
        return torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm, error_if_nonfinite=error_if_nonfinite
        )
    score_parameters = []
    for module in model.modules():
        if hasattr(module, "allocation_score_parameters"):
            score_parameters.extend(module.allocation_score_parameters())
    score_ids = {id(parameter) for parameter in score_parameters}
    task_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in score_ids
    ]
    task_norm = torch.nn.utils.clip_grad_norm_(
        task_parameters, max_norm, error_if_nonfinite=error_if_nonfinite
    )
    if score_parameters:
        torch.nn.utils.clip_grad_norm_(
            score_parameters, max_norm, error_if_nonfinite=error_if_nonfinite
        )
    return task_norm
