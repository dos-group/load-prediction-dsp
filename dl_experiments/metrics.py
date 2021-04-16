from typing import Sequence, Union

import torch
from ignite.exceptions import NotComputableError
# keep this import
from ignite.metrics import *
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce


class SymmetricMeanAbsolutePercentageError(Metric):
    """
    Calculates symmetric mean absolute percentage error.
    - ``update`` must receive output of the form ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
    """

    def __init__(self):
        super().__init__()
        self._sum_of_errors: float = 0.0
        self._num_examples: int = 0

    @reinit__is_reduced
    def reset(self) -> None:
        self._sum_of_errors = 0.0
        self._num_examples = 0
        super().reset()

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        y_pred, y = output
        y_true = y.view_as(y_pred)

        errors = torch.sum(torch.abs(y_true - y_pred) / (torch.abs(y_true) + torch.abs(y_pred)))

        self._sum_of_errors += errors.item()
        self._num_examples += torch.numel(y_pred)

    @sync_all_reduce("_sum_of_errors", "_num_examples")
    def compute(self) -> Union[float, torch.Tensor]:
        if self._num_examples == 0:
            raise NotComputableError(
                "SymmetricMeanAbsolutePercentageError must have at least one example before it can be computed.")
        return self._sum_of_errors / self._num_examples

