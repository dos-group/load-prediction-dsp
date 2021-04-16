import torch
# keep this import
from torch.nn import *


class SMAPELoss(object):
    def __call__(self, y_pred, y):
        y_true = y.view_as(y_pred)

        return torch.mean(torch.abs(y_true - y_pred) / (torch.abs(y_true) + torch.abs(y_pred)))

    def __repr__(self):
        return self.__class__.__name__

    def __str__(self):
        return self.__class__.__name__
