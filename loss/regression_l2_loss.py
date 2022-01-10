import torch.nn as nn
from functools import partial
import torch.nn.functional as F


class MSELoss(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.criterion = partial(F.mse_loss, reduction="none")
        self.use_target_weight = cfg.get("use_target_weight", False)
        self.loss_weight = cfg.get("loss_weight", 1.0)

    def forward(self, output, target, target_weight=None):
        if self.use_target_weight:
            loss = self.criterion(output * target_weight, target * target_weight)
            loss = loss.sum() / target_weight.sum()
        else:
            loss = self.criterion(output, target)
            loss = loss.mean()

        return loss * self.loss_weight
