import torch.nn as nn
import torch.nn.functional as F


class MSELoss(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.criterion = F.mse_loss
        self.use_target_weight = cfg.get("use_target_weight", False)
        self.loss_weight = cfg.get("loss_weight", 1.0)

    def forward(self, output, target, target_weight=None):
        if self.use_target_weight:
            loss = self.criterion(output * target_weight, target * target_weight)
        else:
            loss = self.criterion(output, target)

        return loss * self.loss_weight
