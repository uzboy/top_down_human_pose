import torch
import torch.nn as nn


class RegressionMSELoss(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.criterion = nn.MSELoss(reduction="none")

    def forward(self, output, target, target_weight):
        loss = self.criterion(output, target) * target_weight
        loss = torch.sum(loss) / torch.sum(target_weight)
        return loss
