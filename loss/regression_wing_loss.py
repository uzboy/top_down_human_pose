import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class WingLoss(nn.Module):

    def __init__(self, cfg):
        #omega=10.0, epsilon=2.0, use_target_weight=False, loss_weight=1.
        super().__init__()
        self.omega = cfg.omega
        self.epsilon = cfg.epsilon
        self.use_target_weight = cfg.use_target_weight
        self.loss_weight = cfg.loss_weight
        self.C = self.omega * (1.0 - math.log(1.0 + self.omega / self.epsilon))

    def criterion(self, pred, target):
        delta = (target - pred).abs()
        losses = torch.where(delta < self.omega, self.omega * torch.log(1.0 + delta / self.epsilon), delta - self.C)
        return torch.mean(torch.sum(losses, dim=[1, 2]), dim=0)

    def forward(self, output, target, target_weight=None):
        if self.use_target_weight:
            loss = self.criterion(output * target_weight, target * target_weight)
        else:
            loss = self.criterion(output, target)

        return loss * self.loss_weight
