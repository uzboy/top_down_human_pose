import torch
import torch.nn as nn
import torch.nn.functional as F


class L1Loss(nn.Module):

    def __init__(self, cfg):
        # use_target_weight=False, loss_weight=1.
        super().__init__()
        self.criterion = F.l1_loss
        self.use_target_weight = cfg.use_target_weight
        self.loss_weight = cfg.loss_weight

    def forward(self, output, target, target_weight=None):
        if self.use_target_weight:
            loss = self.criterion(output * target_weight, target * target_weight)
        else:
            loss = self.criterion(output, target)

        return loss * self.loss_weight


class SmoothL1Loss(nn.Module):

    def __init__(self, cfg):
        # use_target_weight=False, loss_weight=1.
        super().__init__()
        self.criterion = F.smooth_l1_loss
        self.use_target_weight = cfg.use_target_weight
        self.loss_weight = cfg.loss_weight

    def forward(self, output, target, target_weight=None):
        if self.use_target_weight:
            loss = self.criterion(output * target_weight, target * target_weight)
        else:
            loss = self.criterion(output, target)

        return loss * self.loss_weight


class MPJPELoss(nn.Module):

    def __init__(self, cfg):
        # use_target_weight=False, loss_weight=1.
        super().__init__()
        self.use_target_weight = cfg.use_target_weight
        self.loss_weight = cfg.loss_weight

    def forward(self, output, target, target_weight=None):
        if self.use_target_weight:
            loss = torch.mean(torch.norm((output - target) * target_weight, dim=-1))
        else:
            loss = torch.mean(torch.norm(output - target, dim=-1))

        return loss * self.loss_weight
