import torch.nn as nn
import torch.nn.functional as F


class JointsBCELoss(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.use_target_weight = cfg.get("use_target_weight", False)
        self.loss_weight = cfg.get("loss_weight", 1.0)

    def forward(self, output, target, target_weight=None):
        B, C, _, _ = output.shape
        output = output.reshape(B, C, -1)
        target = target.reshape(B, C, -1)
        loss = F.binary_cross_entropy(output, target, reduction='none')
        if self.use_target_weight:
            loss = loss.mean(dim=-1, keepdim=True)
            loss = loss * target_weight
            loss = loss.sum() / target_weight.sum()
        else:
            loss = loss.mean()
    
        return loss * self.loss_weight


class VisMaskBCELoss(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.use_target_weight = cfg.get("use_target_weight", False)
        self.loss_weight = cfg.get("loss_weight", 1.0)

    def forward(self, output, target, target_weight=None):
        loss = F.binary_cross_entropy(output, target, reduction='none')
        if self.use_target_weight:
            loss = loss * target_weight
            loss = loss.sum() / target_weight.sum()
        else:
            loss = loss.mean()
    
        return loss * self.loss_weight
