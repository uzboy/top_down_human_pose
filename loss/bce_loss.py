import torch.nn as nn
import torch.nn.functional as F


class JointsBCELoss(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        try:
            self.use_target_weight = cfg.use_target_weight
        except:
            self.use_target_weight = False

        try:
            self.loss_weight = cfg.loss_weight
        except:
            self.loss_weight = 1.

    def forward(self, output, target, target_weight):
        B, C, _, _ = output.shape
        output = output.reshape(B, C, -1)
        output = F.softmax(output, dim=-1)              # 选择softmax而非sigmoid，主要是保证像素点之间互相存在关联
        target = target.reshape(B, C, -1)
        loss = F.binary_cross_entropy_with_logits(output, target, reduction='none')
        loss = loss.mean(dim=-1, keepdim=True)
        loss = loss * target_weight
        loss = loss.mean()
        return loss * self.loss_weight
