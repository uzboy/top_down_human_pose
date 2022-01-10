import torch
import torch.nn as nn


class JointsMSEWithWeightLoss(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.use_target_weight = cfg.get("use_target_weight", False)
        self.loss_weight = cfg.get("loss_weight", 1.0)

    def forward(self, output, target, target_weight):
        batch_size, num_joints, _, _ = output.shape
        heatmaps_pred = output.reshape((batch_size, num_joints, -1))
        heatmaps_gt = target.reshape((batch_size, num_joints, -1))

        pos_values = (heatmaps_gt != 0).sum(dim=-1).float()
        pos_weight = 1 - pos_values / heatmaps_gt.shape[-1]
        pos_weight = pos_weight.expand_as(heatmaps_gt)
        loss_weight = torch.where(heatmaps_gt != 0, pos_weight, 1 - pos_weight)

        loss = (heatmaps_pred - heatmaps_gt) ** 2 * loss_weight
        if self.use_target_weight:
            loss = loss.mean(dim=-1, keepdim=True)
            loss = loss * target_weight
            loss = loss.sum() / target_weight.sum()
        else:
            loss = loss.mean()

        return loss / self.loss_weight
