import torch
import torch.nn as nn


class JointsMSEWithWeightLoss(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        if hasattr(cfg, "use_target_weight"):
            self.use_target_weight = cfg.use_target_weight
        else:
            self.use_target_weight = False

        if hasattr(cfg, "loss_weight"):
            self.loss_weight = cfg.loss_weight
        else:
            self.loss_weight = 1.

    def _compute_loss(self, pred, target):
        pos_value = (target != 0).sum(dim=1).float()
        pos_weight = 1 - pos_value / target.shape[-1]
        pos_weight = pos_weight[:, None].expand_as(target)
        loss_weight = torch.where(target != 0, pos_weight, 1 - pos_weight)
        loss = (pred - target) ** 2 * loss_weight
        return loss.mean()

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)

        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)            # 按照关键点进行split
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = 0.

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze(1)
            heatmap_gt = heatmaps_gt[idx].squeeze(1)
            if self.use_target_weight:
                loss += self._compute_loss(heatmap_pred * target_weight[:, idx], heatmap_gt * target_weight[:, idx])
            else:
                loss += self._compute_loss(heatmap_pred, heatmap_gt)

        return loss / num_joints * self.loss_weight
