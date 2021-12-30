import torch
import torch.nn as nn


class JointsMSEWithWeightLoss(nn.Module):

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

    def _compute_loss(self, pred, target):
        pos_value = (target != 0).sum().float()
        neg_value = (target == 0).sum().float()
        pos_weight = 1 - pos_value / (pos_value + neg_value)

        pos_weight = (pos_value + neg_value) / (pos_value + 1e-6)
        neg_weight = (pos_value + neg_value) / (neg_value + 1e-6)
        loss_weight = torch.ones_like(target)
        loss_weight = torch.where(target != 0, loss_weight * pos_weight, loss_weight * neg_weight)
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


if __name__ == "__main__":
    loss = JointsMSEWithWeightLoss(None)
    output = torch.rand([32, 17, 128, 128])
    target = torch.rand([32, 17, 128, 128])
    loss(output, target, None)