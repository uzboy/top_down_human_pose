import torch.nn as nn


class JointsMSELoss(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.criterion = nn.MSELoss()
        try:
            self.use_target_weight = cfg.use_target_weight
        except:
            self.use_target_weight = False

        try:
            self.loss_weight = cfg.loss_weight
        except:
            self.loss_weight = 1.

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
                loss += self.criterion(heatmap_pred * target_weight[:, idx], heatmap_gt * target_weight[:, idx])
            else:
                loss += self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints * self.loss_weight
