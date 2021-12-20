import torch.nn as nn


class CombinedTargetMSELoss(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        try:
            self.use_target_weight = cfg.use_target_weight
        except:
            self.use_target_weight = False
        try:
            self.loss_weight = cfg.loss_weight
        except:
            self.loss_weight = 1.0

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_channels = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_channels, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_channels, -1)).split(1, 1)
        loss = 0.
        num_joints = num_channels // 3

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx * 3].squeeze()
            heatmap_gt = heatmaps_gt[idx * 3].squeeze()
            offset_x_pred = heatmaps_pred[idx * 3 + 1].squeeze()
            offset_x_gt = heatmaps_gt[idx * 3 + 1].squeeze()
            offset_y_pred = heatmaps_pred[idx * 3 + 2].squeeze()
            offset_y_gt = heatmaps_gt[idx * 3 + 2].squeeze()

            if self.use_target_weight:
                heatmap_pred = heatmap_pred * target_weight[:, idx]
                heatmap_gt = heatmap_gt * target_weight[:, idx]

            loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)
            loss += 0.5 * self.criterion(heatmap_gt * offset_x_pred, heatmap_gt * offset_x_gt)
            loss += 0.5 * self.criterion(heatmap_gt * offset_y_pred, heatmap_gt * offset_y_gt)

        return loss / num_joints * self.loss_weight
