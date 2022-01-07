import torch
import torch.nn as nn


class JointsOHKMMSELoss(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.criterion = nn.MSELoss(reduction='none')
        if hasattr(cfg, "use_target_weight"):
            self.use_target_weight = cfg.use_target_weight
        else:
            self.use_target_weight = False

        if hasattr(cfg, "topk"):
            self.topk = cfg.topk
        else:
            self.topk = 8

        if hasattr(cfg, "loss_weight"):
            self.loss_weight = cfg.loss_weight
        else:
            self.loss_weight = 1.0

    def _ohkm(self, loss):
        ohkm_loss = 0.
        N = len(loss)
        for i in range(N):
            sub_loss = loss[i]
            _, topk_idx = torch.topk(sub_loss, k=self.topk, dim=0, sorted=False)
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / self.topk
        ohkm_loss /= N
        return ohkm_loss

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        if num_joints < self.topk:
            raise ValueError(f'topk ({self.topk}) should not larger than num_joints ({num_joints}).')

        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        losses = []
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze(1)
            heatmap_gt = heatmaps_gt[idx].squeeze(1)
            if self.use_target_weight:
                losses.append(self.criterion(heatmap_pred * target_weight[:, idx], heatmap_gt * target_weight[:, idx]))
            else:
                losses.append(self.criterion(heatmap_pred, heatmap_gt))

        losses = [loss.mean(dim=1).unsqueeze(dim=1) for loss in losses]
        losses = torch.cat(losses, dim=1)

        return self._ohkm(losses) * self.loss_weight
