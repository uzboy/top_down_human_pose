import torch.nn as nn


class JointsMSELoss(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.criterion = nn.MSELoss()
        self.use_target_weight = cfg.get("use_target_weight", False)
        self.loss_weight = cfg.get("loss_weight", 1.0)

    def forward(self, output, target, target_weight):
        batch_size, num_joints, _, _ = output.shape
        heatmaps_pred = output.reshape((batch_size, num_joints, -1))
        heatmaps_gt = target.reshape((batch_size, num_joints, -1))

        loss = (heatmaps_pred - heatmaps_gt) ** 2
        if self.use_target_weight:
            loss = loss.mean(dim=-1, keepdim=True)
            loss = loss * target_weight
            loss = loss.sum() / target_weight.sum()
        else:
            loss = loss.mean()

        return loss / self.loss_weight
