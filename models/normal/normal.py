import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupNorm(nn.GroupNorm):

    def __init__(self, num_channels, num_groups, eps=1e-5, affine=True):
        super().__init__(num_groups, num_channels, eps=eps, affine=affine)

    def forward(self, x):
        return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)


class LayerNorm2d(nn.LayerNorm):

    def __init__(self, num_channels):
        super().__init__(num_channels)

    def forward(self, x):
        return F.layer_norm(x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)
