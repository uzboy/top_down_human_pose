# ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks
import math
from torch import nn
import torch.nn.functional as F


class EcaModule(nn.Module):

    def __init__(self, channels):
        super(EcaModule, self).__init__()
        t = int(abs(math.log(channels, 2) + 1) / 2)
        k_size = max(t if t % 2 else t + 1, 3)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class CecaModule(nn.Module):

    def __init__(self, channels):
        super(CecaModule, self).__init__()
        t = int(abs(math.log(channels, 2) + 1) / 2)
        k_size = max(t if t % 2 else t + 1, 3)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.padding = k_size // 2
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=0, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        y = F.pad(y, (self.padding, self.padding), mode='circular')
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)
