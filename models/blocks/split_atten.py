# ResNeSt: Split-Attention Networks
import torch
import torch.nn.functional as F
from torch import nn


class RSoftmax(nn.Module):

    def __init__(self, radix, groups):
        super().__init__()
        self.radix = radix
        self.groups = groups

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.groups, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)

        return x


class SplitAttentionConv2d(nn.Module):

    def __init__(self, in_channels, channels, kernel_size, stride=1, padding=0, groups=1, radix=2, reduction_factor=4):

        super().__init__()
        inter_channels = max(in_channels * radix // reduction_factor, 32)
        self.radix = radix
        self.groups = groups
        self.channels = channels

        self.split_conv = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                                                                out_channels=channels * radix,
                                                                                                kernel_size=kernel_size,
                                                                                                stride=stride,
                                                                                                padding=padding,
                                                                                                groups=groups * radix,
                                                                                                bias=False),
                                                                        nn.BatchNorm2d(channels * radix),
                                                                        nn.ReLU(inplace=True))

        self.fc = nn.Sequential(nn.Conv2d(in_channels=channels, out_channels=inter_channels, kernel_size=1, groups=groups, bias=False),
                                                        nn.BatchNorm2d(inter_channels),
                                                        nn.Conv2d(in_channels=inter_channels, out_channels=channels * radix, kernel_size=1, groups=groups, bias=True))

        self.softmax = RSoftmax(groups=groups, radix=radix)

    def forward(self, x):
        x = self.split_conv(x)
        batch, rchannel = x.shape[:2]
        if self.radix > 1:
            splited = torch.split(x, int(rchannel // self.radix), dim=1)
            gap = sum(splited)
        else:
            gap = x

        gap = F.adaptive_avg_pool2d(gap, 1)

        gap = self.fc(gap)
        atten = self.softmax(gap).view(batch, -1, 1, 1)

        if self.radix > 1:
            attens = torch.split(atten, int(rchannel // self.radix), dim=1)
            out = sum([att * split for (att, split) in zip(attens, splited)])
        else:
            out = atten * x
    
        return out.contiguous()
