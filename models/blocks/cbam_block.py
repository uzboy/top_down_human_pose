import torch
from torch import nn as nn


class ChannelAttn(nn.Module):

    def __init__(self, channels, ratio=16):
        super(ChannelAttn, self).__init__()
        rd_channels = channels // ratio
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.atten_path = nn.Sequential(nn.Conv2d(in_channels=channels, out_channels=rd_channels, kernel_size=1, stride=1, padding=0, bias=True),
                                                                          nn.ReLU(inplace=True),
                                                                          nn.Conv2d(in_channels=rd_channels, out_channels=channels, kernel_size=1, stride=1, padding=0, bias=True))
        
        self.gate = nn.Sigmoid()            # 使用sigmoid之后，层与层之间没有关系，使用softmax，确保层与层之间权重和为1

    def forward(self, x):
        x_avg = self.avg_pool(x)
        x_avg = self.atten_path(x_avg)

        x_max = self.max_pool(x)
        x_max = self.atten_path(x_max)

        return x * self.gate(x_avg + x_max)


class LightChannelAttn(ChannelAttn):

    def __init__(self, channels, ratio=16):
        super(LightChannelAttn, self).__init__(channels, ratio)

    def forward(self, x):
        x_pool = 0.5 * x.mean((2, 3), keepdim=True) + 0.5 * x.amax((2, 3), keepdim=True)
        x_attn = self.atten_path(x_pool)
        return x * self.gate(x_attn)


class SpatialAttn(nn.Module):

    def __init__(self, kernel_size=7):
        super(SpatialAttn, self).__init__()
        self.atten_path = nn.Sequential(nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
                                                                          nn.Sigmoid())

    def forward(self, x):
        x_attn = torch.cat([x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1)
        x_attn = self.atten_path(x_attn)
        return x * x_attn


class LightSpatialAttn(nn.Module):

    def __init__(self, kernel_size=7):
        super(LightSpatialAttn, self).__init__()
        self.atten_path = nn.Sequential(nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
                                                                          nn.Sigmoid())

    def forward(self, x):
        x_attn = 0.5 * x.mean(dim=1, keepdim=True) + 0.5 * x.amax(dim=1, keepdim=True)
        x_attn = self.atten_path(x_attn)
        return x * x_attn


class CbamModule(nn.Module):

    def __init__(self, channels, ratio=16, spatial_kernel_size=7):
        super(CbamModule, self).__init__()
        self.channel = ChannelAttn(channels, ratio=ratio)
        self.spatial = SpatialAttn(spatial_kernel_size)

    def forward(self, x):
        x = self.channel(x)
        x = self.spatial(x)
        return x


class LightCbamModule(nn.Module):

    def __init__(self, channels, ratio=16, spatial_kernel_size=7):
        super(LightCbamModule, self).__init__()
        self.channel = LightChannelAttn(channels, ratio=ratio)
        self.spatial = LightSpatialAttn(spatial_kernel_size)

    def forward(self, x):
        x = self.channel(x)
        x = self.spatial(x)
        return x
