#Gather-Excite: Exploiting Feature Context in CNNs
import math
from torch import nn as nn
import torch.nn.functional as F


class GatherExcite(nn.Module):

    def __init__(self, channels, feat_size=None, extra_params=False, extent=0, ratio=16, add_maxpool=False):
        super(GatherExcite, self).__init__()
        self.add_maxpool = add_maxpool
        self.extent = extent

        if extra_params:                                                                        # 控制gather路径，通过卷积获取ge信息，否则直接通过池化获取信息
            self.gather = nn.ModuleList()
            if extent == 0:
                self.gather.append(nn.Conv2d(in_channels=channels,
                                                                                out_channels=channels,
                                                                                kernel_size=feat_size,
                                                                                stride=1,
                                                                                padding=feat_size//2,
                                                                                groups=channels,
                                                                                bias=False))
                self.gather.append(nn.BatchNorm2d(channels))
            else:
                num_conv = int(math.log2(extent))       # 多层卷积下采样，获取到最终的全局视野信息
                for i in range(num_conv):
                    self.gather.append(nn.Conv2d(in_channels=channels,
                                                                                    out_channels=channels,
                                                                                    kernel_size=3,
                                                                                    stride=2,
                                                                                    padding=1,
                                                                                    groups=channels,
                                                                                    bias=False))
                    self.gather.append(nn.BatchNorm2d(channels))
                    if i != num_conv - 1:
                        self.gather.append(nn.ReLU(inplace=True))
            self.gather = nn.Sequential(*self.gather)
        else:
            self.gather = None
            if self.extent == 0:
                self.gk = 0
                self.gs = 0
            else:
                self.gk = self.extent * 2 - 1
                self.gs = self.extent

        rd_channels = channels // ratio
        self.mlp = nn.Sequential(nn.Conv2d(in_channels=channels, out_channels=rd_channels, kernel_size=1, stride=1, padding=0, bias=True),
                                                            nn.ReLU(inplace=True),
                                                            nn.Conv2d(in_channels=rd_channels, out_channels=channels, kernel_size=1, stride=1, padding=0, bias=True))
        self.gate = nn.Sigmoid()

    def forward(self, x):
        size = x.shape[-2:]
        if self.gather is not None:
            x_ge = self.gather(x)
        else:
            if self.extent == 0:
                x_ge = x.mean(dim=(2, 3), keepdims=True)
                if self.add_maxpool:
                    x_ge = 0.5 * x_ge + 0.5 * x.amax((2, 3), keepdim=True)
            else:
                x_ge = F.avg_pool2d(x, kernel_size=self.gk, stride=self.gs, padding=self.gk // 2, count_include_pad=False)
                if self.add_maxpool:
                    x_ge = 0.5 * x_ge + 0.5 * F.max_pool2d(x, kernel_size=self.gk, stride=self.gs, padding=self.gk // 2)

        x_ge = self.mlp(x_ge)
        if x_ge.shape[-1] != 1 or x_ge.shape[-2] != 1:
            x_ge = F.interpolate(x_ge, size=size)
        return x * self.gate(x_ge)
