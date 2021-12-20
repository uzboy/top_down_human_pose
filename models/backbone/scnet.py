import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_module import NetBase
from utils.utils import kaiming_init, constant_init


class SCConv(nn.Module):

    def __init__(self, in_channels, out_channels, stride, pooling_r):
        super().__init__()

        assert in_channels == out_channels

        self.k2 = nn.Sequential(nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
                                                        nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, bias=False),
                                                        nn.BatchNorm2d(in_channels))
    
        self.k3 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, bias=False),
                                                        nn.BatchNorm2d(in_channels))

        self.k4 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                                                        nn.BatchNorm2d(in_channels),
                                                        nn.ReLU(inplace=True))

    def forward(self, x):
        identity = x
        out = torch.sigmoid(torch.add(identity, F.interpolate(self.k2(x), identity.size()[2:])))
        out = torch.mul(self.k3(x), out)
        out = self.k4(out)

        return out


class SCBottleneck(nn.Module):

    expansion = 4
    pooling_r = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(SCBottleneck, self).__init__()
        self.downsample = downsample
        mid_channels = out_channels // self.expansion // 2

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
                                                                nn.BatchNorm2d(mid_channels),
                                                                nn.ReLU(inplace=True))
        self.k1 = nn.Sequential(nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                                                        nn.BatchNorm2d(mid_channels),
                                                        nn.ReLU(inplace=True))


        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1, bias=False),
                                                                nn.BatchNorm2d(mid_channels),
                                                                nn.ReLU(inplace=True))
        self.scconv = SCConv(mid_channels, mid_channels, stride, self.pooling_r)

        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=mid_channels * 2, out_channels=out_channels, kernel_size=1, stride=1, bias=False),
                                                                nn.BatchNorm2d(out_channels))
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out_a = self.conv1(x)
        out_a = self.k1(out_a)

        out_b = self.conv2(x)
        out_b = self.scconv(out_b)

        out = self.conv3(torch.cat([out_a, out_b], dim=1))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        return self.relu(out)


class SCNet(NetBase):

    def __init__(self, cfg):
        super(SCNet, self).__init__()
        self.deep_stem = cfg.deep_stem
        self.out_indices = cfg.out_indices
        self._make_stem_layer(cfg.in_channels, cfg.stem_channels)

        self.res_layers = []
        _in_channels = cfg.stem_channels
        _out_channels = cfg.base_channels * SCBottleneck.expansion
        for i, num_blocks in enumerate(cfg.stage_blocks):
            stride = cfg.strides[i]
            res_layer = self.make_res_layer(num_blocks=num_blocks, in_channels=_in_channels,
                                                                              out_channels=_out_channels, stride=stride, avg_down=cfg.avg_down)
            _in_channels = _out_channels
            _out_channels *= 2
            self.res_layers.append(res_layer)

        self.res_layers = nn.ModuleList(self.res_layers)

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (nn.BatchNorm2d)):
                constant_init(m, 1)

    def make_res_layer(self, in_channels, out_channels, stride, num_blocks, avg_down):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = []
            conv_stride = stride
            if avg_down and stride != 1:
                conv_stride = 1
                downsample.append(nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False))
            downsample.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=conv_stride, bias=False))
            downsample.append(nn.BatchNorm2d(out_channels))
            downsample = nn.Sequential(*downsample)

        layers = []

        layers.append(SCBottleneck(in_channels=in_channels, out_channels=out_channels, stride=stride, downsample=downsample))
        in_channels = out_channels

        for _ in range(1, num_blocks):
            layers.append(SCBottleneck(in_channels=in_channels, out_channels=out_channels, stride=1))

        return nn.Sequential(*layers)

    def _make_stem_layer(self, in_channels, stem_channels):
        if self.deep_stem:
            self.stem = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                                                          out_channels=stem_channels // 2,
                                                                                          kernel_size=3,
                                                                                          stride=2,
                                                                                          padding=1,
                                                                                          bias=False),
                                                                  nn.BatchNorm2d(stem_channels // 2),
                                                                  nn.ReLU(inplace=True),
                                                                  nn.Conv2d(in_channels=stem_channels // 2,
                                                                                          out_channels=stem_channels // 2,
                                                                                          kernel_size=3,
                                                                                          stride=1,
                                                                                          padding=1,
                                                                                          bias=False),
                                                                  nn.BatchNorm2d(stem_channels // 2),
                                                                  nn.ReLU(inplace=True),
                                                                  nn.Conv2d(in_channels=stem_channels // 2,
                                                                                          out_channels=stem_channels,
                                                                                          kernel_size=3,
                                                                                          stride=1,
                                                                                          padding=1,
                                                                                          bias=False),
                                                                  nn.BatchNorm2d(stem_channels),
                                                                  nn.ReLU(inplace=True),
                                                                  nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        else:
            self.stem = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                                                          out_channels=stem_channels,
                                                                                          kernel_size=7,
                                                                                          stride=2,
                                                                                          padding=3,
                                                                                          bias=False),
                                                                  nn.BatchNorm2d(stem_channels),
                                                                  nn.ReLU(inplace=True),
                                                                  nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def forward(self, x):
        x = self.stem(x)
        outs = []

        for i, res_layer in enumerate(self.res_layers):
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)

        if len(outs) == 1:
            return outs[0]

        return tuple(outs)

"""
50: (SCBottleneck, [3, 4, 6, 3]),
101: (SCBottleneck, [3, 4, 23, 3])
"""
