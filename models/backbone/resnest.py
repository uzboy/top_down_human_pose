import torch
import torch.nn as nn
from models.base_module import NetBase
from utils.utils import kaiming_init, constant_init
from models.blocks.split_atten import SplitAttentionConv2d


class ResNestBottleneck(nn.Module):

    expansion = 4

    def __init__(self, in_channels, out_channels, groups=1, width_per_group=4, base_channels=64, radix=2, reduction_factor=4, stride=1, 
                             avg_down_stride=True, downsample=None):

        super(ResNestBottleneck, self).__init__()

        self.downsample = downsample
        groups = groups
        width_per_group = width_per_group
        mid_channels = out_channels // self.expansion
        if groups != 1:
            mid_channels = (groups * width_per_group * mid_channels // base_channels)
    
        self.avg_down_stride = avg_down_stride and stride > 1

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1, bias=False),
                                                                nn.BatchNorm2d(mid_channels),
                                                                nn.ReLU())
        self.conv2 = SplitAttentionConv2d(in_channels=mid_channels,
                                                                               channels=mid_channels,
                                                                               kernel_size=3,
                                                                               stride=1 if self.avg_down_stride else stride,
                                                                               padding=1,
                                                                               groups=groups,
                                                                               radix=radix,
                                                                               reduction_factor=reduction_factor)
        if self.avg_down_stride:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)

        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False),
                                                                nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        if self.avg_down_stride:
            out = self.avd_layer(out)
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        out = self.relu(out)

        return out


class ResNeSt(NetBase):

    def __init__(self, cfg):
        super(ResNeSt, self).__init__()
        self.out_indices = cfg.out_indices
        self.deep_stem = cfg.deep_stem
        self._make_stem_layer(cfg.in_channels, cfg.stem_channels)
        self.res_layers = nn.ModuleList()
        _in_channels = cfg.stem_channels
        _out_channels = cfg.base_channels * ResNestBottleneck.expansion
        for i, num_blocks in enumerate(cfg.stage_blocks):
            stride = cfg.strides[i]
            res_layer = self.make_res_layer(num_blocks=num_blocks, in_channels=_in_channels, out_channels=_out_channels, stride=stride, 
                                                                              avg_down=cfg.avg_down, groups=cfg.groups, width_per_group=cfg.width_per_group, base_channels=cfg.base_channels,
                                                                              radix=cfg.radix, reduction_factor=cfg.reduction_factor)
            _in_channels = _out_channels
            _out_channels *= 2
            self.res_layers.append(res_layer)

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (nn.BatchNorm2d)):
                constant_init(m, 1)

    def _make_stem_layer(self, in_channels, stem_channels):
        if self.deep_stem:
            self.stem = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=stem_channels // 2, kernel_size=3, stride=2, padding=1, bias=False),
                                                                nn.BatchNorm2d(stem_channels // 2),
                                                                nn.ReLU(inplace=True),
                                                                nn.Conv2d(in_channels=stem_channels // 2, out_channels=stem_channels // 2, kernel_size=3, padding=1, stride=1, bias=False),
                                                                nn.BatchNorm2d(stem_channels // 2),
                                                                nn.ReLU(inplace=True),
                                                                nn.Conv2d(in_channels=stem_channels // 2, out_channels=stem_channels, kernel_size=3, stride=1, padding=1, bias=False),
                                                                nn.BatchNorm2d(stem_channels),
                                                                nn.ReLU(inplace=True),
                                                                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        else:
            self.stem = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=stem_channels, kernel_size=7, stride=2, padding=3, bias=False),
                                                                nn.BatchNorm2d(stem_channels),
                                                                nn.ReLU(),
                                                                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def  make_res_layer(self, in_channels, out_channels, stride, num_blocks, avg_down, groups, width_per_group, base_channels, radix, reduction_factor):
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

        layers.append(ResNestBottleneck(in_channels=in_channels, out_channels=out_channels, stride=stride, downsample=downsample,
                                                                                groups=groups, width_per_group=width_per_group, base_channels=base_channels, radix=radix,
                                                                                reduction_factor=reduction_factor))
        in_channels = out_channels

        for _ in range(1, num_blocks):
            layers.append(ResNestBottleneck(in_channels=in_channels, out_channels=out_channels, stride=1,
                                                                                    groups=groups, width_per_group=width_per_group, base_channels=base_channels,
                                                                                    radix=radix, reduction_factor=reduction_factor))

        return nn.Sequential(*layers)

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


def get_resnest_50(out_indices=[3]):
    """
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3)),
        200: (Bottleneck, (3, 24, 36, 3)),
        269: (Bottleneck, (3, 30, 48, 8))
    """
    return ResNeSt(stage_blocks=[3, 4, 6, 3], out_indices=out_indices, deep_stem=True, avg_down=True)
