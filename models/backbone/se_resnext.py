import torch.nn as nn
from models.base_module import NetBase
from utils.utils import kaiming_init, constant_init


class SELayer(nn.Module):

    def __init__(self, channels, ratio=16):
        super(SELayer, self).__init__()
    
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.reduce_conv = nn.Sequential(nn.Conv2d(in_channels=channels, out_channels=channels // ratio, kernel_size=1, stride=1, padding=0),
                                                                              nn.ReLU(inplace=True))
        self.expend_conv = nn.Sequential(nn.Conv2d(in_channels=channels // ratio, out_channels=channels, kernel_size=1, stride=1, padding=0),
                                                                               nn.Sigmoid())

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.reduce_conv(out)
        out = self.expend_conv(out)
        return x * out


class ResNextBottleneck(nn.Module):

    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, base_channels=64, groups=32, width_per_group=4, se_ratio=16, downsample=None):
        super(ResNextBottleneck, self).__init__()
        assert out_channels % self.expansion == 0
        self.downsample = downsample

        self.se_layer = SELayer(out_channels, ratio=se_ratio)
        
        mid_channels = out_channels // self.expansion
        if groups != 1:
            mid_channels = (groups * width_per_group * mid_channels // base_channels)

        self.res_conv = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1, bias=False),
                                                                        nn.BatchNorm2d(mid_channels),
                                                                        nn.ReLU(inplace=True),
                                                                        nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3,
                                                                                                groups=groups, stride=stride, padding=1, bias=False),
                                                                        nn.BatchNorm2d(mid_channels),
                                                                        nn.ReLU(inplace=True),
                                                                        nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False),
                                                                        nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res_out = self.res_conv(x)
        res_out = self.se_layer(res_out)
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        res_out += identity
        return self.relu(res_out)


class ResNeXt_SE(NetBase):

    def __init__(self, cfg):

        super(ResNeXt_SE, self).__init__()
        self.deep_stem = cfg.deep_stem
        self.out_indices = cfg.out_indices
        self._make_stem_layer(cfg.in_channels, cfg.stem_channels)

        self.res_layers = nn.ModuleList()
        _in_channels = cfg.stem_channels
        _out_channels = cfg.base_channels * ResNextBottleneck.expansion
        for i, num_blocks in enumerate(cfg.stage_blocks):
            stride = cfg.strides[i]
            res_layer = self.make_res_layer(num_blocks=num_blocks, in_channels=_in_channels, out_channels=_out_channels, stride=stride, 
                                                                              avg_down=cfg.avg_down, groups=cfg.groups, width_per_group=cfg.width_per_group,
                                                                              base_channels=cfg.base_channels, se_ratio=cfg.se_ratio)
            _in_channels = _out_channels
            _out_channels *= 2
            self.res_layers.append(res_layer)
    
    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (nn.BatchNorm2d)):
                constant_init(m, 1)

    def make_res_layer(self, in_channels, out_channels, stride, num_blocks, avg_down, groups, width_per_group, base_channels, se_ratio):
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

        layers.append(ResNextBottleneck(in_channels=in_channels, out_channels=out_channels, stride=stride, downsample=downsample,
                                                                                base_channels=base_channels, groups=groups, width_per_group=width_per_group, se_ratio=se_ratio))
        in_channels = out_channels

        for _ in range(1, num_blocks):
            layers.append(ResNextBottleneck(in_channels=in_channels, out_channels=out_channels, stride=1,
                                                                                    base_channels=base_channels, groups=groups, width_per_group=width_per_group, se_ratio=se_ratio))

        return nn.Sequential(*layers)

    def _make_stem_layer(self, in_channels, stem_channels):
        if self.deep_stem:
            self.stem = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=stem_channels // 2, kernel_size=3, stride=2, padding=1, bias=False),
                                                                  nn.BatchNorm2d(stem_channels // 2),
                                                                  nn.ReLU(),
                                                                  nn.Conv2d(in_channels=stem_channels // 2, out_channels=stem_channels // 2, kernel_size=3, stride=1, padding=0, bias=False),
                                                                  nn.BatchNorm2d(stem_channels // 2),
                                                                  nn.ReLU(),
                                                                  nn.Conv2d(in_channels=stem_channels // 2, out_channels=stem_channels, kernel_size=3, stride=1, padding=1, bias=False),
                                                                  nn.BatchNorm2d(stem_channels),
                                                                  nn.ReLU(),
                                                                  nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        else:
            self.stem = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=stem_channels, kernel_size=7, stride=2, padding=3, bias=False),
                                                                  nn.BatchNorm2d(stem_channels),
                                                                  nn.ReLU(),
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


def get_se_resnet_50(out_indices=(3, )):
    """
        50: (SEBottleneck, (3, 4, 6, 3)),
        101: (SEBottleneck, (3, 4, 23, 3)),
        152: (SEBottleneck, (3, 8, 36, 3))
    """
    return ResNeXt_SE(stage_blocks=[3, 4, 6, 3], out_indices=out_indices, deep_stem=True, avg_down=True)
