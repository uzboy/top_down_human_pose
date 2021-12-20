import torch.nn as nn
from models.base_module import NetBase
from utils.utils import kaiming_init, constant_init

class BasicBlock(nn.Module):

    expansion=1

    def __init__(self, in_channels, out_channels, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        mid_channels = out_channels // self.expansion
        self.downsample = downsample
        self.res_conv = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                                                              out_channels=mid_channels,
                                                                                              kernel_size=3,
                                                                                              stride=stride,
                                                                                              padding=dilation,
                                                                                              dilation=dilation,
                                                                                              bias=False),
                                                                      nn.BatchNorm2d(mid_channels),
                                                                      nn.ReLU(inplace=True),
                                                                      nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                                                                      nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res_out = self.res_conv(x)
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        res_out += identity
        return self.relu(res_out)


class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        assert out_channels % self.expansion == 0

        mid_channels = out_channels // self.expansion
        self.downsample = downsample

        self.res_conv = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1, bias=False),
                                                                        nn.BatchNorm2d(mid_channels),
                                                                        nn.ReLU(inplace=True),
                                                                        nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=stride, padding=dilation,
                                                                                                dilation=dilation, bias=False),
                                                                        nn.BatchNorm2d(mid_channels),
                                                                        nn.ReLU(inplace=True),
                                                                        nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False),
                                                                        nn.BatchNorm2d(out_channels))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res_out = self.res_conv(x)
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        res_out += identity
        return self.relu(res_out)


BLOCKS = {
    "BasicBlock":BasicBlock,
    "Bottleneck":Bottleneck
}

class ResNet(NetBase):

    def __init__(self, cfg):
    
        super(ResNet, self).__init__()
        self.deep_stem = cfg.deep_stem
        self.out_indices = cfg.out_indices
        self._make_stem_layer(cfg.in_channels, cfg.stem_channels)

        block = BLOCKS[cfg.block]
        self.res_layers = nn.ModuleList()
        _in_channels = cfg.stem_channels
        _out_channels = cfg.base_channels * block.expansion
        for i, num_blocks in enumerate(cfg.stage_blocks):
            stride = cfg.strides[i]
            res_layer = self.make_res_layer(block=block, num_blocks=num_blocks, in_channels=_in_channels,
                                                                              out_channels=_out_channels, stride=stride, avg_down=cfg.avg_down)
            _in_channels = _out_channels
            _out_channels *= 2
            self.res_layers.append(res_layer)

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (nn.BatchNorm2d)):
                constant_init(m, 1)

    def make_res_layer(self, block, in_channels, out_channels, stride, num_blocks, avg_down):
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

        layers.append(block(in_channels=in_channels, out_channels=out_channels, stride=stride, downsample=downsample))
        in_channels = out_channels

        for _ in range(1, num_blocks):
            layers.append(block(in_channels=in_channels, out_channels=out_channels, stride=1))

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


def get_resnet_50(out_indices=(3, )):
    """
    18: (BasicBlock, (2, 2, 2, 2)),
    34: (BasicBlock, (3, 4, 6, 3)),
    50: (Bottleneck, (3, 4, 6, 3)),
    101: (Bottleneck, (3, 4, 23, 3)),
    152: (Bottleneck, (3, 8, 36, 3))
    """
    return ResNet(block=Bottleneck, stage_blocks=[3, 4, 6, 3], deep_stem=True, avg_down=True, out_indices=out_indices)
