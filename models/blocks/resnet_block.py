import torch.nn as nn
from models.blocks.se_block import SELayer


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, groups=1, width_per_group=4, base_channels=64, with_se=False, se_ratio=16):
        super(BasicBlock, self).__init__()
        self.downsample = downsample
        self.with_se = with_se
        mid_channels = out_channels // self.expansion
        if groups != 1:
            mid_channels = (groups * width_per_group * mid_channels // base_channels)

        self.res_conv = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                                                              out_channels=mid_channels,
                                                                                              kernel_size=3,
                                                                                              stride=stride,
                                                                                              padding=1,
                                                                                              groups=groups,
                                                                                              bias=False),
                                                                      nn.BatchNorm2d(mid_channels),
                                                                      nn.ReLU(inplace=True),
                                                                      nn.Conv2d(in_channels=mid_channels,
                                                                                              out_channels=out_channels,
                                                                                              kernel_size=3,
                                                                                              stride=1,
                                                                                              padding=1,
                                                                                              bias=False),
                                                                      nn.BatchNorm2d(out_channels))
        if self.with_se:
            self.se_layers = SELayer(out_channels, se_ratio)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        res_out = self.res_conv(x)
        if self.with_se:
            res_out = self.se_layers(res_out)
        
        if self.downsample is not None:
            identity = self.downsample(identity)

        res_out += identity
        return self.relu(res_out)


class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, groups=1, width_per_group=4, base_channels=64, with_se=False, se_ratio=16):
        super(Bottleneck, self).__init__()
        self.with_se = with_se
        self.downsample = downsample
        mid_channels = out_channels // self.expansion
        if groups != 1:
            mid_channels = (groups * width_per_group * mid_channels // base_channels)
    
        self.res_conv = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1, bias=False),
                                                                        nn.BatchNorm2d(mid_channels),
                                                                        nn.ReLU(inplace=True),
                                                                        nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False),
                                                                        nn.BatchNorm2d(mid_channels),
                                                                        nn.ReLU(inplace=True),
                                                                        nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False),
                                                                        nn.BatchNorm2d(out_channels))
        if self.with_se:
            self.se_layer = SELayer(out_channels, ratio=se_ratio)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        res_out = self.res_conv(x)
        if self.with_se:
            res_out = self.se_layer(res_out)
        
        if self.downsample is not None:
            identity = self.downsample(identity)

        res_out += identity
        return self.relu(res_out)


BLOCKS = {
    "BasicBlock":BasicBlock,
    "Bottleneck":Bottleneck
}
