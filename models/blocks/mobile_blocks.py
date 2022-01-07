import torch.nn as nn
from models.blocks.se_block import SELayer
from models.actions.build_actions import ACT_NAME_MAPS


class InvertedResidual(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels, kernel_size=3, groups=None, stride=1, act_name='ReLU6',
                              with_se=False, se_ratio=16, se_acts=['ReLU', 'Sigmoid'], down_samples=None):
        super(InvertedResidual, self).__init__()
        self.with_se = with_se
        self.with_res_shortcut = (in_channels == out_channels)
        self.with_expand_conv = (in_channels != mid_channels)
        self.down_samples = down_samples
        if groups is None:
            groups = mid_channels

        if self.with_expand_conv:
            self.expand_conv = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                                                                           out_channels=mid_channels,
                                                                                                           kernel_size=1,
                                                                                                           stride=1,
                                                                                                           padding=0,
                                                                                                           bias=False),
                                                                                    nn.BatchNorm2d(mid_channels),
                                                                                    ACT_NAME_MAPS[act_name]())

        self.depthwise_conv = nn.Sequential(nn.Conv2d(in_channels=mid_channels,
                                                                                                             out_channels=mid_channels,
                                                                                                             kernel_size=kernel_size,
                                                                                                             stride=stride,
                                                                                                             padding=kernel_size // 2,
                                                                                                             groups=groups,
                                                                                                             bias=False),
                                                                                        nn.BatchNorm2d(mid_channels),
                                                                                        ACT_NAME_MAPS[act_name]())
        if self.with_se:
            self.se = SELayer(mid_channels, se_ratio, se_acts)
    
        self.linear_conv = nn.Sequential(nn.Conv2d(in_channels=mid_channels,
                                                                                                    out_channels=out_channels,
                                                                                                    kernel_size=1,
                                                                                                    stride=1,
                                                                                                    padding=0,
                                                                                                    bias=False),
                                                                            nn.BatchNorm2d(out_channels))

    def forward(self, x):
        out = x

        if self.with_expand_conv:
            out = self.expand_conv(out)

        out = self.depthwise_conv(out)

        if self.with_se:
            out = self.se(out)

        out = self.linear_conv(out)

        if self.with_res_shortcut:
            if self.down_samples is not None:
                x = self.down_samples(x)
            return x + out

        return out
