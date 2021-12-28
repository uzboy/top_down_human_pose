import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_module import NetBase
from models.blocks.mobile_blocks import InvertedResidual
from loss.build_loss import build_loss
from utils.utils import normal_init, constant_init


class TopdownRegressionBlazeHead(NetBase):
    arch_settings = [[1, 288, 5, 2],                 # expand_ratio, channel, num_blocks, stride
                                      [1, 288, 6, 2]]

    def __init__(self, cfg):
        super(TopdownRegressionBlazeHead, self).__init__()

        self.pre_conv = nn.Sequential(nn.Conv2d(in_channels=cfg.in_channels,
                                                                                                out_channels=cfg.in_channels,
                                                                                                kernel_size=3,
                                                                                                groups=cfg.in_channels,
                                                                                                stride=1,
                                                                                                padding=1,
                                                                                                bias=False),
                                                                        nn.BatchNorm2d(cfg.in_channels),
                                                                        nn.ReLU6(inplace=True),
                                                                        nn.Conv2d(in_channels=cfg.in_channels,
                                                                                                out_channels=288,
                                                                                                kernel_size=1,
                                                                                                stride=1,
                                                                                                padding=0,
                                                                                                bias=False),
                                                                        nn.BatchNorm2d(288),
                                                                        nn.ReLU6(inplace=True))
        self.in_channels = 288
        self.layers = nn.ModuleList()
        for _, layer_cfg in enumerate(self.arch_settings):
            expand_ratio, channel, num_blocks, stride = layer_cfg
            out_channels = channel
            self.make_layer(out_channels=out_channels, num_blocks=num_blocks, stride=stride, expand_ratio=expand_ratio)

        self.layers = nn.Sequential(*self.layers)
        self.final_layer = nn.Sequential(nn.Conv2d(in_channels=self.in_channels,
                                                                                                 out_channels=cfg.out_channels,
                                                                                                 kernel_size=2,
                                                                                                 stride=1,
                                                                                                 padding=0,
                                                                                                 bias=True))
        self.loss = None
        try:
            if cfg.loss is not None:
                self.loss = build_loss(cfg.loss)
        except:
            pass

    def make_layer(self, out_channels, num_blocks, stride, expand_ratio):
        for i in range(num_blocks):
            if i >= 1:
                stride = 1
            mid_channels = int(round(self.in_channels * expand_ratio))
            self.layers.append(InvertedResidual(in_channels=self.in_channels,
                                                                                        out_channels=out_channels,
                                                                                        mid_channels=mid_channels,
                                                                                        stride=stride))
            self.in_channels = out_channels

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)

    def get_loss(self, loss_inputs):
        input, target, target_weight = loss_inputs
        assert self.loss != None, "No Loss Function......."
        loss = self.loss(input, target, target_weight)
        return loss

    def forward(self, x):
        x = self.pre_conv(x)
        x = self.layers(x)
        x = self.final_layer(x).squeeze(-1).squeeze(-1)
        return x
