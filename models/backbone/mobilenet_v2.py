import torch.nn as nn
from models.base_module import NetBase
from utils.utils import kaiming_init, constant_init,make_divisible
from models.blocks.mobile_blocks import InvertedResidual


class MobileNetV2(NetBase):

    arch_settings = [[1, 16, 1, 1],                 # expand_ratio, channel, num_blocks, stride
                                      [6, 24, 2, 2],
                                      [6, 32, 3, 2],
                                      [6, 64, 4, 2],
                                      [6, 96, 3, 1],
                                      [6, 160, 3, 2],
                                      [6, 320, 1, 1]]

    def __init__(self, cfg):
        super(MobileNetV2, self).__init__()
        self.widen_factor = cfg.widen_factor

        if cfg.frozen_stages is None:
            self.frozen_stages = -1
        elif isinstance(cfg.frozen_stages, int):
            self.frozen_stages = [cfg.frozen_stages]
        elif isinstance(cfg.frozen_stages, str) and cfg.frozen_stages == "all":
            self.frozen_stages = self.MAX_LAYERS
    
        self.in_channels = make_divisible(32 * cfg.widen_factor, 8)

        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(nn.Conv2d(in_channels=cfg.in_channels,
                                                                                                      out_channels=self.in_channels,
                                                                                                      kernel_size=3, stride=2, padding=1, bias=False),
                                                                              nn.BatchNorm2d(self.in_channels),
                                                                              nn.ReLU6(inplace=True)))

        for _, layer_cfg in enumerate(self.arch_settings):
            expand_ratio, channel, num_blocks, stride = layer_cfg
            out_channels = make_divisible(channel * cfg.widen_factor, 8)
            self.make_layer(out_channels=out_channels, num_blocks=num_blocks, stride=stride, expand_ratio=expand_ratio)

        if cfg.widen_factor > 1.0:
            self.out_channel = int(1280 * cfg.widen_factor)
        else:
            self.out_channel = 1280
        self.layers.append(nn.Sequential(nn.Conv2d(in_channels=self.in_channels,
                                                                                                      out_channels=self.out_channel,
                                                                                                      kernel_size=1,
                                                                                                      stride=1,
                                                                                                      padding=0,
                                                                                                      bias=False),
                                                                                nn.BatchNorm2d(self.out_channel),
                                                                                nn.ReLU6(inplace=True)))

        if isinstance(cfg.out_indices, int):
            self.out_indices = [cfg.out_indices]
        self.out_indices = [index if index > -1 else len(self.layers) + index for index in self.out_indices]            # 输出索引
    
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
                kaiming_init(m)
            elif isinstance(m, (nn.BatchNorm2d)):
                constant_init(m, 1)

    def freeze_model(self):
        if isinstance(self.frozen_stages, int) and self.frozen_stages == -1:
            return
        elif isinstance(self.frozen_stages, int) and self.frozen_stages == self.MAX_LAYERS:
            for param in self.parameters():
                param.requires_grad = False
        else:
            for frozen_index in self.frozen_stages:
                for param in self.layers[frozen_index].parameters():
                    param.requires_grad = False

    def forward(self, x):
        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]
    
        return tuple(outs)
