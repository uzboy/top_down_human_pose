import torch.nn as nn
from models.base_module import NetBase
from utils.utils import kaiming_init, constant_init
from models.blocks.resnet_block import BLOCKS


class ResNet(NetBase):

    def __init__(self, cfg):
    
        super(ResNet, self).__init__()
        self.avg_down = cfg.avg_down
        self.deep_stem = cfg.deep_stem
        self.base_channels = cfg.base_channels
        self._make_stem_layer(cfg.in_channels, cfg.stem_channels)

        if cfg.frozen_stages is None:
            self.frozen_stages = -1
        elif isinstance(cfg.frozen_stages, int):
            self.frozen_stages = [cfg.frozen_stages]
        elif isinstance(cfg.frozen_stages, str) and cfg.frozen_stages == "all":
            self.frozen_stages = self.MAX_LAYERS
    
        self.layers = nn.ModuleList()
        self.in_channels = cfg.stem_channels
        self.out_channels = -1
        for i, settings in enumerate(cfg.stage_settings):
            self.make_res_layer(settings)
            self.out_channels *= 2
        
        if isinstance(cfg.out_indices, int):
            self.out_indices = [cfg.out_indices]
        else:
            self.out_indices = cfg.out_indices

        self.out_indices = [index if index > -1 else len(self.layers) - 1 for index in self.out_indices]

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (nn.BatchNorm2d)):
                constant_init(m, 1)

    def make_res_layer(self, settings):
        block_name, num_blocks, stride, group, width_per_group, base_channels, with_se, se_ratio = settings
        block = BLOCKS[block_name]
        if self.out_channels == -1:
            self.out_channels = self.base_channels * block.expansion

        downsample = None
        if stride != 1 or self.in_channels != self.out_channels:
            downsample = []
            conv_stride = stride
            if self.avg_down and stride != 1:
                conv_stride = 1
                downsample.append(nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False))
            downsample.append(nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1, stride=conv_stride, bias=False))
            downsample.append(nn.BatchNorm2d(self.out_channels))
            downsample = nn.Sequential(*downsample)

        layers = []
        layers.append(block(in_channels=self.in_channels,
                                                    out_channels=self.out_channels,
                                                    stride=stride,
                                                    downsample=downsample,
                                                    groups=group,
                                                    width_per_group=width_per_group,
                                                    base_channels=base_channels,
                                                    with_se=with_se,
                                                    se_ratio=se_ratio))
        self.in_channels = self.out_channels

        for _ in range(1, num_blocks):
            layers.append(block(in_channels=self.in_channels,
                                                        out_channels=self.out_channels,
                                                        stride=1,
                                                        downsample=None,
                                                        groups=group,
                                                        width_per_group=width_per_group,
                                                        base_channels=base_channels,
                                                        with_se=with_se,
                                                        se_ratio=se_ratio))
    
        self.layers.append(nn.Sequential(*layers))

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
                                                                  nn.ReLU(inplace=True),
                                                                  nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

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
        x = self.stem(x)
        outs = []

        for i, res_layer in enumerate(self.res_layers):
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)

        if len(outs) == 1:
            return outs[0]

        return tuple(outs)
