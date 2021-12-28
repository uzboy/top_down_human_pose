import torch.nn as nn
from models.base_module import NetBase
from utils.utils import kaiming_init, constant_init
from models.blocks.mobile_blocks import InvertedResidual
from models.actions.build_actions import ACT_NAME_MAPS
from models.actions.build_actions import HSwish


class MobileNetV3(NetBase):
    arch_settings = {
        'small': [[3, 16, 16, True, 'ReLU', 2],            # [kernel_size, mid_channels, out_channels, with_se, act_name, stride]
                         [3, 72, 24, False, 'ReLU', 2],
                         [3, 88, 24, False, 'ReLU', 1],
                         [5, 96, 40, True, 'HSwish', 2],
                         [5, 240, 40, True, 'HSwish', 1],
                         [5, 240, 40, True, 'HSwish', 1],
                         [5, 120, 48, True, 'HSwish', 1],
                         [5, 144, 48, True, 'HSwish', 1],
                         [5, 288, 96, True, 'HSwish', 2],
                         [5, 576, 96, True, 'HSwish', 1],
                         [5, 576, 96, True, 'HSwish', 1]],
        'big': [[3, 16, 16, False, 'ReLU', 1],
                     [3, 64, 24, False, 'ReLU', 2],
                     [3, 72, 24, False, 'ReLU', 1],
                     [5, 72, 40, True, 'ReLU', 2],
                     [5, 120, 40, True, 'ReLU', 1],
                     [5, 120, 40, True, 'ReLU', 1],
                     [3, 240, 80, False, 'HSwish', 2],
                     [3, 200, 80, False, 'HSwish', 1],
                     [3, 184, 80, False, 'HSwish', 1],
                     [3, 184, 80, False, 'HSwish', 1],
                     [3, 480, 112, True, 'HSwish', 1],
                     [3, 672, 112, True, 'HSwish', 1],
                     [5, 672, 160, True, 'HSwish', 1],
                     [5, 672, 160, True, 'HSwish', 2],
                     [5, 960, 160, True, 'HSwish', 1]]
    }

    def __init__(self, cfg):
        super(NetBase, self).__init__()
        self.arch = cfg.arch

        if cfg.frozen_stages is None:
            self.frozen_stages = -1
        elif isinstance(cfg.frozen_stages, int):
            self.frozen_stages = [cfg.frozen_stages]
        elif isinstance(cfg.frozen_stages, str) and cfg.frozen_stages == "all":
            self.frozen_stages = self.MAX_LAYERS
        self.in_channels = cfg.stem_channels
        self.layers = nn.ModuleList()

        self.layers.append(nn.Sequential(nn.Conv2d(in_channels=cfg.in_channels, out_channels=self.in_channels, kernel_size=3, stride=2, padding=1, bias=False),
                                                                              nn.BatchNorm2d(self.in_channels),
                                                                              HSwish(inplace=True)))

        layer_setting = self.arch_settings[self.arch]
        for _, params in enumerate(layer_setting):
            (kernel_size, mid_channels, out_channels, with_se, act, stride) = params
            if with_se:
                self.layers.append(InvertedResidual(in_channels=self.in_channels,
                                                                                            out_channels=out_channels,
                                                                                            mid_channels=mid_channels,
                                                                                            kernel_size=kernel_size,
                                                                                            stride=stride,
                                                                                            act_name=act,
                                                                                            with_se=True,
                                                                                            se_ratio=4,
                                                                                            se_acts=['ReLU', 'HSigmoid']))
            else:
                self.layers.append(InvertedResidual(in_channels=self.in_channels,
                                                                                            out_channels=out_channels,
                                                                                            mid_channels=mid_channels,
                                                                                            kernel_size=kernel_size,
                                                                                            stride=stride,
                                                                                            act_name=act,
                                                                                            with_se=False))
            self.in_channels = out_channels

        if isinstance(cfg.out_indices, int):
            self.out_indices = [cfg.out_indices]
        self.out_indices = [index if index > -1 else len(self.layers) - 1 for index in self.out_indices]

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, nn.BatchNorm2d):
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
