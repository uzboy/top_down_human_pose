import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_module import NetBase
from loss.build_loss import build_loss
from utils.utils import normal_init, constant_init


class Blocks(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=1, downsample=None):
        super().__init__()
        self.downsample=downsample
        hidden_dim = int(round(in_channels * expand_ratio))

        layers = []
        if expand_ratio != 1:
            layers.append(nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                                                                  out_channels=hidden_dim,
                                                                                                  kernel_size=1,
                                                                                                  stride=1,
                                                                                                  padding=0,
                                                                                                  bias=False),
                                                                          nn.BatchNorm2d(hidden_dim),
                                                                          nn.ReLU6(inplace=True)))
        layers.append(nn.Sequential(nn.Conv2d(in_channels=hidden_dim,
                                                                                              out_channels=hidden_dim,
                                                                                              kernel_size=3,
                                                                                              stride=stride,
                                                                                              groups=hidden_dim,
                                                                                              padding=1,
                                                                                              bias=False)),
                                                                     nn.BatchNorm2d(hidden_dim),
                                                                     nn.ReLU6(inplace=True))
        layers.append(nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                                                              out_channels=out_channels,
                                                                                              kernel_size=1,
                                                                                              stride=1,
                                                                                              padding=0,
                                                                                              bias=False),
                                                                     nn.BatchNorm2d(out_channels)))
        self.conv = nn.Sequential(*layers)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        identity = x
        res_path = self.conv(x)
        if self.downsample:
            identity = self.downsample(identity)
        return self.act(res_path + identity)


class TopdownRegressionHead(NetBase):

    def __init__(self, cfg):
        super(TopdownRegressionHead, self).__init__()

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
                                                                                                out_channels=cfg.pre_channels,
                                                                                                kernel_size=1,
                                                                                                stride=1,
                                                                                                padding=0,
                                                                                                bias=False),
                                                                        nn.BatchNorm2d(cfg.pre_channels),
                                                                        nn.ReLU6(inplace=True))

        self.blocks = []
        for index in range(len(cfg.blocks)):
            self.blocks.append(self.make_layer(cfg.pre_channels, cfg.out_channels, 2, cfg.blocks[index]))

        self.blocks = nn.Sequential(*self.blocks)

        self.final_layer = nn.Sequential(nn.Conv2d(in_channels=cfg.out_channels, out_channels=cfg.num_points * 2, kernel_size=2, stride=1, padding=0, bias=True),
                                                                         nn.Sigmoid())
        self.loss = None
        try:
            if cfg.loss is not None:
                self.loss = build_loss(cfg.loss)
        except:
            pass

    def make_layer(self, in_channels, out_channels, stride, num_blocks, avg_down=True):
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

        layers.append(Blocks(in_channels=in_channels, out_channels=out_channels, stride=stride, downsample=downsample))
        in_channels = out_channels

        for _ in range(1, num_blocks):
            layers.append(Blocks(in_channels=in_channels, out_channels=out_channels, stride=1))

        return nn.Sequential(*layers)

    def init_weight(self):
        for _, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
    
        for m in self.final_layer.modules():
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
        x = self.blocks(x)
        x = self.final_layer(x).squeeze(-1).squeeze(-1)
        return x
