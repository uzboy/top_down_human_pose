import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import BatchNorm2d
from models.base_module import NetBase
from loss.build_loss import build_loss
from utils.utils import normal_init, constant_init
from models.blocks.eca_block import EcaModule


class InvertedResidual(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, down_samples=None):
        super(InvertedResidual, self).__init__()
        self.down_samples = down_samples

        mid_channels = in_channels * 4
        self.expand_conv = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                                                                        out_channels=mid_channels,
                                                                                                        kernel_size=1,
                                                                                                        stride=1,
                                                                                                        padding=0,
                                                                                                        bias=False),
                                                                                nn.BatchNorm2d(mid_channels),
                                                                                nn.ReLU6(inplace=True))

        self.depthwise_conv = nn.Sequential(nn.Conv2d(in_channels=mid_channels,
                                                                                                             out_channels=mid_channels,
                                                                                                             kernel_size=kernel_size,
                                                                                                             stride=stride,
                                                                                                             padding=kernel_size // 2,
                                                                                                             groups=mid_channels,
                                                                                                             bias=False),
                                                                                        BatchNorm2d(mid_channels))                # 激活，尽可能保留信息
        self.attention = EcaModule(mid_channels)

        self.linear_conv = nn.Sequential(nn.Conv2d(in_channels=mid_channels,
                                                                                                    out_channels=out_channels,
                                                                                                    kernel_size=1,
                                                                                                    stride=1,
                                                                                                    padding=0,
                                                                                                    bias=True))

    def forward(self, x):
        out = x

        out = self.expand_conv(out)
        out = self.depthwise_conv(out)
        out = self.attention(out)
        out = self.linear_conv(out)

        if self.down_samples is not None:
            x = self.down_samples(x)

        return out + x


class TopdownRegressionBlazeHead(NetBase):

    def __init__(self, cfg):
        super(TopdownRegressionBlazeHead, self).__init__()
        self.with_mask_layers = cfg.with_mask_layers
        self.joint_num = cfg.joint_num
        self.layers = nn.ModuleList()
        self.make_layer(in_channels=1280, out_channels=640, num_blocks=3, stride=2)
        self.make_layer(in_channels=640, out_channels=640, num_blocks=3, stride=2)
        self.layers = nn.Sequential(*self.layers)
        self.loc_layer = nn.Sequential(nn.Conv2d(in_channels=self.in_channels,
                                                                                               out_channels=self.joint_num * 2,
                                                                                               kernel_size=2,
                                                                                               stride=1,
                                                                                               padding=0))
        try:
            if cfg.loc_loss is not None:
                self.loc_loss = build_loss(cfg.loc_loss)
        except:
            self.loc_loss = None

        if self.with_mask_layers:
            self.mask_layer = nn.Sequential(nn.Conv2d(in_channels=self.in_channels,
                                                                                                        out_channels=self.joint_num,
                                                                                                        kernel_size=2,
                                                                                                        stride=1,
                                                                                                        padding=0),
                                                                                 nn.Sigmoid())
            try:
                if cfg.mask_loss is not None:
                    self.mask_loss = build_loss(cfg.mask_loss)
            except:
                self.mask_loss = None

    def make_layer(self, in_channels, out_channels, num_blocks, stride):
        self.in_channels = in_channels
        for index in range(num_blocks):
            if index > 0:
                stride = 1

            if stride != 1:
                down_samples = nn.MaxPool2d(kernel_size=2, stride=stride)
            else:
                down_samples = None
            
            self.layers.append(InvertedResidual(in_channels=out_channels,
                                                                                        out_channels=out_channels,
                                                                                        stride=stride,
                                                                                        down_samples=down_samples))
            self.in_channels = out_channels

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)

    def get_loss(self, loss_inputs):
        input, target, target_weight = loss_inputs

        loss = self.loc_loss(input[:, :self.joint_num * 2], target[:, :self.joint_num * 2], target_weight[:, :self.joint_num * 2])
        if self.with_mask_layers:
            loss += self.mask_loss(input[:, self.joint_num * 2:], target[:, self.joint_num * 2:], target_weight[:, self.joint_num * 2:])

        return loss

    def forward(self, x):
        x = self.layers(x)
        out_put = self.loc_layer(x).squeeze(-1).squeeze(-1)
        if self.with_mask_layers:
            mask_out = self.mask_layer(x).squeeze(-1).squeeze(-1)
            out_put = torch.cat([out_put, mask_out], dim=-1)
        
        return out_put
