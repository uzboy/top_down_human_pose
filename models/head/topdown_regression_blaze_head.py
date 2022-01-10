import torch.nn as nn
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
                                                                                        BatchNorm2d(mid_channels))
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
        self.joint_num = cfg.get("joint_num", 17)
        self.layers = nn.ModuleList()

        self.make_layer(in_channels=160, out_channels=160, num_blocks=3, stride=2)
        self.make_layer(in_channels=160, out_channels=160, num_blocks=3, stride=2)
        self.layers = nn.Sequential(*self.layers)
        self.loc_layer = nn.Sequential(nn.Conv2d(in_channels=self.in_channels,
                                                                                               out_channels=self.joint_num * 2,
                                                                                               kernel_size=2,
                                                                                               stride=1,
                                                                                               padding=0))
        self.loc_loss = cfg.get("loc_loss", None)
        if self.loc_loss is not None:
            self.loc_loss = build_loss(self.loc_loss)

        self.with_mask_layers = cfg.get("with_mask_layers", False)
        if self.with_mask_layers:
            self.mask_layer = nn.Sequential(nn.Conv2d(in_channels=self.in_channels,
                                                                                                        out_channels=self.joint_num,
                                                                                                        kernel_size=2,
                                                                                                        stride=1,
                                                                                                        padding=0),
                                                                                 nn.Sigmoid())
            self.mask_loss = cfg.get("mask_loss", None)
            if self.mask_loss is not None:
                self.mask_loss = build_loss(self.mask_loss)

    def make_layer(self, in_channels, out_channels, num_blocks, stride):
        self.in_channels = in_channels
        for index in range(num_blocks):
            if index > 0:
                stride = 1

            if stride != 1:
                down_samples = nn.MaxPool2d(kernel_size=2, stride=stride)
            else:
                down_samples = None
            
            self.layers.append(InvertedResidual(in_channels=self.in_channels,
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
        inputs, targets, target_weights = loss_inputs
        if self.with_mask_layers:
            loc_loss = self.loc_loss(inputs[0], targets[0], target_weights[0])
            mask_loss = self.mask_loss(inputs[1], targets[1], target_weights[0])
            return {
                "loc_loss": loc_loss,
                "mask_loss": mask_loss,
                "total_loss": loc_loss + mask_loss}
        else:
            loc_loss = self.loc_loss(inputs, targets, target_weights)
        
        return {
            "total_loss": loc_loss
        }

    def forward(self, x):
        x = self.layers(x)
        loc_out = self.loc_layer(x).squeeze(-1).squeeze(-1)
        if self.with_mask_layers:
            mask_out = self.mask_layer(x).squeeze(-1).squeeze(-1)
            return loc_out, mask_out
        else:
            return loc_out
