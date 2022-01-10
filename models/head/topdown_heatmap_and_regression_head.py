import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import BatchNorm2d
from models.base_module import NetBase
from loss.build_loss import build_loss
from utils.utils import constant_init, kaiming_init
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


class TopdownRegressionAndHeatmap(NetBase):

    def __init__(self, cfg):
        super(TopdownRegressionAndHeatmap, self).__init__()
        self.joint_num = cfg.get("joint_num", 17)
        self.__make_regression_head(cfg)
        self.__make_heatmap_head(cfg)

    def __make_heatmap_head(self, cfg):
        self.in_channels = cfg.get("heatmap_head_input_channels", 1280)
        self.num_deconv_layers = cfg.get("num_deconv_layers", 3)
        selfnum_deconv_kernels = cfg.get("num_deconv_kernels", (4, 4, 4))
        self.num_deconv_filters = cfg.get("num_deconv_filters", (256, 256, 256))
        
        self.heatmap_head = self._make_deconv_layer(self.num_deconv_layers,
                                                                                                          self.num_deconv_filters,
                                                                                                          selfnum_deconv_kernels)
        conv_channels = self.num_deconv_filters[-1]
        layers = nn.ModuleList()
        layers.append(nn.Conv2d(in_channels=conv_channels, out_channels=self.joint_num, kernel_size=1, stride=1, padding=0))
        use_sigmoid = cfg.get("heatmap_use_sigmoid", False)
        if use_sigmoid:
            layers.append(nn.Sigmoid())
        self.heatmap_layers = nn.Sequential(*layers)

        self.heatmap_loss = cfg.get("heatmap_loss", None)
        if self.heatmap_loss is not None:
            self.heatmap_loss = build_loss(self.heatmap_loss)

    def __make_regression_head(self, cfg):
        self.in_channels = cfg.get("reg_head_in_channels", 160)
        self.regression_head = nn.ModuleList()
        self.__make_layer(in_channels=self.in_channels, out_channels=self.in_channels, num_blocks=3, stride=2)
        self.__make_layer(in_channels=self.in_channels, out_channels=self.in_channels, num_blocks=3, stride=2)
        self.regression_head = nn.Sequential(*self.regression_head)
        self.regression_loc_layer = nn.Sequential(nn.Conv2d(in_channels=self.in_channels,
                                                                                                                      out_channels=self.joint_num * 2,
                                                                                                                      kernel_size=2,
                                                                                                                      stride=1,
                                                                                                                      padding=0))
        self.regression_loc_loss = cfg.get("reg_loc_loss", None)
        if self.regression_loc_loss is not None:
            self.regression_loc_loss = build_loss(self.regression_loc_loss)

        self.reg_with_mask_layers = cfg.get("reg_with_mask_layers", False)
        if self.reg_with_mask_layers:
            self.regression_mask_layer = nn.Sequential(nn.Conv2d(in_channels=self.in_channels,
                                                                                                                               out_channels=self.joint_num,
                                                                                                                               kernel_size=2,
                                                                                                                               stride=1,
                                                                                                                               padding=0),
                                                                                                        nn.Sigmoid())
            self.regression_mask_loss = cfg.get("reg_mask_loss", None)
            if self.regression_mask_loss is not None:
                self.regression_mask_loss = build_loss(self.regression_mask_loss)

    def __make_layer(self, in_channels, out_channels, num_blocks, stride):
        self.in_channels = in_channels
        for index in range(num_blocks):
            if index > 0:
                stride = 1

            if stride != 1:
                down_samples = nn.MaxPool2d(kernel_size=2, stride=stride)
            else:
                down_samples = None
            
            self.regression_head.append(InvertedResidual(in_channels=self.in_channels,
                                                                                                              out_channels=out_channels,
                                                                                                              stride=stride,
                                                                                                              down_samples=down_samples))
            self.in_channels = out_channels

    def __get_deconv_cfg(self, deconv_kernel):
            if deconv_kernel == 4:
                padding = 1
                output_padding = 0
            elif deconv_kernel == 3:
                padding = 1
                output_padding = 1
            elif deconv_kernel == 2:
                padding = 0
                output_padding = 0
            else:
                raise ValueError(f'Not supported num_kernels ({deconv_kernel}).')

            return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = self.__get_deconv_cfg(num_kernels[i])
            planes = num_filters[i]
            layers.append(nn.ConvTranspose2d(in_channels=self.in_channels,
                                                                                        out_channels=planes,
                                                                                        kernel_size=kernel,
                                                                                        stride=2,
                                                                                        padding=padding,
                                                                                        output_padding=output_padding,
                                                                                        bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.in_channels = planes

        return nn.Sequential(*layers)

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)

    def get_loss(self, loss_inputs):
        pred_values, targets, target_weights = loss_inputs
        heatmap_loss = self.heatmap_loss(pred_values[0], targets[0], target_weights[0])
        reg_loc_loss = self.regression_loc_loss(pred_values[1], targets[1], target_weights[1])
        if self.reg_with_mask_layers:
            reg_mask_loss = self.regression_mask_loss(pred_values[2], targets[2], target_weights[2])
            return {
                "heatmap_loss":heatmap_loss,
                "reg_loc_loss":reg_loc_loss,
                "reg_mask_loss":reg_mask_loss,
                "total_loss":heatmap_loss + reg_loc_loss + reg_mask_loss
            }
        else:
            return {
                "heatmap_loss":heatmap_loss,
                "reg_loc_loss":reg_loc_loss,
                "total_loss":heatmap_loss + reg_loc_loss
            }

    def forward(self, inputs):
        heatmap_out = self.heatmap_head(inputs[1])
        heatmap_out = self.heatmap_layers(heatmap_out)

        reg_out = self.regression_head(inputs[0])
        reg_loc_out = self.regression_loc_layer(reg_out).squeeze(-1).squeeze(-1)
        if self.reg_with_mask_layers:
            reg_mask_out = self.regression_mask_layer(reg_out).squeeze(-1).squeeze(-1)
            return heatmap_out, reg_loc_out, reg_mask_out
        else:
            return heatmap_out, reg_loc_out
