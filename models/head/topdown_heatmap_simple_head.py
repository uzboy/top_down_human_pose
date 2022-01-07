import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_module import NetBase
from loss.build_loss import build_loss
from utils.utils import normal_init, constant_init


class TopdownHeatmapSimpleHead(NetBase):

    def __init__(self, cfg):
        super(TopdownHeatmapSimpleHead, self).__init__()

        self.input_transform = cfg.input_transform
        self.in_index = cfg.in_index
        if cfg.input_transform is not None:
            if cfg.input_transform == 'resize_concat':
                self.in_channels = sum(cfg.in_channels)
            else:
                self.in_channels = cfg.in_channels
        else:
            self.in_channels = cfg.in_channels

        self.align_corners = cfg.align_corners
    
        if cfg.num_deconv_layers > 0:
            self.deconv_layers = self._make_deconv_layer(cfg.num_deconv_layers, cfg.num_deconv_filters, cfg.num_deconv_kernels)
        else:
            self.deconv_layers = nn.Identity()

        conv_channels = cfg.num_deconv_filters[-1] if cfg.num_deconv_layers > 0 else self.in_channels

        layers = nn.ModuleList()
        if cfg.num_conv_layers != 0:
            for i in range(cfg.num_conv_layers):
                layers.append(nn.Conv2d(in_channels=conv_channels,
                                                                       out_channels=cfg.conv_layers_out[i],
                                                                       kernel_size=cfg.conv_layer_kernel[i],
                                                                       stride=1,
                                                                       padding=cfg.conv_layer_kernel[i] // 2,
                                                                       bias=False))
                layers.append(nn.BatchNorm2d(cfg.conv_layers_out[i]))
                layers.append(nn.ReLU(inplace=True))
                conv_channels = cfg.conv_layers_out[i]

        layers.append(nn.Conv2d(in_channels=conv_channels, out_channels=cfg.out_channels, kernel_size=1, stride=1, padding=0))
        if hasattr(cfg, "use_sigmoid") and cfg.use_sigmoid:
            layers.append(nn.Sigmoid())

        self.final_layer = nn.Sequential(*layers)

        
        if hasattr(cfg, "loss") and cfg.loss is not None:
            self.loss = build_loss(cfg.loss)
        else:
            self.loss = None

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
        x = self._transform_inputs(x)
        x = self.deconv_layers(x)
        x = self.final_layer(x)
        return x

    def _transform_inputs(self, inputs):
        if not isinstance(inputs, list):
            return inputs

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [F.interpolate(input=x,
                                                                                     size=inputs[0].shape[2:],
                                                                                     mode='bilinear',
                                                                                     align_corners=self.align_corners) for x in inputs]
            inputs = torch.cat(upsampled_inputs, dim=1)
            inputs = inputs[self.in_index]

        return inputs

    def _get_deconv_cfg(self, deconv_kernel):
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
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i])
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
