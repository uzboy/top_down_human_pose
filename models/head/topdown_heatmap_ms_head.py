import torch.nn as nn
import torch.nn.functional as F
from loss.build_loss import build_loss
from models.base_module import NetBase
from utils.utils import normal_init, kaiming_init, constant_init


class TopdownHeatmapMultiStageHead(NetBase):

    def __init__(self, cfg):
        super(TopdownHeatmapMultiStageHead, self).__init__()

        self.in_channels = cfg.in_channels
        self.num_stages = cfg.num_stages

        self.multi_deconv_layers = nn.ModuleList([])
        for _ in range(self.num_stages):
            if cfg.num_deconv_layers > 0:
                deconv_layers = self._make_deconv_layer(cfg.num_deconv_layers, cfg.num_deconv_filters, cfg.num_deconv_kernels)
            else:
                deconv_layers = nn.Identity()
            self.multi_deconv_layers.append(deconv_layers)

        self.multi_final_layers = nn.ModuleList([])
        for i in range(self.num_stages):
            self.multi_deconv_layers.append(nn.Conv2d(in_channels=cfg.num_deconv_filters[-1] if cfg.num_deconv_layers > 0 else cfg.in_channels,
                                                                                                           out_channels=cfg.out_channels,
                                                                                                           kernel_size=1,
                                                                                                           stride=1,
                                                                                                           padding=0))

        self.loss = None
        try:
            if cfg.loss is not None:
                self.loss = []
                for loss_cfg in cfg.loss:
                    self.loss.append(build_loss(loss_cfg))
        except:
            pass

    def get_loss(self, loss_inputs):
        outputs, targets, target_weights = loss_inputs
        assert isinstance(outputs, list)
        assert targets.dim() == 5 and target_weights.dim() == 4
        assert targets.size(1) == len(outputs)
        assert len(self.loss) == len(outputs)

        losses = []
        total_loss = 0
        for i in range(len(outputs)):
            target_i = targets[:, i, :, :, :]
            target_weight_i = target_weights[:, i, :, :]
            loss = self.loss[i](outputs[i], target_i, target_weight_i)
            total_loss += loss
            losses.append(loss)

        losses.append(total_loss)
        return losses

    def init_weight(self):
        for m in self.predict_layers.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
            elif isinstance(m, nn.Linear):
                normal_init(m, std=0.01)

    def _get_deconv_cfg(deconv_kernel):
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

    def forward(self, x):
        out = []
        for i in range(self.num_stages):
            y = self.multi_deconv_layers[i](x[i])
            y = self.multi_final_layers[i](y)
            out.append(y)

        return out
