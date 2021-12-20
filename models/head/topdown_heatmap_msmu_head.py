import torch.nn as nn
import torch.nn.functional as F
from models.base_module import NetBase
from utils.utils import kaiming_init
from utils.utils import normal_init
from utils.utils import constant_init
from loss.build_loss import build_loss


class PRM(nn.Module):

    def __init__(self, out_channels):
        super(PRM, self).__init__()
        self.out_channels = out_channels
        self.top_path = nn.Sequential(nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                                                                      nn.BatchNorm2d(out_channels),
                                                                      nn.ReLU(inplace=True))

        # 可以替换成1维卷积实现的类似于SE的结构，同时确保Sigmoid之前的ReLU有效
        self.middle_path = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                                                              nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                                                                              nn.BatchNorm2d(out_channels),
                                                                              nn.ReLU(inplace=True),
                                                                              nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                                                                              nn.BatchNorm2d(out_channels),
                                                                              nn.ReLU(inplace=True),    # 确保每一个通道的权重都 （>=0.5)
                                                                              nn.Sigmoid())
        
        self.dwn_path = nn.Sequential(nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                                                                         nn.BatchNorm2d(out_channels),
                                                                         nn.ReLU(inplace=True),
                                                                         nn.Conv2d(in_channels=out_channels,
                                                                                                 out_channels=out_channels,
                                                                                                 kernel_size=9,
                                                                                                 stride=1,
                                                                                                 padding=4,
                                                                                                 groups=out_channels,
                                                                                                 bias=False),
                                                                          nn.BatchNorm2d(out_channels),
                                                                          nn.ReLU(inplace=True),
                                                                          nn.Conv2d(in_channels=out_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False),
                                                                          nn.BatchNorm2d(1),
                                                                          nn.ReLU(inplace=True),
                                                                          nn.Sigmoid())

    def forward(self, x):
        out = self.top_path(x)
        mid_out = self.middle_path(out)
        dwn_out = self.dwn_path(out)
        return out * (1 + mid_out * dwn_out)


class PredictHeatmap(nn.Module):

    def __init__(self, unit_channels, out_channels, out_shape, use_prm=False):

        super().__init__()
        self.out_shape = out_shape
        self.use_prm = use_prm

        if use_prm:
            self.prm = PRM(out_channels)

        # 第一个1*1的卷积是否有效，后续尝试是否可以去除
        self.conv_layers = nn.Sequential(nn.Conv2d(in_channels=unit_channels, out_channels=unit_channels,
                                                                                                    kernel_size=1, stride=1, padding=0, bias=False),
                                                                            nn.BatchNorm2d(unit_channels),
                                                                            nn.ReLU(inplace=True),
                                                                            nn.Conv2d(in_channels=unit_channels, out_channels=out_channels,
                                                                                                    kernel_size=3, stride=1, padding=1, bias=False),
                                                                            nn.BatchNorm2d(out_channels))

    def forward(self, feature):
        feature = self.conv_layers(feature)
        output = F.interpolate(feature, size=self.out_shape, mode='bilinear', align_corners=True)
        if self.use_prm:
            output = self.prm(output)
        return output


class TopdownHeatmapMSMUHead(NetBase):

    def __init__(self, cfg):
        super(TopdownHeatmapMSMUHead, self).__init__()
        self.out_shape = cfg.out_shape
        self.unit_channels = cfg.unit_channels
        self.out_channels = cfg.out_channels
        self.num_stages = cfg.num_stages
        self.num_units = cfg.num_units

        self.predict_layers = nn.ModuleList([])
        for _ in range(self.num_stages):
            for _ in range(self.num_units):
                self.predict_layers.append(PredictHeatmap(cfg.unit_channels, cfg.out_channels, cfg.out_shape, cfg.use_prm))

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

    def forward(self, x):
        out = []
        for i in range(self.num_stages):
            for j in range(self.num_units):
                y = self.predict_layers[i * self.num_units + j](x[i][j])
                out.append(y)

        return out
