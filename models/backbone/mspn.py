import torch.nn as nn
import torch.nn.functional as F
from models.base_module import NetBase
from utils.utils import normal_init, kaiming_init, constant_init


class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        assert out_channels % self.expansion == 0

        mid_channels = out_channels // self.expansion
        self.downsample = downsample

        self.res_conv = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1, bias=False),
                                                                        nn.BatchNorm2d(mid_channels),
                                                                        nn.ReLU(inplace=True),
                                                                        nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=stride, padding=dilation,
                                                                                                dilation=dilation, bias=False),
                                                                        nn.BatchNorm2d(mid_channels),
                                                                        nn.ReLU(inplace=True),
                                                                        nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False),
                                                                        nn.BatchNorm2d(out_channels))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res_out = self.res_conv(x)
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        res_out += identity
        return self.relu(res_out)


class DownsampleModule(nn.Module):

    def __init__(self, num_blocks, num_units=4, has_skip=False, in_channels=64):
        super(DownsampleModule, self).__init__()
        self.has_skip = has_skip
        self.in_channels = in_channels
        assert len(num_blocks) == num_units
        self.num_blocks = num_blocks
        self.num_units = num_units

        self.layers = nn.ModuleList()
        self.layers.append(self._make_layer(in_channels, num_blocks[0]))
        for i in range(1, num_units):
            self.layers.append(self._make_layer(in_channels * pow(2, i), num_blocks[i], stride=2))

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * Bottleneck.expansion:
            downsample = nn.Sequential(nn.Conv2d(in_channels=self.in_channels,
                                                                                                  out_channels=out_channels * Bottleneck.expansion,
                                                                                                  kernel_size=1, stride=stride, padding=0, bias=False),
                                                                         nn.BatchNorm2d(out_channels * Bottleneck.expansion))

        units = list()
        units.append(Bottleneck(self.in_channels, out_channels * Bottleneck.expansion, stride=stride, downsample=downsample))
        self.in_channels = out_channels * Bottleneck.expansion

        for _ in range(1, blocks):
            units.append(Bottleneck(self.in_channels, out_channels * Bottleneck.expansion))

        return nn.Sequential(*units)

    def forward(self, x, skip1, skip2):
        out = list()

        for i in range(self.num_units):
            x = self.layers[i](x)
            if self.has_skip:
                x = x + skip1[i] + skip2[i]
            out.append(x)

        out.reverse()

        return tuple(out)


class UpsampleUnit(nn.Module):

    def __init__(self, ind, num_units, in_channels, unit_channels=256, gen_skip=False, gen_cross_conv=False, out_channels=64):
        super(UpsampleUnit, self).__init__()
        self.num_units = num_units

        self.in_skip = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=unit_channels, kernel_size=1, stride=1, padding=0, bias=False),
                                                                  nn.BatchNorm2d(unit_channels))

        self.relu = nn.ReLU(inplace=True)

        self.ind = ind
        if self.ind > 0:
            self.up_conv = nn.Sequential(nn.Conv2d(in_channels=unit_channels, out_channels=unit_channels, kernel_size=1, stride=1, padding=0, bias=False),
                                                                         nn.BatchNorm2d(unit_channels))

        self.gen_skip = gen_skip
        if self.gen_skip:
            self.out_skip1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0, bias=False),
                                                                            nn.BatchNorm2d(in_channels),
                                                                            nn.ReLU(inplace=True))
            self.out_skip2 = nn.Sequential(nn.Conv2d(in_channels=unit_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0, bias=False),
                                                                            nn.BatchNorm2d(in_channels),
                                                                            nn.ReLU(inplace=True))

        self.gen_cross_conv = gen_cross_conv
        if self.ind == num_units - 1 and self.gen_cross_conv:
            self.cross_conv = nn.Sequential(nn.Conv2d(in_channels=unit_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                                                                                nn.BatchNorm2d(out_channels),
                                                                                nn.ReLU(inplace=True))

    def forward(self, x, up_x):
        out = self.in_skip(x)

        if self.ind > 0:
            up_x = F.interpolate(up_x, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
            up_x = self.up_conv(up_x)
            out = out + up_x

        out = self.relu(out)

        skip1 = None
        skip2 = None
        if self.gen_skip:
            skip1 = self.out_skip1(x)
            skip2 = self.out_skip2(out)

        cross_conv = None
        if self.ind == self.num_units - 1 and self.gen_cross_conv:
            cross_conv = self.cross_conv(out)

        return out, skip1, skip2, cross_conv


class UpsampleModule(nn.Module):

    def __init__(self, unit_channels=256, num_units=4, gen_skip=False, gen_cross_conv=False, out_channels=64):
        super(UpsampleModule, self).__init__()
        self.in_channels = list()

        for i in range(num_units):
            self.in_channels.append(Bottleneck.expansion * out_channels * pow(2, i))

        self.in_channels.reverse()
        self.num_units = num_units
        self.gen_skip = gen_skip
        self.gen_cross_conv = gen_cross_conv

        self.units = nn.ModuleList()
        for i in range(num_units):
            self.units.append(UpsampleUnit(i, self.num_units, self.in_channels[i], unit_channels, self.gen_skip, self.gen_cross_conv, out_channels=64))

    def forward(self, x):
        out = list()
        skip1 = list()
        skip2 = list()
        cross_conv = None
        for i in range(self.num_units):
            module_i = self.units[i]

            if i == 0:
                outi, skip1_i, skip2_i, _ = module_i(x[i], None)
            elif i == self.num_units - 1:
                outi, skip1_i, skip2_i, cross_conv = module_i(x[i], out[i - 1])
            else:
                outi, skip1_i, skip2_i, _ = module_i(x[i], out[i - 1])
    
            out.append(outi)
            skip1.append(skip1_i)
            skip2.append(skip2_i)
    
        skip1.reverse()
        skip2.reverse()

        return out, skip1, skip2, cross_conv


class SingleStageNetwork(nn.Module):

    def __init__(self, has_skip=False, gen_skip=False, gen_cross_conv=False, unit_channels=256,
                              num_units=4, num_blocks=[2, 2, 2, 2], in_channels=64):
        super(SingleStageNetwork, self).__init__()
        assert len(num_blocks) == num_units
        self.has_skip = has_skip
        self.gen_skip = gen_skip
        self.gen_cross_conv = gen_cross_conv
        self.num_units = num_units
        self.unit_channels = unit_channels
        self.num_blocks = num_blocks

        self.downsample = DownsampleModule(num_blocks, num_units, has_skip, in_channels)
        self.upsample = UpsampleModule(unit_channels, num_units, gen_skip, gen_cross_conv, in_channels)

    def forward(self, x, skip1, skip2):
        mid = self.downsample(x, skip1, skip2)
        out, skip1, skip2, cross_conv = self.upsample(mid)

        return out, skip1, skip2, cross_conv


class ResNetTop(nn.Module):

    def __init__(self, channels=64):
        super(ResNetTop, self).__init__()
        self.top = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=7, stride=2, padding=3, bias=False),
                                                           nn.BatchNorm2d(channels),
                                                           nn.ReLU(inplace=True),
                                                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def forward(self, img):
        return self.top(img)


class MSPN(NetBase):

    def __init__(self, cfg):
        super(MSPN, self).__init__()
        self.unit_channels = cfg.unit_channels
        self.num_stages = cfg.num_stages
        self.num_units = cfg.num_units
        self.num_blocks = cfg.num_blocks

        self.top = ResNetTop()
        self.multi_stage_mspn = nn.ModuleList([])

        for i in range(self.num_stages):
            if i == 0:
                has_skip = False
            else:
                has_skip = True

            if i != self.num_stages - 1:
                gen_skip = True
                gen_cross_conv = True
            else:
                gen_skip = False
                gen_cross_conv = False

            self.multi_stage_mspn.append(SingleStageNetwork(has_skip, gen_skip, gen_cross_conv, cfg.unit_channels,
                                                                                                                         cfg.num_units, cfg.num_blocks, cfg.res_top_channels))

    def init_weight(self):
        for m in self.multi_stage_mspn.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
            elif isinstance(m, nn.Linear):
                normal_init(m, std=0.01)

        for m in self.top.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)

    def forward(self, x):
        out_feats = []
        skip1 = None
        skip2 = None
        x = self.top(x)

        for i in range(self.num_stages):
            out, skip1, skip2, x = self.multi_stage_mspn[i](x, skip1, skip2)
            out_feats.append(out)

        return out_feats
