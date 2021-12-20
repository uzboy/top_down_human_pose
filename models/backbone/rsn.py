import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_module import NetBase
from utils.utils import kaiming_init, constant_init, normal_init

"""
1. 需要优化：
        1. 网络开始地方的 7*7卷积，替换成两个3*3卷积
        2. 对于RSB中stride =2 时，使用的1*1卷积进行下采样，丢掉一半数据，对下采样进行修改
"""

class RSB(nn.Module):

    expansion = 1

    def __init__(self, in_channels, out_channels, num_steps=4, stride=1, downsample=None, expand_times=26, res_top_channels=64):
        super(RSB, self).__init__()
        assert num_steps > 1

        self.branch_channels = in_channels * expand_times
        self.branch_channels //= res_top_channels
        self.downsample = downsample
        self.num_steps = num_steps

        # TODO: 1 * 1的卷积，卷积的stride只能为1， 此处为self.stride
        self.conv_bn_relu1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=self.num_steps * self.branch_channels,
                                                                                                            kernel_size=1, stride=stride, padding=0, bias=False),
                                                                                   nn.BatchNorm2d(self.num_steps * self.branch_channels),
                                                                                   nn.ReLU(inplace=True))       # TODO inplace 原始版本实现为false

        self.step_convs = []
        for i in range(self.num_steps):
            step_convs = nn.ModuleList()
            for _ in range(i + 1):
                convs = nn.Sequential(nn.Conv2d(in_channels=self.branch_channels,
                                                                                       out_channels=self.branch_channels,
                                                                                       kernel_size=3,
                                                                                       stride=1,
                                                                                       padding=1,
                                                                                       bias=False),
                                                                nn.BatchNorm2d(self.branch_channels),
                                                                nn.ReLU(inplace=True))
                step_convs.append(convs)
            self.step_convs.append(step_convs)
        self.step_convs = nn.ModuleList(self.step_convs)
    
        self.conv_bn3 = nn.Sequential(nn.Conv2d(in_channels=self.num_steps * self.branch_channels,
                                                                                                out_channels=out_channels * self.expansion, kernel_size=1, stride=1, bias=False),
                                                                        nn.BatchNorm2d(out_channels * self.expansion))

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        identity = x
        x = self.conv_bn_relu1(x)
        spx = torch.split(x, self.branch_channels, 1)

        outputs = list()
        outs = list()
    
        for i in range(self.num_steps):
            outputs_i = list()
            outputs.append(outputs_i)
            for j in range(i + 1):
                if j == 0:
                    inputs = spx[i]
                else:
                    inputs = outputs[i][j - 1]

                if i > j:
                    inputs = inputs + outputs[i - 1][j]

                outputs[i].append(self.step_convs[i][j](inputs))

            outs.append(outputs[i][i])
    
        out = torch.cat(tuple(outs), 1)
        out = self.conv_bn3(out)

        if self.downsample is not None:
            identity = self.downsample(identity)
    
        out = out + identity
        out = self.relu(out)

        return out


class Downsample_module(nn.Module):

    def __init__(self, num_blocks, num_steps=4, num_units=4, has_skip=False, in_channels=64, expand_times=26):

        super(Downsample_module, self).__init__()
        assert len(num_blocks) == num_units
    
        self.has_skip = has_skip
        self.in_channels = in_channels
        self.num_units = num_units
        self.num_steps = num_steps
    
        self.layers = []
        self.layers.append(self._make_layer(in_channels, num_blocks[0], expand_times=expand_times, res_top_channels=in_channels))

        for i in range(1, num_units):
            self.layers.append(self._make_layer(in_channels * pow(2, i), num_blocks[i], stride=2, expand_times=expand_times, res_top_channels=in_channels))
        self.layers = nn.ModuleList(self.layers)

    def _make_layer(self, out_channels, blocks, stride=1, expand_times=26, res_top_channels=64):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * RSB.expansion:
            downsample = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels * RSB.expansion, kernel_size=1, stride=stride, padding=0, bias=False),
                                                                          nn.BatchNorm2d(out_channels * RSB.expansion),
                                                                          nn.ReLU(inplace=True))
    
        units = list()
        units.append(RSB(self.in_channels, out_channels,
                                               num_steps=self.num_steps,
                                               stride=stride,
                                               downsample=downsample,
                                               expand_times=expand_times,
                                               res_top_channels=res_top_channels))

        self.in_channels = out_channels * RSB.expansion
        for _ in range(1, blocks):
            units.append(RSB(self.in_channels, out_channels,
                                                   num_steps=self.num_steps,
                                                   expand_times=expand_times,
                                                   res_top_channels=res_top_channels))

        return nn.Sequential(*units)

    def forward(self, x, skip1, skip2):
        out = list()
        for i in range(self.num_units):
            module_i = self.layers[i]
            x = module_i(x)
            if self.has_skip:
                x = x + skip1[i] + skip2[i]
            out.append(x)

        out.reverse()

        return tuple(out)


class Upsample_unit(nn.Module):

    def __init__(self, ind, num_units, in_channels, unit_channels=256, gen_skip=False, gen_cross_conv=False, out_channels=64):
        super(Upsample_unit, self).__init__()
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
            self.out_skip2 = nn.Sequential(nn.Conv2d(in_channels=unit_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0, bias=False,),
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


class Upsample_module(nn.Module):

    def __init__(self, unit_channels=256, num_units=4, gen_skip=False, gen_cross_conv=False, out_channels=64):
        super().__init__()
        self.in_channels = list()
        for i in range(num_units):
            self.in_channels.append(RSB.expansion * out_channels * pow(2, i))

        self.in_channels.reverse()
        self.num_units = num_units
        self.gen_skip = gen_skip
        self.gen_cross_conv = gen_cross_conv

        self.layers = nn.ModuleList()

        for i in range(num_units):
            self.layers.append(Upsample_unit(i, self.num_units, self.in_channels[i], unit_channels, self.gen_skip, self.gen_cross_conv, out_channels=64))

    def forward(self, x):
        out = list()
        skip1 = list()
        skip2 = list()
        cross_conv = None

        for i in range(self.num_units):
            module_i = self.layers[i]
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


class Single_stage_RSN(nn.Module):

    def __init__(self, has_skip=False, gen_skip=False, gen_cross_conv=False, unit_channels=256, num_units=4,
                              num_steps=4, num_blocks=[2, 2, 2, 2], in_channels=64, expand_times=26):
        super(Single_stage_RSN, self).__init__()
        assert len(num_blocks) == num_units
        self.has_skip = has_skip
        self.gen_skip = gen_skip
        self.gen_cross_conv = gen_cross_conv
        self.num_units = num_units
        self.num_steps = num_steps
        self.unit_channels = unit_channels
        self.num_blocks = num_blocks

        self.downsample = Downsample_module(num_blocks, num_steps, num_units, has_skip, in_channels, expand_times)
        self.upsample = Upsample_module(unit_channels, num_units, gen_skip, gen_cross_conv, in_channels)

    def forward(self, x, skip1, skip2):
        mid = self.downsample(x, skip1, skip2)
        out, skip1, skip2, cross_conv = self.upsample(mid)
        return out, skip1, skip2, cross_conv


class ResNet_top(nn.Module):

    def __init__(self, channels=64):
        super(ResNet_top, self).__init__()

        self.top = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=7, stride=2, padding=3, bias=False),
                                                            nn.BatchNorm2d(channels),
                                                            nn.ReLU(inplace=True),
                                                            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def forward(self, img):
        return self.top(img)


class RSN(NetBase):

    def __init__(self, cfg):
        super(RSN, self).__init__()
        self.unit_channels = cfg.unit_channels
        self.num_stages = cfg.num_stages
        self.num_units = cfg.num_units
        self.num_blocks = cfg.num_blocks
        self.num_steps = cfg.num_steps

        assert self.num_stages > 0
        assert self.num_steps > 1
        assert self.num_units > 1
        assert self.num_units == len(self.num_blocks)

        self.top = ResNet_top()         # stem: 替换成resnet中的stem

        self.multi_stage_rsn = nn.ModuleList([])
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

            self.multi_stage_rsn.append(Single_stage_RSN(has_skip, gen_skip, gen_cross_conv, cfg.unit_channels, cfg.num_units, cfg.num_steps,
                                                                                                                cfg.num_blocks, cfg.res_top_channels, cfg.expand_times))

    def _make_stem_layer(self, in_channels, stem_channels):
        if self.deep_stem:
            self.stem = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=stem_channels // 2, kernel_size=3, stride=2, padding=1, bias=False),
                                                                  nn.BatchNorm2d(stem_channels // 2),
                                                                  nn.ReLU(),
                                                                  nn.Conv2d(in_channels=stem_channels // 2, out_channels=stem_channels // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                                                  nn.BatchNorm2d(stem_channels // 2),
                                                                  nn.ReLU(),
                                                                  nn.Conv2d(in_channels=stem_channels // 2, out_channels=stem_channels, kernel_size=3, stride=1, padding=1, bias=False),
                                                                  nn.BatchNorm2d(stem_channels),
                                                                  nn.ReLU(),
                                                                  nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        else:
            self.stem = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=stem_channels, kernel_size=7, stride=2, padding=3, bias=False),
                                                                  nn.BatchNorm2d(stem_channels),
                                                                  nn.ReLU(),
                                                                  nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def init_weight(self):
        for m in self.multi_stage_rsn.modules():
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
            out, skip1, skip2, x = self.multi_stage_rsn[i](x, skip1, skip2)
            out_feats.append(out)

        return out_feats
