import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from models.base_module import NetBase
from utils.utils import constant_init, normal_init


class Channel_Shuffle(nn.Module):

    def __init__(self, groups):
        super(Channel_Shuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        batch_size, num_channels, height, width = x.size()
        assert (num_channels % self.groups == 0), ('num_channels should be divisible by groups')
        channels_per_group = num_channels // self.groups
        x = x.view(batch_size, self.groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batch_size, -1, height, width)
        return x


class SpatialWeighting(nn.Module):

    def __init__(self, channels, ratio=16):
        super().__init__()
    
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=channels, out_channels=int(channels / ratio), kernel_size=1, stride=1, padding=0, bias=False),
                                                                nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=int(channels / ratio), out_channels=channels, kernel_size=1, stride=1, padding=0, bias=False),
                                                                nn.Sigmoid())

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.conv2(out)
        return x * out


class CrossResolutionWeighting(nn.Module):

    def __init__(self, channels, ratio=16):
        super().__init__()

        self.channels = channels
        total_channel = sum(channels)

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=total_channel, out_channels=int(total_channel / ratio), kernel_size=1, stride=1, padding=0, bias=False),
                                                                nn.BatchNorm2d(int(total_channel / ratio)),
                                                                nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=int(total_channel / ratio), out_channels=total_channel, kernel_size=1, stride=1, padding=0, bias=False),
                                                                nn.BatchNorm2d(total_channel),
                                                                nn.Sigmoid())

    def forward(self, x):
        mini_size = x[-1].size()[-2:]
        out = [F.adaptive_avg_pool2d(s, mini_size) for s in x[:-1]] + [x[-1]]
        out = torch.cat(out, dim=1)
        out = self.conv1(out)
        out = self.conv2(out)
        out = torch.split(out, self.channels, dim=1)
        out = [s * F.interpolate(a, size=s.size()[-2:], mode='nearest') for s, a in zip(x, out)]
        return out


class ConditionalChannelWeighting(nn.Module):

    def __init__(self, in_channels, stride, reduce_ratio):
        super().__init__()
    
        self.stride = stride

        branch_channels = [channel // 2 for channel in in_channels]

        self.cross_resolution_weighting = CrossResolutionWeighting(branch_channels, ratio=reduce_ratio)

        self.depthwise_convs = nn.ModuleList([nn.Sequential(nn.Conv2d(in_channels=channel, out_channels= channel, kernel_size=3,
                                                                                                                                                stride=self.stride, padding=1, groups=channel, bias=False),
                                                                                                                        nn.BatchNorm2d(channel)) for channel in branch_channels])

        self.spatial_weighting = nn.ModuleList([SpatialWeighting(channels=channel, ratio=4) for channel in branch_channels])
        self.shuffle = Channel_Shuffle(2)

    def forward(self, x):
        x = [s.chunk(2, dim=1) for s in x]
        x1 = [s[0] for s in x]
        x2 = [s[1] for s in x]
        x2 = self.cross_resolution_weighting(x2)
        x2 = [dw(s) for s, dw in zip(x2, self.depthwise_convs)]
        x2 = [sw(s) for s, sw in zip(x2, self.spatial_weighting)]
        out = [torch.cat([s1, s2], dim=1) for s1, s2 in zip(x1, x2)]
        out = [self.shuffle(s) for s in out]

        return out


class Stem(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.in_channels = cfg.in_channels
        self.out_channels = cfg.out_channels

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=cfg.in_channels, out_channels=cfg.stem_channels, kernel_size=3, stride=2, padding=1, bias=False),
                                                                nn.BatchNorm2d(cfg.stem_channels),
                                                                nn.ReLU())
        branch_channels = cfg.stem_channels // 2
        mid_channels = int(round(cfg.stem_channels * cfg.expand_ratio))
        if cfg.stem_channels == self.out_channels:
            inc_channels = self.out_channels - branch_channels
        else:
            inc_channels = self.out_channels - cfg.stem_channels

        expand_out_channel = branch_channels if cfg.stem_channels == self.out_channels else cfg.stem_channels

    
        self.branch_batch = nn.Sequential(nn.Conv2d(in_channels=branch_channels, out_channels=branch_channels, kernel_size=3,
                                                                                                        stride=2, padding=1,  groups=branch_channels, bias=False),
                                                                                nn.BatchNorm2d(branch_channels),
                                                                                nn.Conv2d(in_channels=branch_channels, out_channels=inc_channels, kernel_size=1, stride=1, padding=0, bias=False),
                                                                                nn.BatchNorm2d(inc_channels),
                                                                                nn.ReLU())

        self.expand_path = nn.Sequential(nn.Conv2d(in_channels=branch_channels, out_channels=mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
                                                                                nn.BatchNorm2d(mid_channels),
                                                                                nn.ReLU(),
                                                                                nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=2, padding=1,
                                                                                                        groups=mid_channels, bias=False),
                                                                                nn.BatchNorm2d(mid_channels),
                                                                                nn.Conv2d(in_channels=mid_channels, out_channels=expand_out_channel, kernel_size=1, stride=1, padding=0, bias=False),
                                                                                nn.BatchNorm2d(expand_out_channel),
                                                                                nn.ReLU())

        self.shuffle = Channel_Shuffle(2)

    def forward(self, x):
        x = self.conv1(x)
        x1, x2 = x.chunk(2, dim=1)
        x2 = self.expand_path(x2)
        x1 = self.branch_batch(x1)
        out = torch.cat((x1, x2), dim=1)
        out = self.shuffle(out)
        return out


class IterativeHead(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        projects = []
        num_branchs = len(in_channels)
        self.in_channels = in_channels[::-1]

        for i in range(num_branchs):
            if i != num_branchs - 1:
                projects.append(nn.Sequential(nn.Conv2d(in_channels=self.in_channels[i], out_channels=self.in_channels[i], kernel_size=3, stride=1,
                                                                                                          padding=1, groups=self.in_channels[i], bias=False),
                                                                                 nn.BatchNorm2d(self.in_channels[i]),
                                                                                 nn.Conv2d(in_channels=self.in_channels[i], out_channels=self.in_channels[i + 1], kernel_size=1, stride=1, padding=0, bias=False),
                                                                                 nn.BatchNorm2d(self.in_channels[i + 1]),
                                                                                 nn.ReLU()))
            else:
                projects.append(nn.Sequential(nn.Conv2d(in_channels=self.in_channels[i], out_channels=self.in_channels[i], kernel_size=3, stride=1, padding=1,
                                                                                                          groups=self.in_channels[i], bias=False),
                                                                                    nn.BatchNorm2d(self.in_channels[i]),
                                                                                    nn.Conv2d(in_channels=self.in_channels[i], out_channels=self.in_channels[i], kernel_size=1, stride=1, padding=0, bias=False),
                                                                                    nn.BatchNorm2d(self.in_channels[i]),
                                                                                    nn.ReLU()))

        self.projects = nn.ModuleList(projects)

    def forward(self, x):
        x = x[::-1]
        y = []
        last_x = None

        for i, s in enumerate(x):
            if last_x is not None:
                last_x = F.interpolate(last_x, size=s.size()[-2:], mode='bilinear', align_corners=True)
                s = s + last_x
            s = self.projects[i](s)
            y.append(s)
            last_x = s

        return y[::-1]


class ShuffleUnit(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.stride = stride

        branch_features = out_channels // 2

        if self.stride > 1:
            self.branch1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=self.stride, padding=1,
                                                                                                groups=in_channels, bias=False),
                                                                        nn.BatchNorm2d(in_channels),
                                                                        nn.Conv2d(in_channels=in_channels, out_channels=branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                                                                        nn.BatchNorm2d(branch_features),
                                                                        nn.ReLU())

        self.branch2 = nn.Sequential(nn.Conv2d(in_channels=in_channels if self.stride > 1 else branch_features, out_channels=branch_features, kernel_size=1,
                                                                                            stride=1, padding=0, bias=False),
                                                                    nn.BatchNorm2d(branch_features),
                                                                    nn.ReLU(),
                                                                    nn.Conv2d(in_channels=branch_features, out_channels=branch_features, kernel_size=3, stride=self.stride, padding=1,
                                                                                            groups=branch_features, bias=False),
                                                                    nn.BatchNorm2d(branch_features),
                                                                    nn.Conv2d(in_channels=branch_features, out_channels=branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                                                                    nn.BatchNorm2d(branch_features),
                                                                    nn.ReLU())
        self.shuffle = Channel_Shuffle(2)

    def forward(self, x):
        if self.stride > 1:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        else:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        out = self.shuffle(out)
        return out


class LiteHRModule(nn.Module):

    def __init__(self, num_branches, num_blocks, in_channels, reduce_ratio, module_type, multiscale_output=False, with_fuse=True):
        super().__init__()
        self._check_branches(num_branches, in_channels)
    
        self.in_channels = in_channels
        self.num_branches = num_branches

        self.module_type = module_type
        self.multiscale_output = multiscale_output

        self.with_fuse = with_fuse

        if self.module_type == 'LITE':
            self.layers = self._make_weighting_blocks(num_blocks, reduce_ratio)
        elif self.module_type == 'NAIVE':
            self.layers = self._make_naive_branches(num_branches, num_blocks)

        if self.with_fuse:
            self.fuse_layers = self._make_fuse_layers()
            self.relu = nn.ReLU()

    def _check_branches(self, num_branches, in_channels):
        if num_branches != len(in_channels):
            error_msg = f'NUM_BRANCHES({num_branches}) != NUM_INCHANNELS({len(in_channels)})'
            raise ValueError(error_msg)

    def _make_weighting_blocks(self, num_blocks, reduce_ratio, stride=1):
        layers = []
        for i in range(num_blocks):
            layers.append(ConditionalChannelWeighting(self.in_channels,
                                                                                                            stride=stride,
                                                                                                            reduce_ratio=reduce_ratio))

        return nn.Sequential(*layers)

    def _make_one_branch(self, branch_index, num_blocks, stride=1):
        layers = []
        layers.append(ShuffleUnit(self.in_channels[branch_index],
                                                                self.in_channels[branch_index],
                                                                stride=stride))
        for i in range(1, num_blocks):
            layers.append(ShuffleUnit(self.in_channels[branch_index],
                                                                    self.in_channels[branch_index],
                                                                    stride=1))
        return nn.Sequential(*layers)

    def _make_naive_branches(self, num_branches, num_blocks):
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, num_blocks))
        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        in_channels = self.in_channels
        fuse_layers = []
        num_out_branches = num_branches if self.multiscale_output else 1

        for i in range(num_out_branches):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(nn.Conv2d(in_channels=in_channels[j], out_channels=in_channels[i], kernel_size=1, stride=1, padding=0, bias=False),
                                                                                            nn.BatchNorm2d(in_channels[i]),
                                                                                            nn.Upsample(scale_factor=2**(j - i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv_downsamples = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv_downsamples.append(nn.Sequential(nn.Conv2d(in_channels=in_channels[j], out_channels=in_channels[j], kernel_size=3,
                                                                                                                                               stride=2, padding=1, groups=in_channels[j], bias=False),
                                                                                                                       nn.BatchNorm2d(in_channels[j]),
                                                                                                                       nn.Conv2d(in_channels=in_channels[j], out_channels=in_channels[i], kernel_size=1,
                                                                                                                                              stride=1, padding=0, bias=False),
                                                                                                                       nn.BatchNorm2d(in_channels[i])))
                        else:
                            conv_downsamples.append(nn.Sequential(nn.Conv2d(in_channels=in_channels[j], out_channels=in_channels[j], kernel_size=3,
                                                                                                                                              stride=2, padding=1, groups=in_channels[j], bias=False),
                                                                                                                      nn.BatchNorm2d(in_channels[j]),
                                                                                                                      nn.Conv2d(in_channels=in_channels[j], out_channels=in_channels[j], kernel_size=1,
                                                                                                                                              stride=1, padding=0, bias=False),
                                                                                                                      nn.BatchNorm2d(in_channels[j]),
                                                                                                                      nn.ReLU()))
                    fuse_layer.append(nn.Sequential(*conv_downsamples))

            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def forward(self, x):
        if self.num_branches == 1:
            return [self.layers[0](x[0])]

        if self.module_type == 'LITE':
            out = self.layers(x)
        elif self.module_type == 'NAIVE':
            for i in range(self.num_branches):
                x[i] = self.layers[i](x[i])
            out = x

        if self.with_fuse:
            out_fuse = []
            for i in range(len(self.fuse_layers)):
                y = out[0] if i == 0 else self.fuse_layers[i][0](out[0])        # brach_index = 0会进行重复一次
                for j in range(self.num_branches):          # range(1, self.num_branches)
                    if i == j:
                        y += out[j]
                    else:
                        y += self.fuse_layers[i][j](out[j])
                out_fuse.append(self.relu(y))
            out = out_fuse
        elif not self.multiscale_output:
            out = [out[0]]

        return out


class LiteHRNet(NetBase):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.stem = Stem(cfg.stem)
        num_channels_last = [cfg.stem.out_channels]
        self.transitions = nn.ModuleList()
        self.stages = nn.ModuleList()
        for index in range(len(cfg.stages)):
            stage_cfg = cfg.stages[index]
            self.transitions.append(self._make_transition_layer(num_channels_last, stage_cfg.num_channels))
            stage, num_channels_last = self._make_stage(stage_cfg)
            self.stages.append(stage)

        if cfg.with_head:
            self.head_layer = IterativeHead(in_channels=num_channels_last)
    
    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []

        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(nn.Conv2d(in_channels=num_channels_pre_layer[i],
                                                                                                                                out_channels=num_channels_pre_layer[i],
                                                                                                                                kernel_size=3, stride=1, padding=1, groups=num_channels_pre_layer[i], bias=False),
                                                                                                        nn.BatchNorm2d(num_channels_pre_layer[i]),
                                                                                                        nn.Conv2d(in_channels=num_channels_pre_layer[i],
                                                                                                                                out_channels=num_channels_cur_layer[i],
                                                                                                                                kernel_size=1, stride=1, padding=0, bias=False),
                                                                                                        nn.BatchNorm2d(num_channels_cur_layer[i]),
                                                                                                        nn.ReLU()))
                else:
                    transition_layers.append(None)
            else:
                conv_downsamples = []
                for j in range(i + 1 - num_branches_pre):
                    in_channels = num_channels_pre_layer[-1]
                    out_channels = num_channels_cur_layer[i] if j == i - num_branches_pre else in_channels
                    conv_downsamples.append(nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3,
                                                                                                                                       stride=2, padding=1, groups=in_channels, bias=False),
                                                                                                                nn.BatchNorm2d(in_channels),
                                                                                                                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                                                                                                                nn.BatchNorm2d(out_channels),
                                                                                                                nn.ReLU()))
                transition_layers.append(nn.Sequential(*conv_downsamples))

        return nn.ModuleList(transition_layers)

    def _make_stage(self, stage_cfg):
        num_modules = stage_cfg.num_modules
        num_branches = len(stage_cfg.num_channels)
        num_blocks = stage_cfg.num_blocks
        reduce_ratio = stage_cfg.reduce_ratios
        with_fuse = stage_cfg.with_fuse
        module_type = stage_cfg.module_type
        in_channels = stage_cfg.num_channels
        multiscale_output = stage_cfg.multiscale_output

        modules = []

        for i in range(num_modules):
            if not multiscale_output and i == num_modules - 1:
                reset_multiscale_output = False
            else:
                reset_multiscale_output = True

            modules.append(LiteHRModule(num_branches,
                                                                                num_blocks,
                                                                                in_channels,
                                                                                reduce_ratio,
                                                                                module_type,
                                                                                multiscale_output=reset_multiscale_output,
                                                                                with_fuse=with_fuse))
    
        in_channels = modules[-1].in_channels

        return nn.Sequential(*modules), in_channels

    def forward(self, x):
        x = self.stem(x)

        y_list = [x]
        for index in range(len(self.cfg.stages)):
            x_list = []
            transition = self.transitions[index]
            for j in range(len(self.cfg.stages[index].num_channels)):
                if transition[j]:
                    if j >= len(y_list):
                        x_list.append(transition[j](y_list[-1]))
                    else:
                        x_list.append(transition[j](y_list[j]))
                else:
                    x_list.append(y_list[j])
            y_list = self.stages[index](x_list)

        x = y_list
    
        if self.cfg.with_head:
            x = self.head_layer(x)

        return x[0]
