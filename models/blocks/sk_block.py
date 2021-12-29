import torch
from torch import nn as nn
from utils.utils import make_divisible


def get_padding(kernel_size, stride, dilation):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


class SelectiveKernelAttn(nn.Module):

    def __init__(self, channels, num_paths=2, attn_channels=32):
        super(SelectiveKernelAttn, self).__init__()
        self.num_paths = num_paths
        self.fc_reduce = nn.Conv2d(channels, attn_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(attn_channels)
        self.act = nn.ReLU(inplace=True)
        self.fc_select = nn.Conv2d(attn_channels, channels * num_paths, kernel_size=1, bias=False)

    def forward(self, x):
        x = x.sum(1).mean((2, 3), keepdim=True)
        x = self.fc_reduce(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.fc_select(x)
        B, C, H, W = x.shape
        x = x.view(B, self.num_paths, C // self.num_paths, H, W)
        x = torch.softmax(x, dim=1)
        return x


class SelectiveKernel(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=None, stride=1, dilation=1, groups=1, ratio=16, keep_3x3=True, split_input=True):
        super(SelectiveKernel, self).__init__()
        kernel_size = kernel_size or [3, 5]
        if not isinstance(kernel_size, list):
            kernel_size = [kernel_size] * 2

        if keep_3x3:
            dilation = [dilation * (k - 1) // 2 for k in kernel_size]
            kernel_size = [3] * len(kernel_size)
        else:
            dilation = [dilation] * len(kernel_size)

        self.num_paths = len(kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.split_input = split_input
        if self.split_input:
            in_channels = in_channels // self.num_paths

        groups = min(out_channels, groups)
    
        self.paths = nn.ModuleList()
        for index in len(kernel_size):
            k_size = kernel_size[index]
            dila = dilation[index]
            pad = get_padding(kernel_size, stride, dilation)
            self.paths.append(nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                                                                         out_channels=out_channels,
                                                                                                         kernel_size=k_size,
                                                                                                         stride=stride,
                                                                                                         padding=pad,
                                                                                                         dilation=dila,
                                                                                                         groups=groups,
                                                                                                         bias=False),
                                                                                 nn.BatchNorm2d(out_channels),
                                                                                 nn.ReLU(inplace=True)))

        attn_channels =  make_divisible(out_channels // ratio, divisor=8)
        self.attn = SelectiveKernelAttn(out_channels, self.num_paths, attn_channels)

    def forward(self, x):
        if self.split_input:
            x_split = torch.split(x, self.in_channels // self.num_paths, 1)
            x_paths = [op(x_split[i]) for i, op in enumerate(self.paths)]
        else:
            x_paths = [op(x) for op in self.paths]
        x = torch.stack(x_paths, dim=1)
        x_attn = self.attn(x)
        x = x * x_attn
        x = torch.sum(x, dim=1)
        return x
