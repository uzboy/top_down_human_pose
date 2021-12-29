# GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond
from torch import nn as nn
import torch.nn.functional as F
from utils.utils import make_divisible
from models.normal.normal import LayerNorm2d


class GlobalContext(nn.Module):

    def __init__(self, channels, use_attn=True, fuse_add=False, fuse_scale=True, ratio=8, rd_divisor=1):
        super(GlobalContext, self).__init__()
        self.conv_attn = nn.Conv2d(channels, 1, kernel_size=1, bias=True) if use_attn else None
        rd_channels = make_divisible(channels / ratio, rd_divisor)

        if fuse_add:
            self.mlp_add = nn.Sequential(nn.Conv2d(in_channels=channels, out_channels=rd_channels, kernel_size=1, stride=1, padding=0, bias=False),
                                                                          LayerNorm2d(rd_channels),
                                                                          nn.ReLU(inplace=True),
                                                                          nn.Conv2d(in_channels=rd_channels, out_channels=channels, kernel_size=1, stride=1, padding=0))
        else:
            self.mlp_add = None
    
        if fuse_scale:
            self.mlp_add = nn.Sequential(nn.Conv2d(in_channels=channels, out_channels=rd_channels, kernel_size=1, stride=1, padding=0, bias=False),
                                                                          LayerNorm2d(rd_channels),
                                                                          nn.ReLU(inplace=True),
                                                                          nn.Conv2d(in_channels=rd_channels, out_channels=channels, kernel_size=1, stride=1, padding=0))
        else:
            self.mlp_scale = None

        self.gate = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape

        attn = self.conv_attn(x).reshape(B, 1, H * W)
        attn = F.softmax(attn, dim=-1).unsqueeze(3)
        context = x.reshape(B, C, H * W).unsqueeze(1) @ attn
        context = context.view(B, C, 1, 1)

        if self.mlp_scale is not None:
            mlp_x = self.mlp_scale(context)
            x = x * self.gate(mlp_x)
    
        if self.mlp_add is not None:
            mlp_x = self.mlp_add(context)
            x = x + mlp_x

        return x
