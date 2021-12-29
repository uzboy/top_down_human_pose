import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelGate(nn.Module):

    def __init__(self, gate_channel, reduction_ratio=16, num_layers=1):
        super(ChannelGate, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        gate_channels = [gate_channel]
        gate_channels += [gate_channel // reduction_ratio] * num_layers
        gate_channels += [gate_channel]

        self.gate_c = []
        for i in range( len(gate_channels) - 2 ):
            self.gate_c.append(nn.Sequential(nn.Conv2d(in_channels=gate_channels[i], out_channels=gate_channels[i + 1], kernel_size=1, stride=1, padding=0, bias=False),
                                                                                    nn.BatchNorm2d(gate_channels[i + 1]),
                                                                                    nn.ReLU(inplace=True)))
        self.gate_c.append(nn.Conv2d(in_channels=gate_channels[-2], out_channels=gate_channels[-1], kernel_size=1, stride=1, padding=0))
        self.gate_c = nn.Sequential(*self.gate_c)

    def forward(self, x):
        avg_pool = self.avg_pool(x)
        return self.gate_c(avg_pool).expand_as(x)


class SpatialGate(nn.Module):

    def __init__(self, gate_channel, reduction_ratio=16, dilation_conv_num=2, dilation_val=4):
        super(SpatialGate, self).__init__()
        self.gate_s = []
        self.gate_s.append(nn.Sequential(nn.Conv2d(in_channels=gate_channel,
                                                                                                       out_channels=gate_channel //reduction_ratio,
                                                                                                       kernel_size=1, stride=1, padding=0, bias=False),
                                                                                nn.BatchNorm2d(gate_channel // reduction_ratio),
                                                                                nn.ReLU(inplace=True)))

        for i in range( dilation_conv_num ):
            self.gate_s.append(nn.Sequential(nn.Conv2d(in_channels=gate_channel // reduction_ratio,
                                                                                                           out_channels=gate_channel // reduction_ratio,
                                                                                                           kernel_size=3, stride=1, padding=dilation_val, dilation=dilation_val, bias=False),
                                                                                    nn.BatchNorm2d(gate_channel // reduction_ratio),
                                                                                    nn.ReLU(inplace=True)))

        self.gate_s.append(nn.Conv2d(in_channels=gate_channel // reduction_ratio, out_channels=1, kernel_size=1, stride=1, padding=0))
        self.gate_s = nn.Sequential(*self.gate_s)

    def forward(self, x):
        return self.gate_s(x).expand_as(x)


class BAM(nn.Module):

    def __init__(self, gate_channel):
        super(BAM, self).__init__()
        self.channel_att = ChannelGate(gate_channel)
        self.spatial_att = SpatialGate(gate_channel)

    def forward(self,in_tensor):
        att = 1 + F.sigmoid( self.channel_att(in_tensor) * self.spatial_att(in_tensor) )
        return att * in_tensor
