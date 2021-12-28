import torch.nn as nn
from models.actions.build_actions import ACT_NAME_MAPS


class SELayer(nn.Module):

    def __init__(self, channels, ratio=16, act_names=['ReLU', 'Sigmoid']):
        super(SELayer, self).__init__()

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        middle_channel = int(channels / ratio)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=channels, out_channels=middle_channel, kernel_size=1, stride=1, padding=0),
                                                                ACT_NAME_MAPS[act_names[0]]())
        
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=middle_channel, out_channels=channels, kernel_size=1, stride=1, padding=0),
                                                                ACT_NAME_MAPS[act_names[1]]())

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.conv2(out)
        return x * out
