import torch
import functools
import torch.nn as nn
import torch.functional as F


class Swish(nn.Module):

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class Clamp(nn.Module):

    def __init__(self, min=-1., max=1.):
        super(Clamp, self).__init__()
        self.min = min
        self.max = max

    def forward(self, x):
        return torch.clamp(x, min=self.min, max=self.max)


class GELU(nn.Module):

    def forward(self, input):
        return F.gelu(input)


class HSigmoid(nn.Module):

    def __init__(self, bias=1.0, divisor=2.0, min_value=0.0, max_value=1.0):
        super(HSigmoid, self).__init__()
        self.bias = bias
        self.divisor = divisor
        assert self.divisor != 0
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x):
        x = (x + self.bias) / self.divisor

        return x.clamp_(self.min_value, self.max_value)


class HSwish(nn.Module):

    def __init__(self, inplace=False):
        super(HSwish, self).__init__()
        self.act = nn.ReLU6(inplace)

    def forward(self, x):
        return x * self.act(x + 3) / 6


ACT_NAME_MAPS = {
    "Tanh":nn.Tanh,
    "PReLU": nn.PReLU,
    "Sigmoid":nn.Sigmoid,
    "ELU": functools.partial(nn.ELU, inplace=True),
    "ReLU": functools.partial(nn.ReLU, inplace=True),
    "ReLU6": functools.partial(nn.ReLU6, inplace=True),
    "RReLU": functools.partial(nn.RReLU, inplace=True),
    "LeakyReLU": functools.partial(nn.LeakyReLU, inplace=True),
    "HSwish":functools.partial(HSwish, inplace=True),
    "HSigmoid":HSigmoid,
    "GELU":GELU,
    "Clamp":Clamp,
    "Swish":Swish
}

