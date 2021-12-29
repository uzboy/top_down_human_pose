import functools
import torch.nn as nn
from models.actions.actions import HSigmoid, HSwish, GELU, Clamp, Swish


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
    "HSigmoid":functools.partial(HSigmoid, inplace=True),
    "GELU":GELU,
    "Clamp":Clamp,
    "Swish":Swish
}

