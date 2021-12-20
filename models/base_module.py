from abc import abstractmethod
import torch.nn as nn


class NetBase(nn.Module):
    def __init__(self):
        super(NetBase, self).__init__()

    @abstractmethod
    def forward(self, x):
        ""

    @abstractmethod
    def init_weight(self):
        ""

