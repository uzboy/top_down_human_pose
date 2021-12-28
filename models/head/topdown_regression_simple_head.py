import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_module import NetBase
from loss.build_loss import build_loss
from utils.utils import normal_init, constant_init


class TopdownRegressionSimpleHead(NetBase):

    def __init__(self, cfg):
        super(TopdownRegressionSimpleHead, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.out_layers = nn.Linear(cfg.input_channels, cfg.num_joints * 2)
        self.loss = None
        try:
            if cfg.loss is not None:
                self.loss = build_loss(cfg.loss)
        except:
            pass

    def init_weight(self):
        normal_init(self.out_layers, mean=0, std=0.01, bias=0)

    def get_loss(self, loss_inputs):
        input, target, target_weight = loss_inputs
        assert self.loss != None, "No Loss Function......."
        loss = self.loss(input, target, target_weight)
        return loss

    def forward(self, x):
        x = self.avg_pool(x)
        N, C = x.shape
        x = x.view(N, -1)
        return self.out_layers(x)
