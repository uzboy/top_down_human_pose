import os
import torch.nn as nn
from utils.utils import load_checkpoint
from models.head.build_head import build_head
from models.backbone.build_backbone import build_backbone


class Model(nn.Module):

    def __init__(self, cfg):
        super(Model, self).__init__()
        self.backbone = build_backbone(cfg.backbone)
        self.head = build_head(cfg.head)
        resum_path = None
        try:
            resum_path = cfg.resum_path
        except:
            resum_path = None

        self.init_weights(resum_path)

    def init_weights(self, resum_path):
        if resum_path is not None and os.path.exists(resum_path):
            load_checkpoint(self, resum_path)
        else:
            self.backbone.init_weight()
            self.head.init_weight()

    def get_loss(self, loss_inputs):
        return self.head.get_loss(loss_inputs)

    def forward(self, inputs):
        features = self.backbone(inputs)
        outs = self.head(features)
        return outs


def build_model(cfg):
    model = Model(cfg)
    if cfg.us_multi_gpus:
        model = nn.DataParallel(model, device_ids = cfg.gup_ids)
        model = model.to(cfg.device)
    else:
        model = model.to(cfg.device)

    return model
