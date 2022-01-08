import os
import torch.nn as nn
from utils.utils import save_checkpoint
from models.head.build_head import build_head
from models.backbone.build_backbone import build_backbone


class Model(nn.Module):

    def __init__(self, cfg):
        super(Model, self).__init__()
        self.backbone = build_backbone(cfg.backbone)
        self.head = build_head(cfg.head)

        self.save_backbone_prev = cfg.backbone.get("pre_name", None)
        self.save_backbone_base = cfg.backbone.get("base_path", None)
        if self.save_backbone_base is not None and not os.path.exists(self.save_backbone_base):
            os.makedirs(self.save_backbone_base)
    
        self.save_head_prev = cfg.head.get("pre_name", None)
        self.save_head_base = cfg.head.get("base_path", None)
        if self.save_head_base is not None and not os.path.exists(self.save_head_base):
            os.makedirs(self.save_head_base)

    def save_ckps(self, epoch_index):
        if self.save_backbone_base is not None and self.save_backbone_prev is not None:
            save_file_name = os.path.join(self.save_backbone_base, self.save_backbone_prev + "_%d.pth" % (epoch_index + 1))
            save_checkpoint(self.backbone, save_file_name)

        if self.save_head_base is not None and self.save_head_prev is not None:
            save_file_name = os.path.join(self.save_head_base, self.save_head_prev + "_%d.pth" % (epoch_index + 1))
            save_checkpoint(self.head, save_file_name)

    def get_loss(self, loss_inputs):
        return self.head.get_loss(loss_inputs)

    def forward(self, inputs):
        features = self.backbone(inputs)
        outs = self.head(features)
        return outs


def build_model(cfg):
    model = Model(cfg)
    us_multi_gpus = cfg.get("us_multi_gpus", False)
    if us_multi_gpus:
        assert hasattr(cfg, "gup_ids")
        assert hasattr(cfg, "device")
        model = nn.DataParallel(model, device_ids = cfg.gup_ids)
        model = model.to(cfg.device)
    else:
        model = model.to(cfg.get("device", "cpu"))

    return model
