import os
from utils.utils import load_checkpoint
from models.backbone.rsn import RSN
from models.backbone.mspn import MSPN
from models.backbone.scnet import SCNet
from models.backbone.resnet import ResNet
from models.backbone.resnest import ResNeSt
from models.backbone.litehrnet import LiteHRNet
from models.backbone.mobilenet_v2 import MobileNetV2
from models.backbone.mobilenet_v3 import MobileNetV3


BACKBONES = {
    "LiteHRNet": LiteHRNet,
    "SCNet": SCNet,
    "RSN":RSN,
    "MSPN": MSPN,
    "ResNeSt": ResNeSt,
    "ResNet":ResNet,
    "MobileNetV2":MobileNetV2,
    "MobileNetV3":MobileNetV3
}


def build_backbone(cfg):
    model = BACKBONES[cfg.name](cfg)
    if cfg.resum_path is not None and os.path.exists(cfg.resum_path):
        load_checkpoint(model, cfg.resum_path)
    else:
        model.init_weight()

    if cfg.frozen_stages is not None:
        model.freeze_model()

    return model
