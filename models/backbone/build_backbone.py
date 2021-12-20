import os
from utils.utils import load_checkpoint
from models.backbone.rsn import RSN
from models.backbone.mspn import MSPN
from models.backbone.scnet import SCNet
from models.backbone.resnet import ResNet
from models.backbone.resnest import ResNeSt
from models.backbone.resnext import ResNeXt
from models.backbone.litehrnet import LiteHRNet
from models.backbone.se_resnet import ResNet_SE
from models.backbone.se_resnext import ResNeXt_SE


BACKBONES = {
    "LiteHRNet": LiteHRNet,
    "SCNet": SCNet,
    "RSN":RSN,
    "MSPN": MSPN,
    "ResNeSt": ResNeSt,
    "ResNeXt":ResNeXt,
    "ResNet":ResNet,
    "ResNet_SE":ResNet_SE,
    "ResNeXt_SE":ResNeXt_SE
}


def build_backbone(cfg):
    model = BACKBONES[cfg.name](cfg)
    try:
        resum_path = cfg.resum_path
    except:
        resum_path = None

    if resum_path is not None and os.path.exists(resum_path):
        load_checkpoint(model, resum_path)
    else:
        model.init_weight()

    try:
        is_freeze = cfg.is_freeze
    except:
        is_freeze = False

    if is_freeze:
        model.freeze_model()

    return model
