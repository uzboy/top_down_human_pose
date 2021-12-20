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
    return BACKBONES[cfg.name](cfg)
