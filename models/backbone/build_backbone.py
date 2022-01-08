import os
from utils.utils import load_checkpoint
from models.backbone.mobilenet_v2 import MobileNetV2



BACKBONES = {
    "MobileNetV2":MobileNetV2,
}


def build_backbone(cfg):
    model = BACKBONES[cfg.name](cfg)
    resum_path = cfg.get("resum_path", None)
    if resum_path is not None and os.path.exists(resum_path):
        load_checkpoint(model, resum_path)
    else:
        model.init_weight()

    frozen_stages = cfg.get("frozen_stages", None)
    if frozen_stages is not None:
        model.freeze_model()

    return model
