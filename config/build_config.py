from config.mobile_v2_config import config as MobileNetV2
from config.mobile_v2_bce_config import config as MobileNetV2BCE
from config.mobile_v2_blaze_pose import config as MobileNetV2Blaze


CONFIGS = {
    "MobileNetV2":MobileNetV2,
    "MobileNetV2BCE":MobileNetV2BCE,
    "MobileNetV2Blaze":MobileNetV2Blaze
}


def build_config(cfg_name):
    return CONFIGS[cfg_name]
