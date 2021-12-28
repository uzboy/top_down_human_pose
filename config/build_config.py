from config.mobile_v2_config import config as MobileNetV2
from config.mobile_v3_config import config as MobileNetV3
from resnet_50_config import config as ResNet50
from resnet_50_se_config import config as ResNetSe50
from resnext_50_config import config as ResNext50
from resnext_50_se_config import config as ResNextSe50
from config.rsn_config import config as RSN
from config.scnet_config import config as SCNet
from config.mspn_config import config as MSPN
from config.resnest_config import config as ResNeSt
from config.litehrnet_config import config as LiteHRNet



CONFIGS = {
    "MobileNetV2":MobileNetV2,
    "MobileNetV3":MobileNetV3,
    "ResNet50":ResNet50,
    "ResNetSe50":ResNetSe50,
    "ResNext50":ResNext50,
    "ResNextSe50":ResNextSe50,
    "RSN":RSN,
    "SCNet":SCNet,
    "MSPN":MSPN,
    "ResNeSt":ResNeSt,
    "LiteHRNet":LiteHRNet,
}


def build_config(cfg_name):
    return CONFIGS[cfg_name]

