from config.rsn_config import config as RSN
from config.scnet_config import config as SCNet
from config.mspn_config import config as MSPN
from config.resnet_config import config as ResNet
from config.resnext_config import config as ResNext
from config.resnest_config import config as ResNeSt
from config.litehrnet_config import config as LiteHRNet
from config.se_resnet_config import config as SE_ResNet
from config.se_resnext_config import config as SE_ResNext


CONFIGS = {
    "RSN":RSN,
    "SCNet":SCNet,
    "MSPN":MSPN,
    "ResNet":ResNet,
    "ResNext":ResNext,
    "ResNeSt":ResNeSt,
    "LiteHRNet":LiteHRNet,
    "SE_ResNet":SE_ResNet,
    "SE_ResNext":SE_ResNext
}


def build_config(cfg_name):
    return CONFIGS[cfg_name]

