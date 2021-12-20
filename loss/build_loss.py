from loss.joint_mse import JointsMSELoss
from loss.joint_ohk_mse import JointsOHKMMSELoss
from loss.combined_target_mse import CombinedTargetMSELoss


LOSS = {
    "JointsMSELoss": JointsMSELoss,
    "JointsOHKMMSELoss": JointsOHKMMSELoss,
    "CombinedTargetMSELoss": CombinedTargetMSELoss
}


def build_loss(cfg):
    return LOSS[cfg.name](cfg)
