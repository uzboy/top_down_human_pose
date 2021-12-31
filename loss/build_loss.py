from loss.joint_mse import JointsMSELoss
from loss.joint_ohk_mse import JointsOHKMMSELoss
from loss.combined_target_mse import CombinedTargetMSELoss
from loss.regression_l1_loss import L1Loss, SmoothL1Loss, MPJPELoss
from loss.regression_l2_loss import MSELoss
from loss.regression_wing_loss import WingLoss
from loss.joint_mse_with_weight import JointsMSEWithWeightLoss


LOSS = {
    "JointsMSELoss": JointsMSELoss,
    "JointsOHKMMSELoss": JointsOHKMMSELoss,
    "CombinedTargetMSELoss": CombinedTargetMSELoss,
    "L1Loss":L1Loss,
    "SmoothL1Loss":SmoothL1Loss,
    "MPJPELoss": MPJPELoss,
    "MSELoss":MSELoss,
    "WingLoss":WingLoss,
    "JointsMSEWithWeightLoss":JointsMSEWithWeightLoss,
}

def build_loss(cfg):
    return LOSS[cfg.name](cfg)
