from loss.bce_loss import JointsBCELoss
from loss.bce_loss import VisMaskBCELoss
from loss.joint_mse import JointsMSELoss
from loss.regression_l2_loss import MSELoss
from loss.joint_mse_with_weight import JointsMSEWithWeightLoss
from loss.regression_l1_loss import L1Loss, SmoothL1Loss, MPJPELoss


LOSS = {
    "JointsMSELoss": JointsMSELoss,
    "L1Loss":L1Loss,
    "SmoothL1Loss":SmoothL1Loss,
    "MPJPELoss": MPJPELoss,
    "MSELoss":MSELoss,
    "JointsMSEWithWeightLoss":JointsMSEWithWeightLoss,
    "JointsBCELoss":JointsBCELoss,
    "VisMaskBCELoss":VisMaskBCELoss
}

def build_loss(cfg):
    return LOSS[cfg.name](cfg)
