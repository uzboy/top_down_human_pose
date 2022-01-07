import os
from utils.utils import load_checkpoint
from models.head.topdown_regression_blaze_head import TopdownRegressionBlazeHead
from models.head.topdown_regression_simple_head import TopdownRegressionSimpleHead


HEADS = {
    "TopdownRegressionBlazeHead":TopdownRegressionBlazeHead,
    "TopdownRegressionSimpleHead":TopdownRegressionSimpleHead,
}


def build_head(cfg):
    model = HEADS[cfg.name](cfg)
    if hasattr(cfg, "resum_path"):
        resum_path = cfg.resum_path
    else:
        resum_path = None

    if resum_path is not None and os.path.exists(resum_path):
        load_checkpoint(model, resum_path)
    else:
        model.init_weight()

    if hasattr(cfg, "is_freeze") and cfg.is_freeze:
        model.freeze_model()

    return model
