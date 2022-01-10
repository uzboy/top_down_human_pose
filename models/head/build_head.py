import os
from utils.utils import load_checkpoint
from models.head.topdown_heatmap_simple_head import TopdownHeatmapSimpleHead
from models.head.topdown_regression_blaze_head import TopdownRegressionBlazeHead
from models.head.topdown_heatmap_and_regression_head import TopdownRegressionAndHeatmap


HEADS = {
    "TopdownHeatmapSimpleHead":TopdownHeatmapSimpleHead,
    "TopdownRegressionBlazeHead":TopdownRegressionBlazeHead,
    "TopdownRegressionAndHeatmap":TopdownRegressionAndHeatmap
}


def build_head(cfg):
    model = HEADS[cfg.name](cfg)
    resum_path = cfg.get("resum_path", None)
    if resum_path is not None and os.path.exists(resum_path):
        load_checkpoint(model, resum_path)
    else:
        model.init_weight()

    is_freeze = cfg.get("is_freeze", False)
    if is_freeze:
        model.freeze_model()

    return model
