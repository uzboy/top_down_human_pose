import os
from utils.utils import load_checkpoint
from models.head.topdown_heatmap_msmu_head import TopdownHeatmapMSMUHead
from models.head.topdown_heatmap_simple_head import TopdownHeatmapSimpleHead
from models.head.topdown_heatmap_ms_head import TopdownHeatmapMultiStageHead
from models.head.topdown_regression_blaze_head import TopdownRegressionBlazeHead
from models.head.topdown_regression_simple_head import TopdownRegressionSimpleHead


HEADS = {
    "TopdownHeatmapMSMUHead": TopdownHeatmapMSMUHead,
    "TopdownHeatmapSimpleHead": TopdownHeatmapSimpleHead,
    "TopdownHeatmapMultiStageHead": TopdownHeatmapMultiStageHead,
    "TopdownRegressionBlazeHead":TopdownRegressionBlazeHead,
    "TopdownRegressionSimpleHead":TopdownRegressionSimpleHead,
}


def build_head(cfg):
    model = HEADS[cfg.name](cfg)
    try:
        resum_path = cfg.resum_path
    except:
        resum_path = None

    if resum_path is not None and os.path.exists(resum_path):
        load_checkpoint(model, resum_path)
    else:
        model.init_weight()

    try:
        is_freeze = cfg.is_freeze
    except:
        is_freeze = False

    if is_freeze:
        model.freeze_model()

    return model
