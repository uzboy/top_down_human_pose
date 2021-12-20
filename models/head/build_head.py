from models.head.topdown_heatmap_msmu_head import TopdownHeatmapMSMUHead
from models.head.topdown_heatmap_simple_head import TopdownHeatmapSimpleHead
from models.head.topdown_heatmap_ms_head import TopdownHeatmapMultiStageHead


HEADS = {
    "TopdownHeatmapMSMUHead": TopdownHeatmapMSMUHead,
    "TopdownHeatmapSimpleHead": TopdownHeatmapSimpleHead,
    "TopdownHeatmapMultiStageHead": TopdownHeatmapMultiStageHead
}


def build_head(cfg):
    return HEADS[cfg.name](cfg)

