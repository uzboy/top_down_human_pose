from data_process.heatmaps.udp_heatmaps import UDPHeatmap
from data_process.heatmaps.msra_heatmap import MSRAHeatmap
from data_process.heatmaps.megvii_heatmap import MegviiHeatmap


HEATMAPS = {
    "UDPHeatmap": UDPHeatmap,
    "MSRAHeatmap": MSRAHeatmap,
    "MegviiHeatmap": MegviiHeatmap
}


def build_heatmaps(cfgs):
    heatmaps = []
    for index in range(len(cfgs)):
        heatmaps.append(HEATMAPS[cfgs[index].name](cfgs[index]))

    return heatmaps
