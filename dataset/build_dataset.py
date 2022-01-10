from dataset.coco_data_with_box import CocoDataWithBox
from dataset.coco_data_regression import CocoDataRegression
from dataset.coco_data_regression_and_heatmap import CocoDataRegressionAndHeatmap


DATASETS = {
    "CocoDataWithBox": CocoDataWithBox,
    "CocoDataRegression":CocoDataRegression,
    "CocoDataRegressionAndHeatmap":CocoDataRegressionAndHeatmap
}


def build_dataset(cfg):
        return DATASETS[cfg.name](cfg)
