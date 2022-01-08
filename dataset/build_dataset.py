from dataset.coco_data_with_box import CocoDataWithBox
from dataset.coco_data_regression import CocoDataRegression


DATASETS = {
    "CocoDataWithBox": CocoDataWithBox,
    "CocoDataRegression":CocoDataRegression
}


def build_dataset(cfg):
        return DATASETS[cfg.name](cfg)
