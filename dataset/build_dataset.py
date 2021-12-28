from dataset.coco_data_for_eval import CocoDataEval
from dataset.coco_data_with_mutex import CocoDataWithMutex
from dataset.coco_data_regression import CocoDataRegression


DATASETS = {
    "CocoDataEval": CocoDataEval,
    "CocoDataWithMutex": CocoDataWithMutex,
    "CocoDataRegression":CocoDataRegression
}


def build_dataset(cfg):
        return DATASETS[cfg.name](cfg)
