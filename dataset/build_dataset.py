from dataset.coco_data_for_eval import CocoDataEval
from dataset.coco_data_with_mutex import CocoDataWithMutex


DATASETS = {
    "CocoDataEval": CocoDataEval,
    "CocoDataWithMutex": CocoDataWithMutex
}


def build_dataset(cfg):
        return DATASETS[cfg.name](cfg)
