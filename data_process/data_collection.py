import torch
import numpy as np


def data_regression_with_mask_collect_func(batch):
    images = torch.cat([batch[index][0].unsqueeze(0) for index in range(len(batch))], dim=0)
    regression_targets = np.concatenate([batch[index][1][None, :] for index in range(len(batch))], axis=0)
    regression_weight = np.concatenate([batch[index][2][None, :] for index in range(len(batch))], axis=0)
    mask_targets = np.concatenate([batch[index][3][None, :] for index in range(len(batch))], axis=0)

    return images,\
                  (torch.from_numpy(regression_targets), torch.from_numpy(mask_targets)),\
                  (torch.from_numpy(regression_weight), None)


def data_regression_and_heatmap_collect_func(batch):
    images = torch.cat([batch[index][0].unsqueeze(0) for index in range(len(batch))], dim=0)
    heatmap_targets = np.concatenate([batch[index][1][None, :] for index in range(len(batch))], axis=0)
    heatmap_target_weights = np.concatenate([batch[index][2][None, :] for index in range(len(batch))], axis=0)
    regression_target = np.concatenate([batch[index][3][None, :] for index in range(len(batch))], axis=0)
    regression_target_weight = np.concatenate([batch[index][4][None, :] for index in range(len(batch))], axis=0)
    return images, \
                  (torch.from_numpy(heatmap_targets), torch.from_numpy(regression_target)), \
                  (torch.from_numpy(heatmap_target_weights), torch.from_numpy(regression_target_weight))


def data_regression_and_heatmap_with_mask_collect_func(batch):
    images = torch.cat([batch[index][0].unsqueeze(0) for index in range(len(batch))], dim=0)
    heatmap_targets = np.concatenate([batch[index][1][None, :] for index in range(len(batch))], axis=0)
    heatmap_target_weights = np.concatenate([batch[index][2][None, :] for index in range(len(batch))], axis=0)
    regression_target = np.concatenate([batch[index][3][None, :] for index in range(len(batch))], axis=0)
    regression_target_weight = np.concatenate([batch[index][4][None, :] for index in range(len(batch))], axis=0)
    mask_targets = np.concatenate([batch[index][5][None, :] for index in range(len(batch))], axis=0)

    return images\
                  (torch.from_numpy(heatmap_targets), torch.from_numpy(regression_target), torch.from_numpy(mask_targets)),\
                  (torch.from_numpy(heatmap_target_weights), torch.from_numpy(regression_target_weight), None)


COLLECTIONS = {
    "data_regression_with_mask_collect_func":data_regression_with_mask_collect_func,
    "data_regression_and_heatmap_collect_func":data_regression_and_heatmap_collect_func,
    "data_regression_and_heatmap_with_mask_collect_func":data_regression_and_heatmap_with_mask_collect_func
}


def build_collect_func(collect_name):
    return COLLECTIONS[collect_name]
