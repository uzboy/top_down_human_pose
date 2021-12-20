import numpy as np


def transform_preds(coords, center, scale, output_size):

    scale = scale * 200.0
    scale_x = scale[0] / output_size[0]
    scale_y = scale[1] / output_size[1]

    target_coords = np.ones_like(coords)
    target_coords[:, 0] = coords[:, 0] * scale_x + center[0] - scale[0] * 0.5
    target_coords[:, 1] = coords[:, 1] * scale_y + center[1] - scale[1] * 0.5

    return target_coords


def _get_max_preds(heatmaps):
    N, K, _, W = heatmaps.shape
    heatmaps_reshaped = heatmaps.reshape((N, K, -1))
    idx = np.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
    maxvals = np.amax(heatmaps_reshaped, 2).reshape((N, K, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = preds[:, :, 0] % W
    preds[:, :, 1] = preds[:, :, 1] // W

    preds = np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)

    return preds, maxvals


def decode_heat_map(heatmaps, center, scale):
    heatmaps = heatmaps.copy()
    N, K, H, W = heatmaps.shape
    preds, maxvals = _get_max_preds(heatmaps)
    for n in range(N):
        for k in range(K):
            heatmap = heatmaps[n][k]
            px = int(preds[n][k][0])
            py = int(preds[n][k][1])
            if 1 < px < W - 1 and 1 < py < H - 1:
                diff = np.array([heatmap[py][px + 1] - heatmap[py][px - 1],
                                                heatmap[py + 1][px] - heatmap[py - 1][px]])
                preds[n][k] += np.sign(diff) * .25

    for i in range(N):
        preds[i] = transform_preds(preds[i], center, scale, [W, H])

    return preds, maxvals

