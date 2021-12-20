import numpy as np


class MSRAHeatmap:

    def __init__(self, cfg):
        self.sigma = cfg.sigma
        self.unbiased_encoding = cfg.unbiased_encoding
        self.num_joints = cfg.num_joints
        self.image_size = cfg.image_size
        self.heatmap_size = cfg.heatmap_size
        self.joint_weights = cfg.joint_weights
        self.use_different_joint_weights = cfg.use_different_joint_weights

    def __call__(self, joints):
        W, H = self.heatmap_size
        target_weight = np.zeros((self.num_joints, 1), dtype=np.float32)
        target = np.zeros((self.num_joints, H, W), dtype=np.float32)

        tmp_size = self.sigma * 3

        if self.unbiased_encoding:
            for joint_id in range(self.num_joints):
                target_weight[joint_id] = joints[joint_id, -1]

                feat_stride = self.image_size / [W, H]
                mu_x = joints[joint_id][0] / feat_stride[0]
                mu_y = joints[joint_id][1] / feat_stride[1]

                ul = [mu_x - tmp_size, mu_y - tmp_size]
                br = [mu_x + tmp_size + 1, mu_y + tmp_size + 1]
                if ul[0] >= W or ul[1] >= H or br[0] < 0 or br[1] < 0:
                    target_weight[joint_id] = 0

                if target_weight[joint_id] == 0:
                    continue

                x = np.arange(0, W, 1, np.float32)
                y = np.arange(0, H, 1, np.float32)
                y = y[:, None]

                if target_weight[joint_id] > 0.5:
                    target[joint_id] = np.exp(-((x - mu_x) ** 2 + (y - mu_y) ** 2) / (2 * self.sigma ** 2))
        else:
            for joint_id in range(self.num_joints):
                target_weight[joint_id] = joints[joint_id, -1]
                feat_stride = [self.image_size[0] / W,  self.image_size[1] / H]
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= W or ul[1] >= H or br[0] < 0 or br[1] < 0:
                    target_weight[joint_id] = 0

                if target_weight[joint_id] > 0.5:
                    size = 2 * tmp_size + 1
                    x = np.arange(0, size, 1, np.float32)
                    y = x[:, None]
                    x0 = y0 = size // 2

                    g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

                    g_x = max(0, -ul[0]), min(br[0], W) - ul[0]
                    g_y = max(0, -ul[1]), min(br[1], H) - ul[1]

                    img_x = max(0, ul[0]), min(br[0], W)
                    img_y = max(0, ul[1]), min(br[1], H)

                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        if self.use_different_joint_weights:
            target_weight = np.multiply(target_weight, self.joint_weights)

        return target, target_weight

