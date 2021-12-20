import cv2
import numpy as np


class MegviiHeatmap:

    def __init__(self, cfg):
        self.kernel = cfg.kernel
        self.num_joints = cfg.num_joints
        self.image_size = cfg.image_size
        self.heatmap_size = cfg.heatmap_size
        self.joint_weights = cfg.joint_weights
        self.use_different_joint_weights = cfg.use_different_joint_weights

    def __call__(self, joints):
        W, H = self.heatmap_size
        target = np.zeros((self.num_joints, H, W), dtype=np.float32)
        target_weight = np.zeros((self.num_joints, 1), dtype=np.float32)

        for i in range(self.num_joints):
            target_weight[i] = joints[i, -1]
            if target_weight[i] < 1:
                continue

            target_y = int(joints[i, 1] * H / self.image_size[1])
            target_x = int(joints[i, 0] * W / self.image_size[0])
            if (target_x >= W or target_x < 0) or (target_y >= H or target_y < 0):
                target_weight[i] = 0
                continue

            target[i, target_y, target_x] = 1
            target[i] = cv2.GaussianBlur(target[i], self.kernel, 0)
            maxi = target[i, target_y, target_x]

            target[i] /= maxi / 255

        if self.use_different_joint_weights:
            target_weight = np.multiply(target_weight, self.joint_weights)

        return target, target_weight

