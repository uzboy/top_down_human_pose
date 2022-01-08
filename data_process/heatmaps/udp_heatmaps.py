import numpy as np


class UDPHeatmap:

    def __init__(self, cfg):
        self.num_joints = cfg.num_joints
        self.heatmap_size = cfg.heatmap_size
        self.image_size = cfg.image_size
        self.sigma = cfg.get("sigma", 2)
        self.factor = cfg.get("factor", 2)
        self.is_gauss = cfg.get("is_gauss", True)
        self.joint_weights = cfg.get("joint_weights", None)
        self.use_different_joint_weights = cfg.get("use_different_joint_weights", False)

    def _generate_gaussian_heatmap(self, joints):
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints[:, -1]
        target = np.zeros((self.num_joints, self.heatmap_size[1], self.heatmap_size[0]), dtype=np.float32)
        tmp_size = self.sigma * 3

        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, None]

        for joint_id in range(self.num_joints):
            feat_stride = (self.image_size - 1.0) / (self.heatmap_size - 1.0)
            mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)

            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] or br[0] < 0 or br[1] < 0:
                target_weight[joint_id] = 0
                continue

            mu_x_ac = joints[joint_id][0] / feat_stride[0]
            mu_y_ac = joints[joint_id][1] / feat_stride[1]
            x0 = y0 = size // 2
            x0 += mu_x_ac - mu_x
            y0 += mu_y_ac - mu_y
            g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * self.sigma**2))

            g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
            img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
            img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

            if target_weight[joint_id] > 0.5:
                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        
        return target, target_weight

    def _generate_combined_target(self, joints):
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints[:, -1]
        target = np.zeros((self.num_joints, 3, self.heatmap_size[1] * self.heatmap_size[0]), dtype=np.float32)

        feat_width = self.heatmap_size[0]
        feat_height = self.heatmap_size[1]
        feat_x_int = np.arange(0, feat_width)
        feat_y_int = np.arange(0, feat_height)
        feat_x_int, feat_y_int = np.meshgrid(feat_x_int, feat_y_int)
        feat_x_int = feat_x_int.flatten()
        feat_y_int = feat_y_int.flatten()

        valid_radius = self.factor * self.heatmap_size[1]
        feat_stride = (self.image_size - 1.0) / (self.heatmap_size - 1.0)

        for joint_id in range(self.num_joints):
            mu_x = joints[joint_id][0] / feat_stride[0]
            mu_y = joints[joint_id][1] / feat_stride[1]
            x_offset = (mu_x - feat_x_int) / valid_radius
            y_offset = (mu_y - feat_y_int) / valid_radius

            dis = x_offset ** 2 + y_offset ** 2
            keep_pos = np.where(dis <= 1)[0]
            if target_weight[joint_id] > 0.5:
                target[joint_id, 0, keep_pos] = 1
                target[joint_id, 1, keep_pos] = x_offset[keep_pos]
                target[joint_id, 2, keep_pos] = y_offset[keep_pos]
    
        target = target.reshape(self.num_joints * 3, self.heatmap_size[1], self.heatmap_size[0])

        return target, target_weight

    def __call__(self, joints):
        if self.is_gauss:
            target, target_weight = self._generate_gaussian_heatmap(joints)
        else:
            target, target_weight = self._generate_combined_target(joints)

        if self.use_different_joint_weights:
            target_weight = np.multiply(target_weight, self.joint_weights)

        return target, target_weight
