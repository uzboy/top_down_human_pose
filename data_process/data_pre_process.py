import cv2
import numpy as np
from numpy import random
from torchvision.transforms import functional as F


class ToTensor:

    def __call__(self, image):
        image = F.to_tensor(image)
        return image


class NormalizeTensor:

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image


class PhotometricDistortion:

    def __init__(self, brightness_delta=32, contrast_range=(0.5, 1.5), saturation_range=(0.5, 1.5), hue_delta=18,
                                        brightness_prob=0.1, contrast_porb=0.1, saturation_prob=0.1, hue_prob=0.1):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta
        self.brightness_prob=brightness_prob
        self.contrast_porb=contrast_porb
        self.saturation_prob=saturation_prob
        self.hue_prob=hue_prob

    def convert(self, img, alpha=1, beta=0):
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img):
        if random.rand() < self.brightness_prob:
            return self.convert(img, beta=random.uniform(-self.brightness_delta, self.brightness_delta))
        return img

    def contrast(self, img):
        if random.rand() < self.contrast_porb:
            return self.convert(img, alpha=random.uniform(self.contrast_lower, self.contrast_upper))
        return img

    def saturation(self, img):
        if random.rand() < self.saturation_prob:
            img[:, :, 1] = self.convert(img[:, :, 1], alpha=random.uniform(self.saturation_lower, self.saturation_upper))
        return img

    def hue(self, img):
        if random.rand() < self.hue_prob:
            img[:, :, 0] = (img[:, :, 0].astype(int) + random.randint(-self.hue_delta, self.hue_delta)) % 180
        return img

    def swap_channels(self, img):
        if random.randint(2):
            img = img[..., random.permutation(3)]
        return img

    def __call__(self, image):
        img = self.brightness(image)
    
        mode = random.randint(2)
        if mode == 1:
            img = self.contrast(img)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        img = self.saturation(img)
        img = self.hue(img)
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        if mode == 0:
            img = self.contrast(img)

        # self.swap_channels(img)
        return image


class GetRandomScaleRotation:

    def __init__(self, rot_factor=40, scale_factor=0.5, rot_prob=0.6):
        self.rot_factor = rot_factor
        self.scale_factor = scale_factor
        self.rot_prob = rot_prob

    def __call__(self, scale):
        sf = self.scale_factor
        rf = self.rot_factor

        s_factor = np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
        scale = scale * s_factor

        r_factor = np.clip(np.random.randn() * rf, -rf * 2, rf * 2)
        r_factor = r_factor if np.random.rand() <= self.rot_prob else 0

        return scale, r_factor


class Affine:

    def __init__(self):
        pass

    def __call__(self, image, image_size, joints, center, scale, rotation):

        trans = self.get_affine_transform(center, scale, rotation, image_size)
        image = cv2.warpAffine(image, trans, (int(image_size[0]), int(image_size[1])), flags=cv2.INTER_LINEAR)
        for i in range(len(joints)):
            if joints[i, 2] > 0.0:
                joints[i, 0:2] = self.affine_transform(joints[i, 0:2], trans)

        return image, joints

    def rotate_point(self, pt, angle_rad):
        sn, cs = np.sin(angle_rad), np.cos(angle_rad)
        new_x = pt[0] * cs - pt[1] * sn
        new_y = pt[0] * sn + pt[1] * cs
        rotated_pt = [new_x, new_y]
        return rotated_pt

    def _get_3rd_point(self, a, b):
        direction = a - b
        third_pt = b + np.array([-direction[1], direction[0]], dtype=np.float32)
        return third_pt

    def get_affine_transform(self, center, scale, rot, output_size):
        scale_tmp = scale * 200.0

        src_w = scale_tmp[0]
        dst_w = output_size[0]
        dst_h = output_size[1]

        rot_rad = np.pi * rot / 180
        src_dir = self.rotate_point([0., src_w * -0.5], rot_rad)
        dst_dir = np.array([0., dst_w * -0.5])

        src = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center
        src[1, :] = center + src_dir
        src[2, :] = self._get_3rd_point(src[0, :], src[1, :])

        dst = np.zeros((3, 2), dtype=np.float32)
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
        dst[2, :] = self._get_3rd_point(dst[0, :], dst[1, :])

        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
        return trans

    def affine_transform(self, pt, trans_mat):
        assert len(pt) == 2
        new_pt = np.array(trans_mat) @ np.array([pt[0], pt[1], 1.])
        return new_pt


class RandomFlip:

    def __init__(self, flip_prob=0.5, flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]):
        self.flip_prob = flip_prob
        self.flip_pairs = flip_pairs

    def __call__(self, images, joints, centers):
        if np.random.rand() <= self.flip_prob:
            images = images[:, ::-1, :]
            joints = self.fliplr_joints(joints, images.shape[1])
            centers[0] = images.shape[1] - centers[0] - 1

        return images, joints, centers

    def fliplr_joints(self, joints, img_width):
        joints_flipped = joints.copy()
        for left, right in self.flip_pairs:
            joints_flipped[left, :] = joints[right, :]
            joints_flipped[right, :] = joints[left, :]

        joints_flipped[:, 0] = img_width - 1 - joints_flipped[:, 0]

        return joints_flipped


class HalfBodyTransform:

    def __init__(self, image_size, num_joints_half_body=8, prob_half_body=0.3, num_joints = 17,
                              upper_body_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                              lower_body_index = [11, 12, 13, 14, 15, 16]):
        self.image_size = image_size
        self.num_joints_half_body = num_joints_half_body
        self.prob_half_body = prob_half_body
        self.upper_body_index = upper_body_index
        self.lower_body_index = lower_body_index
        self.num_joints = num_joints

    def half_body_transform(self, joints):
        upper_joints = []
        lower_joints = []
        for joint_id in range(self.num_joints):
            if joints[joint_id][-1] > 0:
                if joint_id in self.upper_body_index:
                    upper_joints.append(joints[joint_id])
                else:
                    lower_joints.append(joints[joint_id])

        if np.random.randn() < 0.5 and len(upper_joints) > 2:
            selected_joints = upper_joints
        elif len(lower_joints) > 2:
            selected_joints = lower_joints
        else:
            selected_joints = upper_joints

        if len(selected_joints) < 2:
            return None, None

        selected_joints = np.array(selected_joints, dtype=np.float32)
        center = selected_joints.mean(axis=0)[:2]

        left_top = np.amin(selected_joints, axis=0)
        right_bottom = np.amax(selected_joints, axis=0)

        w = right_bottom[0] - left_top[0]
        h = right_bottom[1] - left_top[1]

        aspect_ratio = self.image_size[0] / self.image_size[1]

        if w > aspect_ratio * h:
            h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio

        scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)
        scale = scale * 1.5         # 边框外扩了75%

        return center, scale

    def __call__(self, joints):
        if (np.sum(joints[:, -1]) > self.num_joints_half_body and np.random.rand() < self.prob_half_body):
            center, scale = self.half_body_transform(joints)
            return center, scale

        return None, None


class TopDownRandomTranslation:

    def __init__(self, trans_factor=0.15, trans_prob=1.0):
        self.trans_factor = trans_factor
        self.trans_prob = trans_prob

    def __call__(self, center, scale):
        if np.random.rand() <= self.trans_prob:
            center += self.trans_factor * np.random.uniform(-1, 1, size=2) * scale * 200
        return center
