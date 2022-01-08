import os
from posixpath import join
import cv2
import json
import numpy as np
import torch.utils.data as data
from data_process.data_pre_process import Affine
from data_process.data_pre_process import ToTensor
from data_process.data_pre_process import RandomFlip
from data_process.data_pre_process import NormalizeTensor
from data_process.data_pre_process import HalfBodyTransform
from data_process.data_pre_process import PhotometricDistortion
from data_process.data_pre_process import GetRandomScaleRotation
from data_process.heatmaps.build_heatmap_generator import build_heatmaps


class CocoDataWithMutex(data.Dataset):

    def __init__(self, cfg):
        super(CocoDataWithMutex, self).__init__()
        self.anno = []
        self.image_size = cfg.image_size
        self.num_joints = cfg.num_joints
        self.image_root = cfg.image_root
        self.load_annotions(cfg.annotion_file)

        self.is_train = cfg.get("is_train", False)
        self.is_flip = cfg.get("is_flip", False)
        self.is_half_body = cfg.get("is_half_body", False)
        self.is_rot = cfg.get("is_rot", False)
        self.is_pic = cfg.get("is_pic", False)

        if self.is_train:
            self.target_generators = build_heatmaps(cfg.heatmaps)

        if self.is_flip:
            self.random_flip = RandomFlip(cfg.get("flip_prob", 0.5),
                                                                             cfg.get("flip_pairs",  [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]))
    
        if self.is_half_body:
            self.half_body = HalfBodyTransform(cfg.image_size,
                                                                                        cfg.get("num_joints_half_body", 8),
                                                                                        cfg.get("prob_half_body", 0.3),
                                                                                        cfg.get("num_joints", 17),
                                                                                        cfg.get("upper_body_index", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                                                                                        cfg.get("lower_body_index", [11, 12, 13, 14, 15, 16]))
    
        if self.is_rot:
            self.scale_rotation = GetRandomScaleRotation(cfg.get("rot_factor", 40),
                                                                                                              cfg.get("scale_factor", 0.5),
                                                                                                              cfg.get("rot_prob", 0.6))


        if self.is_pic:
            self.pix_aug = PhotometricDistortion(cfg.get("brightness_delta", 32),
                                                                                         cfg.get("contrast_range", (0.5, 1.5)),
                                                                                         cfg.get("saturation_range", (0.5, 1.5)),
                                                                                         cfg.get("hue_delta", 18),
                                                                                         cfg.get("brightness_prob", 0.1),
                                                                                         cfg.get("contrast_prob", 0.1),
                                                                                         cfg.get("saturation_prob", 0.1),
                                                                                         cfg.get("hue_prob", 0.1))
    
        self.img_affine = Affine()
        self.to_tensor = ToTensor()
        self.norm_image = NormalizeTensor(cfg.get("mean", [0.485, 0.456, 0.406]),
                                                                                    cfg.get("std", [0.229, 0.224, 0.225]))

    def load_annotions(self, annotion_file):
        with open(annotion_file, 'r', encoding='utf-8') as f:
            annotion_data = json.load(f)

        images_annotions = annotion_data["images"]
        id_images_maps = {}
        for image_anno in images_annotions:
            img_urls = image_anno["coco_url"].strip().split("/")
            img_url = img_urls[-2] + "/" + img_urls[-1]
            id_images_maps[image_anno["id"]] = {"img_url":img_url,
                                                                                             "height":image_anno["height"],
                                                                                             "width": image_anno["width"]}
    
        kps_annotions = annotion_data["annotations"]
        for kps_annotion in kps_annotions:
            if kps_annotion["category_id"] != 1:
                continue
            if kps_annotion["iscrowd"] != 0:
                continue
            if 'keypoints' not in kps_annotion:
                continue
            if max(kps_annotion['keypoints']) == 0:
                continue
            if 'num_keypoints' in kps_annotion and kps_annotion['num_keypoints'] == 0:
                continue
        
            img_id = kps_annotion["image_id"]
            img_url = id_images_maps[img_id]["img_url"]
            
            width = id_images_maps[img_id]["width"]
            height = id_images_maps[img_id]["height"]

            x, y, w, h = kps_annotion["bbox"]
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width - 1, x1 + max(0, w - 1))
            y2 = min(height - 1, y1 + max(0, h - 1))
            if ('area' not in kps_annotion or kps_annotion['area'] > 0) and x2 > x1 and y2 > y1:
                box = [x1, y1, x2 - x1, y2 - y1]

            joints = np.zeros((self.num_joints, 3), dtype=np.float32)

            keypoints = np.array(kps_annotion['keypoints']).reshape(-1, 3)
            joints[:, :2] = keypoints[:, :2]
            joints[:, 2] = np.minimum(1, keypoints[:, 2])

            one_annotions = {
                "img_url": os.path.join(self.image_root, img_url),
                "joints": joints,
                "box": box
            }
            self.anno.append(one_annotions)

    def _xywh2cs(self, x, y, w, h, padding=1.25):
        aspect_ratio = self.image_size[0] / self.image_size[1]
        center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)

        if np.random.rand() < 0.3:
            center += 0.4 * (np.random.rand(2) - 0.5) * [w, h]          # 随机移动中心点 20%

        if w > aspect_ratio * h:                # 保证宽高比例
            h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio

        scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)
        scale = scale * padding             # scale进行了外扩： 62.5%

        return center, scale

    def __getitem__(self, index):
        sample_infos = self.anno[index]
        center, scale = self._xywh2cs(sample_infos["box"][0],
                                                                    sample_infos["box"][1],
                                                                    sample_infos["box"][2],
                                                                    sample_infos["box"][3])
        joints = sample_infos["joints"]
        img_url = sample_infos["img_url"]
        image = cv2.imread(img_url)
        rot = 0
    
        if self.is_train:
            if self.is_flip:
                image, joints, center = self.random_flip(image, joints, center)

            if self.is_half_body:
                center_tmp, scale_tmp = self.half_body(joints)
                if center_tmp is not None and scale_tmp is not None:
                    center = center_tmp
                    scale = scale_tmp

            if self.is_rot:
                scale, rot = self.scale_rotation(scale)

            if self.is_pic:
                image = self.pix_aug(image)

        image, joints = self.img_affine(image, self.image_size, joints, center, scale, rot)
        image = self.to_tensor(image)
        image = self.norm_image(image)
    
        if self.is_train:
            targets = []
            target_weights = []
            for index in range(len(self.target_generators)):
                target, target_weight = self.target_generators[index](joints)
                targets.append(np.expand_dims(target, 0))
                target_weights.append(np.expand_dims(target_weight, 0))
            if len(targets) > 1:
                targets = np.concatenate(targets, axis=0)
                target_weights = np.concatenate(target_weights, axis=0)
            else:
                targets = targets[0].squeeze(0)
                target_weights = target_weights[0].squeeze(0)
        else:
            targets = None
            target_weights = None
        return image, targets, target_weights

    def __len__(self):
        return len(self.anno)
