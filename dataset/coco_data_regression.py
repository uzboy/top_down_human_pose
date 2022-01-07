import os
import cv2
import json
import numpy as np
import torch.utils.data as data
from data_process.data_pre_process import ToTensor
from data_process.data_pre_process import NormalizeTensor
from data_process.data_pre_process import PhotometricDistortion
from data_process.data_pre_process import Rotation
from data_process.data_pre_process import ExpanBorder
from data_process.data_pre_process import GetTargetSampleImage


class CocoDataRegression(data.Dataset):

    def __init__(self, cfg):
        super(CocoDataRegression, self).__init__()
        self.anno = []
        self.with_mask = cfg.with_mask
        self.image_size = cfg.image_size
        self.num_joints = cfg.num_joints
        self.image_root = cfg.image_root
        self.load_annotions(cfg.annotion_file)

        if hasattr(cfg, "is_rot"):
            self.is_rot = cfg.is_rot
        else:
            self.is_rot = False
    
        if self.is_rot:
            self.rotation = Rotation(cfg.rot_factor, cfg.rot_prob)

        self.expan_border = ExpanBorder(cfg.expan_factor, cfg.expan_prob, cfg.min_expan_factor)
    
        if hasattr(cfg, "is_pic"):
            self.is_pic = cfg.is_pic
        else:
            self.is_pic = False
    
        if self.is_pic:
            self.pix_aug = PhotometricDistortion(cfg.brightness_delta,
                                                                                         cfg.contrast_range,
                                                                                         cfg.saturation_range,
                                                                                         cfg.hue_delta,
                                                                                         cfg.brightness_prob,
                                                                                         cfg.contrast_prob,
                                                                                         cfg.saturation_prob,
                                                                                         cfg.hue_prob)
        self.get_target = GetTargetSampleImage(cfg.image_size, cfg.is_shift)
        self.to_tensor = ToTensor()
        self.norm_image = NormalizeTensor(cfg.mean, cfg.std)

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
            if x2 <= x1 or y2 <= y1:
                continue

            joints = np.zeros((self.num_joints + 4, 3), dtype=np.float32)
            joints[0, :] = [x1, y1, 1]
            joints[1, :] = [x1, y2, 1]
            joints[2, :] = [x2, y2, 1]
            joints[3, :] = [x2, y1, 1]

            keypoints = np.array(kps_annotion['keypoints']).reshape(-1, 3)
            joints[4:, :2] = keypoints[:, :2]
            joints[4:, 2] = np.minimum(1, keypoints[:, 2])

            one_annotions = {
                "img_url": os.path.join(self.image_root, img_url),
                "joints": joints
            }
            self.anno.append(one_annotions)

    def __getitem__(self, index):
        sample_infos = self.anno[index]
        joints = sample_infos["joints"]
        img_url = sample_infos["img_url"]
        image = cv2.imread(img_url)

        if self.is_rot:
            image, joints = self.rotation(image, joints)
    
        joints = self.expan_border(joints, image.shape)

        dst_image, joints = self.get_target(image, joints)
        image = self.to_tensor(dst_image)
        image = self.norm_image(image)

        joints = joints[4:]

        if self.with_mask:
            target = np.zeros((self.num_joints * 3), dtype=np.float32)
            target_weight = np.zeros((self.num_joints * 3), dtype=np.float32)
        else:
            target = np.zeros((self.num_joints * 2), dtype=np.float32)
            target_weight = np.zeros((self.num_joints * 2), dtype=np.float32)

        for i in range(self.num_joints):
            if self.with_mask:
                target_weight[i + 2 * self.num_joints] = 1
    
            weight = joints[i, -1]
            if weight < 1:
                continue
            target_weight[i * 2] = 1
            target_weight[i * 2 + 1] = 1
            target[i * 2 + 0] = joints[i, 0] / self.image_size[0]
            target[i * 2 + 1] = joints[i, 1] / self.image_size[1]
            if self.with_mask:
                target[i + 2 * self.num_joints] = 1

        return image, target, target_weight

    def __len__(self):
        return len(self.anno)
