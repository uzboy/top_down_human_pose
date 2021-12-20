import os
import cv2
import json
import numpy as np
from data_process.data_pre_process import Affine
from data_process.data_pre_process import ToTensor
from data_process.data_pre_process import NormalizeTensor


class CocoDataEval:

    def __init__(self, cfg):
        self.cfg = cfg
        self.anno = []
        self.load_annotions(self.cfg.annotion_file)

        self.img_affine = Affine()
        self.to_tensor = ToTensor()
        self.norm_image = NormalizeTensor(self.cfg.mean, self.cfg.std)

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
                box = [x1, y1, x2 - x1, y2 - y1]            # 左上角坐标 + 宽高

            
            joints = np.zeros((self.cfg.num_joints, 3), dtype=np.float32)
        
            keypoints = np.array(kps_annotion['keypoints']).reshape(-1, 3)
            joints[:, :2] = keypoints[:, :2]
            joints[:, 2] = np.minimum(1, keypoints[:, 2])

            one_annotions = {
                "img_url": os.path.join(self.cfg.image_root, img_url),
                "joints": joints,
                "box": box,
                "area":box[2] * box[3]
            }
            self.anno.append(one_annotions)

    def _xywh2cs(self, x, y, w, h, padding=1.25):
        aspect_ratio = self.cfg.image_size[0] / self.cfg.image_size[1]
        center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)

        if w > aspect_ratio * h:                # 保证宽高比例
            h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio

        scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)
        scale = scale * padding

        return center, scale

    def getitem(self, index):
        sample_infos = self.anno[index]
        center, scale = self._xywh2cs(sample_infos["box"][0],
                                                                    sample_infos["box"][1],
                                                                    sample_infos["box"][2],
                                                                    sample_infos["box"][3])
        joints = sample_infos["joints"]
        img_url = sample_infos["img_url"]

        image = cv2.imread(img_url)
        image, _ = self.img_affine(image, self.cfg.image_size, joints.copy(), center, scale, 0)
        image = self.to_tensor(image)
        image = self.norm_image(image)

        target_info = {
            "image": image.unsqueeze(0),
            "joints": joints,
            "area": sample_infos["area"],
            "scale": scale,
            "box": sample_infos["box"],
            "img_url":sample_infos["img_url"],
            "center": center
        }
        return target_info

    def __len__(self):
        return len(self.anno)
