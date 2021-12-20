import numpy as np
from dataset.coco_data_for_eval import CocoDataEval
from data_process.data_decode_target import decode_heat_map


class TopDownHumanPoseEval:

    def __init__(self, cfg, model):
        self.cfg = cfg
        self.model = model
        self.dataset = CocoDataEval(cfg.data)
        self.sigmas = np.array([0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062,
                                                        0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089])
        self.vars = (self.sigmas * 2) ** 2
        self.joint_num = len(self.sigmas)
        self.oks_ths = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.area_ranges = [[0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2], [0 ** 2, 1e5 ** 2]]
        self.area_map = {"small": 0, 
                                            "middle": 1,
                                            "large": 2,
                                            "all": 3}

    def _forward(self):
        forward_infos = []
        for index in range(len(self.dataset)):
            target_info = self.dataset.getitem(index)
            heatmaps = self.model(target_info["image"].to(self.cfg.device))
            if isinstance(heatmaps, list):
                heatmaps = heatmaps[-1]
            heatmaps = heatmaps.detach().cpu().numpy()
            preds, _ = decode_heat_map(heatmaps, target_info["center"], target_info["scale"])
            oks_values = self.get_oks_value(target_info["joints"], target_info["box"], target_info["area"], preds.squeeze(0))
            forward_infos.append({"target_area":target_info["area"],
                                                             "oks_value": oks_values})

        return forward_infos

    def get_oks_value(self, gt_kps, gt_box, gt_area, preds):
        xg = gt_kps[:, 0]
        yg = gt_kps[:, 1]
        vg = gt_kps[:, 2]

        k1 = np.count_nonzero(vg > 0)

        xd = preds[:, 0]
        yd = preds[:, 1]
        if k1>0:
            dx = xd - xg
            dy = yd - yg
        else:
            x0 = gt_box[0] - gt_box[2]
            x1 = gt_box[0] + gt_box[2] * 2
            y0 = gt_box[1] - gt_box[3]
            y1 = gt_box[1] + gt_box[3] * 2
            z = np.zeros((self.joint_num))
            dx = np.max((z, x0 - xd),axis=0) + np.max((z, xd - x1), axis=0)
            dy = np.max((z, y0 - yd),axis=0) + np.max((z, yd - y1), axis=0)

        e = (dx ** 2 + dy ** 2) / self.vars / (gt_area + np.spacing(1)) / 2
        if k1 > 0:
            e=e[vg > 0]
    
        data = np.sum(np.exp(-e)) / e.shape[0]
        return data

    def get_match_result(self, forward_infos, oks_th, area_range):
        match_result = -np.ones(len(forward_infos))

        for index in range(len(forward_infos)):
            info = forward_infos[index]
            target_area = info["target_area"]
            if target_area < area_range[0] or target_area > area_range[1]:
                continue
            if info["oks_value"] >= oks_th:
                match_result[index] = 1
            else:
                match_result[index] = 0
    
        return match_result

    def get_eval_result(self, match_result):
        map = np.sum(match_result == 1) / (np.sum(match_result != -1) + np.spacing(1))

        th_index = np.where(0.5 == self.oks_ths)[0]
        tmp_result = match_result[th_index]
        ap_50 = np.sum(tmp_result == 1) / (np.sum(tmp_result != -1) + np.spacing(1))

        th_index = np.where(0.75 == self.oks_ths)[0]
        tmp_result = match_result[th_index]
        ap_75 = np.sum(tmp_result == 1) / (np.sum(tmp_result != -1)  + np.spacing(1))

        return map, ap_50, ap_75

    def eval(self):
        forward_infos = self._forward()
        match_result = -np.ones([len(self.oks_ths), len(self.area_ranges), len(forward_infos)])
        for oks_index in range(len(self.oks_ths)):
            for area_index in range(len(self.area_ranges)):
                match_result[oks_index, area_index] = self.get_match_result(forward_infos, self.oks_ths[oks_index], self.area_ranges[area_index])

        # for all:
        all_map, all_ap_50, all_ap_75 = self.get_eval_result(match_result[:, self.area_map["all"]])
        # for small:
        small_map, small_ap_50, small_ap_75 = self.get_eval_result(match_result[:, self.area_map["small"]])
        # for large:
        large_map, large_ap_50, large_ap_75 = self.get_eval_result(match_result[:, self.area_map["large"]])
        # for middle:
        mid_map, mid_ap_50, mid_ap_75 = self.get_eval_result(match_result[:, self.area_map["middle"]])

        str_info = "\tmap\t\tap50\t\tap75\r\n"
        str_info += "small:\t%.4f\t\t%.4f\t\t%.4f\r\n" % (small_map, small_ap_50, small_ap_75)
        str_info += "large:\t%.4f\t\t%.4f\t\t%.4f\r\n" % (large_map, large_ap_50, large_ap_75)
        str_info += "middle:\t %.4f\t\t %.4f\t\t%.4f\r\n" % (mid_map, mid_ap_50, mid_ap_75)
        str_info += "all:\t%.4f\t\t%.4f\t\t%.4f\n" % (all_map, all_ap_50, all_ap_75)
        return str_info
