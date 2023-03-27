import sys
# sys.path.insert(0, "../mlsd_pytorch")
# from models.mbv2_mlsd_large import MobileV2_MLSD_Large
# from models.convnext_mlsd_large import Convnext_MLSD_Large
# from models.repvgg_mlsd_large import Repvgg_MLSD_Large
# from tmp_models.mbv2_mlsd_large import MobileV2_MLSD_Large
# from local_utils import color_list

import torch
import cv2
import copy
import numpy as np
import os
import math
import shutil
from tqdm import tqdm
from torch.nn import functional as F
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
from network.mbv2_mlsd import MobileV2_MLSD
from network.mbv2_mlsd_large import MobileV2_MLSD_Large

from local_utils import get_mlsd_cfg, sigmoid, cv2AddChineseText
from local_utils import curve_class_dict, curve_class_chinese_dict, \
                        color_list, line_class_dict


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def getX(self):
        return self.x

    def getY(self):
        return self.y


class MLSD_LineDetector(object):
    class_names = {
                    'concave_wall_line': 0,
                    'convex_wall_line': 1,
                    'wall_ground_line': 2,
                    'cabinet_wall_line': 3,
                    'cabinet_ground_line': 4,
                    'sofa_ground_line': 5,
                    'stair_line': 6,
                    'ceiling_line': 7,
                    'wall_ground_curve': 8,
                    'stair_curve': 9,
                    'cabinet_ground_curve': 10,
                    'sofa_ground_curve': 11,
                    "concave_wall_line_easy": 12,   # new 0704
                    "convex_wall_line_easy": 13,
                    "door_wall_line": 14,
                    "line_reserve": 15,
                    "outside_elevator_door_ground_line": 16,
                    "inside_elevator_door_ground_line": 17,
                    "outside_elevator_door_concave_wall_line": 18,
                    "inside_elevator_door_concave_line": 19,
                    "inside_elevator_door_convex_line": 20,
                }

    class_names_chinese = {
                    '凹墙线': 0,
                    '凸墙线': 1,
                    '墙地线': 2,
                    '柜墙线': 3,
                    '柜地线': 4,
                    '沙发地线': 5,
                    '楼梯线': 6,
                    '天花板线': 7,
                    '墙地曲线': 8,
                    '楼梯曲线': 9,
                    '柜地曲线': 10,
                    '沙发地曲线': 11,
                    "凹墙线-容易": 12,
                    "凸墙线-容易": 13,
                    "门框墙线": 14,
                    "保留线位置": 15,
                    "电梯门外地沿线": 16,
                    "电梯门内地沿线": 17,
                    "电梯门外凹沿线": 18,
                    "电梯门内凹沿线": 19,
                    "电梯门内凸沿线": 20,
                }

    def __init__(self, cfg,  model_path):
        # self.model = MobileV2_MLSD_Large().cuda().eval()
        # self.model = Convnext_MLSD_Large().cuda().eval()
        self.model = MobileV2_MLSD(cfg).cuda().eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.load_state_dict(torch.load(model_path, map_location=device), strict=True)

        for module in self.model.modules():
            if hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()

        # self.input_shape = input_shape
        # self.input_w = input_shape[0]
        # self.input_h = input_shape[1]
        # self.src_height = 1080
        # self.src_width = 1920

        self.src_height = cfg.datasets.src_height
        self.src_width = cfg.datasets.src_width

        self.input_h = cfg.datasets.input_h
        self.input_w = cfg.datasets.input_w

        self.h_ratio = self.src_height/self.input_h
        self.w_ratio = self.src_width / self.input_w

        self.cls_num = len(self.class_names.keys())

        self.score_thr = 0.5   # 0.4
        self.dist_thr = 5.0
        self.model_down_scale = cfg.model.down_scale
        self.name_2_chinese = self.get_name_2_chinese()

        self.debug_img = None

    def infer(self, img):
        batch_image = self.pre_process(img)
        outputs = self.model(batch_image)
        # pred_lines, centers, centermap, clsmap, linemap = self.decode_line(outputs)
        pred_lines = self.decode_line(outputs)
        return pred_lines

    def decode_line(self, outputs):
        # centermap = outputs[:, 19, :, :]
        # centermap = centermap.squeeze(0)
        # centermap = centermap.detach().cpu().numpy()
        #
        # cls_map = outputs[:, 26:38, :, :]
        # cls_map = cls_map.squeeze(0)
        # cls_map = cls_map.detach().cpu().numpy()
        # self.debug_img = cls_map[0, :, :]
        # self.debug_img[self.debug_img < 0.2] = 0
        #
        # # line map
        # line_map = outputs[:, 39, :, :]
        # line_map = line_map.squeeze(0)
        # line_map = line_map.detach().cpu().numpy()

        # vmap: displacement h x w x 4 ,center_with_cls: 1 x 200 x 4(ys,xs,scores,cls)
        vmap, center_with_cls = self.deccode_output_score_and_ptss_and_cls(outputs, 200, 3)
        center_with_cls = center_with_cls.detach().cpu().numpy()

        detections = []
        results = []
        for i in range(center_with_cls.shape[0]):
            ret = {}
            classes = center_with_cls[i, :, -1]
            for j in range(self.cls_num):
                inds = (classes == j)
                ret[j + 1] = np.concatenate(
                    [center_with_cls[i, inds, :2].astype(np.float32), center_with_cls[i, inds, 2:3].astype(np.float32)],
                    axis=1).tolist()
            results.append(ret)

        for j in range(1, self.cls_num + 1):
            results[0][j] = np.array(results[0][j], dtype=np.float32).reshape(-1, 3)

        detections.append(results[0])

        res = {}
        for k in range(1, self.cls_num + 1):
            res[k] = np.concatenate([detection[k] for detection in detections], axis=0).astype(np.float32)

        start = vmap[:, :, :2]  # 256 x 256 x [0, 1] displacement(xs, ys)
        end = vmap[:, :, 2:]  # 256 x 256 x [2, 3] displacement(xe, ye)
        dist_map = np.sqrt(np.sum((start - end) ** 2, axis=-1))  # 256 x 256
        segments_list_with_cls = []
        center_list = []
        seeds = []

        for m in range(1, self.cls_num + 1):
            for n in res[m]:
                y, x = n[0], n[1]
                distance = dist_map[int(y), int(x)]
                if n[2] > self.score_thr and distance > self.dist_thr:
                    seeds.append([Point(int(y), int(x)), int(m - 1)])
                    disp_x_start, disp_y_start, disp_x_end, disp_y_end = vmap[int(y), int(x), :]
                    x_start = x + disp_x_start
                    y_start = y + disp_y_start
                    x_end = x + disp_x_end
                    y_end = y + disp_y_end

                    x_start = self.model_down_scale * x_start * self.w_ratio
                    y_start = self.model_down_scale * y_start * self.h_ratio
                    x_end = self.model_down_scale * x_end * self.w_ratio
                    y_end = self.model_down_scale * y_end * self.h_ratio

                    segments_list_with_cls.append([x_start, y_start, x_end, y_end, int(m - 1), n[2]])
                    center_list.append([x_start + (x_end - x_start) / 2, y_start + (y_end - y_start) / 2])

        if segments_list_with_cls != []:
            segments_list_with_cls = np.asarray(segments_list_with_cls)

            segments_list_with_cls = self.line_nms2(segments_list_with_cls)
            segments_list_with_cls = segments_list_with_cls.tolist()
        # return segments_list_with_cls, center_list, centermap, cls_map, line_map
        return segments_list_with_cls


    def pre_process(self, img):
        img_h, img_w, _ = img.shape
        # assert img_h == self.src_height, "img_h not equel to 1080"
        # assert img_w == self.src_width, "img_w not equel to 1920"

        img_proc = copy.deepcopy(img)
        img_proc = cv2.cvtColor(img_proc, cv2.COLOR_BGR2RGB)

        self.h_ratio, self.w_ratio = [img_h / self.input_h, img_w / self.input_w]
        res_image = cv2.resize(img_proc, (self.input_w, self.input_h))

        res_image = res_image.transpose((2, 0, 1))
        batch_image = np.expand_dims(res_image, axis=0).astype('float32')
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        max_pixel_value = 255
        batch_image = (batch_image - (mean * max_pixel_value)[None, ..., None, None]) / (std * max_pixel_value)[
            None, ..., None, None]
        batch_image = torch.from_numpy(batch_image).float().cuda()
        return batch_image

    def deccode_output_score_and_ptss_and_cls(self, tpMap, topk_n=200, ksize=3):
        '''
        tpMap:
        center: tpMap[1, 19, :, :]
        displacement: tpMap[1, 20:24, :, :]
        cls: tpMap[1, 24:36, :, :]
        '''
        b, c, h, w = tpMap.shape
        assert b == 1, 'only support bsize==1'

        displacement = tpMap[:, self.cls_num + 7 + 1:self.cls_num + 7 + 5, :, :][0]  # 4 x 256 x 256
        displacement = displacement.detach().cpu().numpy()
        displacement = displacement.transpose((1, 2, 0))  # 256 x 256 x 4

        cls = tpMap[:, self.cls_num + 7 + 7:(self.cls_num + 7)*2, :, :]
        # cls = tpMap[:, 0:1, :, :]
        heat_cls = torch.sigmoid(cls)
        hmax_cls = F.max_pool2d(heat_cls, (ksize, ksize), stride=1, padding=(ksize - 1) // 2)
        keep_cls = (hmax_cls == heat_cls).float()
        heat_cls = heat_cls * keep_cls

        cls_score, clss, ys, xs = self._topk(heat_cls, topk_n)  # cls_score, clss, ys, xs: 1 x topk_n

        cls_score = cls_score.view(1, topk_n, 1)
        clss = clss.view(1, topk_n, 1).float()
        ys = ys.view(1, topk_n, 1).float()
        xs = xs.view(1, topk_n, 1).float()

        center_and_cls = torch.cat([ys, xs, cls_score, clss], dim=2)  # 1 x topk_n x 4

        return displacement, center_and_cls

    def _topk(self, scores, K=40):
        batch, cat, height, width = scores.size()

        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1),
                                            K)  # scores.view(batch, cat, -1) = scores(batch, cat, height*width) eg: 1x12x256x256 -- 1x12x65536

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds / width).int().float()
        topk_xs = (
                    topk_inds % width).int().float()  # 求取对应的在 height width的x，y坐标. outputs:  topk_scores, topk_inds,topk_ys, topk_xs: 1x12xK

        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1),
                                          K)  # topk_scores.view(batch, -1): 1x12K, topk_score,topk_ind: 1xK
        topk_clses = (topk_ind / K).int()  # 对应的cls 1 x K

        topk_inds = self._gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)  # 1 x K
        topk_ys = self._gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)  # 1 x K
        topk_xs = self._gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)  # 1 x K

        return topk_score, topk_clses, topk_ys, topk_xs

    def _gather_feat(self, feat, ind, mask=None):  # feat 1x480x1, ind 1x40
        dim = feat.size(2)  # 1
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat


    def line_nms2(self, lines_with_cls_np):
        # lines_with_cls_np = lines_with_cls_np[[4, 3], :]
        lines_with_cls = list(lines_with_cls_np)
        lines_with_cls = sorted(lines_with_cls, key=lambda x: x[5], reverse=True)

        mask = [True] * len(lines_with_cls)
        nms_num = len(mask)
        if nms_num != 0:
            for i in range(nms_num):
                line_1 = lines_with_cls[i]
                l1_cls = line_1[4]

                l1x1 = line_1[0]
                l1y1 = line_1[1]
                l1x2 = line_1[2]
                l1y2 = line_1[3]
                l1xc = (l1x1 + l1x2) / 2
                l1yc = (l1y1 + l1y2) / 2

                A = l1y2 - l1y1
                B = l1x1 - l1x2
                C = l1x2 * l1y1 - l1x1 * l1y2

                l1_left = min(l1x1, l1x2)
                l1_right = max(l1x1, l1x2)
                l1_top = min(l1y1, l1y2)
                l1_bottom = max(l1y1, l1y2)

                if not mask[i]:
                    continue

                for j in range(i + 1, nms_num):
                    line_2 = lines_with_cls[j]
                    l2_cls = line_2[4]

                    # if l1_cls != l2_cls:
                    #     continue

                    l2x1 = line_2[0]
                    l2y1 = line_2[1]
                    l2x2 = line_2[2]
                    l2y2 = line_2[3]
                    l2xc = (l2x1 + l2x2) / 2
                    l2yc = (l2y1 + l2y2) / 2

                    line_dist = abs(l1x1 - l2x1) + abs(l1y1 - l2y1) + abs(l1x2 - l2x2) + abs(l1y2 - l2y2)
                    center_dist = abs(l1xc - l2xc) + abs(l1yc - l2yc)
                    #
                    # B center to line A distance
                    pt_to_line_distance = (A * l2xc + B * l2yc + C) / (math.sqrt(A * A + B * B) + 1e-6)
                    pt_to_line_distance = abs(pt_to_line_distance)

                    #  iou
                    l2_left = min(l2x1, l2x2)
                    l2_right = max(l2x1, l2x2)
                    l2_top = min(l2y1, l2y2)
                    l2_bottom = max(l2y1, l2y2)

                    overlap_left = max(l1_left, l2_left)
                    overlap_right = min(l1_right, l2_right)
                    overlap_top = max(l1_top, l2_top)
                    overlap_bottom = min(l1_bottom, l2_bottom)
                    union_left = min(l1_left, l2_left)
                    union_right = max(l1_right, l2_right)
                    union_top = min(l1_top, l2_top)
                    union_bottom = max(l1_bottom, l2_bottom)

                    iou_x = max(0.0, overlap_right - overlap_left) / (max(0.0, union_right - union_left) + 1e-6)
                    iou_y = max(0.0, overlap_bottom - overlap_top) / (max(0.0, union_bottom - union_top) + 1e-6)

                    if iou_x < 0.1 or iou_y < 0.1:
                        iou = 0
                    else:
                        iou = max(iou_x, iou_y)
                    iou = max(iou_x, iou_y)

                    iou_thr = 0.5
                    center_to_line_thr = 10
                    is_overlap = False
                    if pt_to_line_distance < center_to_line_thr and iou > iou_thr:
                        is_overlap = True

                    line_dist_thr = 30
                    center_dist_thr = 8
                    if line_dist < line_dist_thr or center_dist < center_dist_thr or is_overlap:
                        mask[j] = False

            lines_with_cls_np = np.array(lines_with_cls)[mask]
        return lines_with_cls_np


    def draw_chinese(self, img, text, position, textColor=(0, 255, 0), textSize=30):
        textColor = textColor[::-1]
        if (isinstance(img, np.ndarray)):
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        fontStyle = ImageFont.truetype(
            "/mnt/data10/liyj/programs/mlsd_with_cls/local_files/simhei.ttf", textSize, encoding="utf-8")

        draw.text(position, text, textColor, font=fontStyle)
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        return img

    def get_cls_name(self, value):
        return [k for k, v in self.class_names.items() if v == value]

    def get_name_2_chinese(self):
        name_2_chinese = {}
        for key, val in zip(self.class_names.keys(), self.class_names_chinese.keys()):
            name_2_chinese.update({key: val})
        return name_2_chinese


def show_line_demo():
    img_root = "/mnt/data10/liyj/programs/ULSD-ISPRS/dataset/mlsd_20230327/train/images"
    save_root = "/mnt/data10/liyj/debug"
    if os.path.exists(save_root):
        shutil.rmtree(save_root)
    os.mkdir(save_root)

    model_path = "/mnt/data10/liyj/programs/ULSD-ISPRS/model/mlsd_20230327/latest.pth"

    img_names = [name for name in os.listdir(img_root)
                 if name.endswith('.png') or name.endswith('.jpg')]
    # model set
    cfg_path = "/mnt/data10/liyj/programs/ULSD-ISPRS/configs/mlsd_cfg.yaml"
    cfg = get_mlsd_cfg(cfg_path)

    classes = curve_class_dict
    mlsd_detector = MLSD_LineDetector(cfg, model_path)

    CLASSES = {
        'concave_wall_line': 0,
        'convex_wall_line': 1,
        'wall_ground_line': 2,
        'cabinet_wall_line': 3,
        'cabinet_ground_line': 4,
        'sofa_ground_line': 5,
        'stair_line': 6,
        'ceiling_line': 7,
        'wall_ground_curve': 8,
        'stair_curve': 9,
        'cabinet_ground_curve': 10,
        'sofa_ground_curve': 11,
        "concave_wall_line_easy": 12,  # new 0704
        "convex_wall_line_easy": 13,
        "door_wall_line": 14,
        "line_reserve": 15,
        "outside_elevator_door_ground_line": 16,
        "inside_elevator_door_ground_line": 17,
        "outside_elevator_door_concave_wall_line": 18,
        "inside_elevator_door_concave_line": 19,
        "inside_elevator_door_convex_line": 20,
    }

    line_names = list(CLASSES.keys())

    Save_Vis = 1
    for i_num, img_name in enumerate(img_names):
        # img_name = "23c7a542-c646-11ec-acc7-72d8ab8687f9.png"
        img_path = os.path.join(img_root, img_name)
        # print(img_path)
        img = cv2.imread(img_path)
        img_show = copy.deepcopy(img)
        pred_lines = mlsd_detector.infer(img)

        for l in pred_lines:
            minx, miny = min(int(l[0]), int(l[2])), min(int(l[1]), int(l[3]))
            w, h = int(abs(l[0] - l[2]) / 2), int(abs(l[1] - l[3]) / 2)
            xc, yc = minx + w, miny + h

            conf = str(round(l[5], 5))
            cls = int(l[4])
            # if cls not in [0, 1]:
            #     continue

            txt = mlsd_detector.name_2_chinese[mlsd_detector.get_cls_name(l[4])[0]] + ":" + conf

            color = color_list[cls % len(color_list)]
            cv2.circle(img_show, (int(xc), int(yc)), 5, color, 5)
            cv2.line(img_show, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), color, 2, 16)
            img_show = mlsd_detector.draw_chinese(img_show, txt, (xc, yc), color, 20)

        if Save_Vis:
            dst_img_path = os.path.join(save_root, img_name.replace(".png", ".jpg"))
            img_blank = np.ones((img_show.shape[0], 30, 3), dtype=np.uint8) * 255
            # img_show = cv2.hconcat([img, img_blank,  img_show])
            # img_show = cv2.Vconcat([img, img_show])
            cv2.imwrite(dst_img_path, img_show)

        else:
            plt.imshow(img_show[:, :, ::-1])
            # plt.imshow(line_detector.debug_img)
            plt.show()
            exit(1)


if __name__ == "__main__":
    print("Start")
    show_line_demo()
    print("Done")