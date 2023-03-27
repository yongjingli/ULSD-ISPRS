import os
import sys
import argparse
import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
from PIL import Image, ImageFont, ImageDraw

# sys.path.insert(0, "../mlsd_pytorch")
# from cfg.default import get_cfg_defaults
from configs.lld_default import get_cfg_defaults as get_lld_cfg_defaults
from configs.mlsd_default import get_cfg_defaults as get_mlsd_cfg_defaults

curve_class_dict = {
    'road_plant_line': 0,
    'road_shoulder_line': 1,
    'road_parterre_line': 2,
    'road_building_line': 3,
    'road_fence_line': 4,
    'road_stair_line': 5,
    'road_sewer_line': 6,
    'road_other_canspan_line': 7,
    'road_other_cannotspan_line': 8,
    'off_the_shoulder_line': 9,
    'stair_line': 10,
    'uncertain_line': 11
}

curve_class_chinese_dict = {
    '道路植被沿线': 0,
    '道路路肩沿线': 1,
    '道路花坛沿线': 2,
    '道路建筑物沿线': 3,
    '道路栅栏沿线': 4,
    '道路楼梯沿线': 5,
    '道路下水道沿线': 6,
    '道路其他可跨越沿线': 7,
    '道路其他不可跨越沿线': 8,
    '路肩外沿线': 9,
    '楼梯沿线': 10,
    '不确定沿线': 11
}

line_dic = {
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
    'sofa_ground_curve': 11
}

line_class_dict = {
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
    '沙发地曲线': 11
}

color_list = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (10, 215, 255), (0, 255, 255),
              (221, 160, 221),  (128, 0, 128), (203, 192, 255), (238, 130, 238), (0, 69, 255),
              (130, 0, 75), (255, 255, 0), (250, 51, 153), (214, 112, 218), (255, 165, 0),
              (169, 169, 169), (18, 74, 115),
              (240, 32, 160), (192, 192, 192), (112, 128, 105), (105, 128, 128),
              ]

# [纯红、纯绿、纯蓝、金色、纯黄、
# 梅红色、紫色、粉色、灰色、藏青色、
# 紫罗兰、橙红色、湖紫色、淡紫色、天蓝]
# 深灰色、标土棕、
# 紫色、冷灰、石板灰、暖灰色、


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        default= os.path.dirname(os.path.abspath(__file__))+ '/configs/mobilev2_mlsd_large_512_base2_bsize24.yaml',
                        type=str,
                        help="")
    return parser.parse_args()


def get_ll_cfg(cfg_path):
    cfg = get_lld_cfg_defaults()
    # print(cfg)
    args = get_args()
    args.config = cfg_path
    if args.config.endswith('\r'):
        args.config = args.config[:-1]
    print('using config: ', args.config.strip())

    cfg.merge_from_file(args.config)
    print(cfg)
    return cfg


def get_mlsd_cfg(cfg_path):
    cfg = get_mlsd_cfg_defaults()
    # print(cfg)
    args = get_args()
    args.config = cfg_path
    if args.config.endswith('\r'):
        args.config = args.config[:-1]
    print('using config: ', args.config.strip())

    cfg.merge_from_file(args.config)
    print(cfg)
    return cfg



def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_key(dict, value):
    return [k for k, v in dict.items() if v == value]


def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "/mnt/data10/liyj/programs/mlsd_with_cls/local_files/simhei.ttf", textSize, encoding="utf-8")

    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    return img


def line_nms(lines_with_cls_np):
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

                if l1_cls != l2_cls:
                    continue

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
                # print("distance:", pt_to_line_distance)

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
                iou = max(iou_x, iou_y)

                iou_thr = 0.3
                center_to_line_thr = 10
                is_overlap = False
                if pt_to_line_distance < center_to_line_thr and iou > iou_thr:
                    is_overlap = True

                line_dist_thr = 30
                center_dist_thr = 8
                if line_dist < line_dist_thr or center_dist < center_dist_thr or is_overlap:
                    mask[j] = False

        lines_with_cls_np = lines_with_cls_np[mask]
    return lines_with_cls_np
