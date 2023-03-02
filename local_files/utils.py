import os
import cv2
import shutil
import json
import numpy as np
from tqdm import tqdm
from copy import copy


color_list = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (10, 215, 255), (0, 255, 255),
              (221, 160, 221),  (128, 0, 128), (203, 192, 255), (238, 130, 238), (0, 69, 255),
              (130, 0, 75), (255, 255, 0), (250, 51, 153), (214, 112, 218), (255, 165, 0),
              (169, 169, 169), (18, 74, 115),
              (240, 32, 160), (192, 192, 192), (112, 128, 105), (105, 128, 128),
              ]

class_dict = {
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

class_dict_chinese = {
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

CLASS_NUM = len(class_dict.keys())


def get_cls_names():
    cls_names = dict()
    for k, v in class_dict.items():
        cls_names.update({v: k})
    return cls_names


def parse_object_map(object_map, img):
    object_ids = np.unique(object_map)
    object_lines = []
    for object_id in object_ids:
        if object_id == 0:
            continue
        indx_y, indx_x = np.where(object_map == object_id)
        indx_x = indx_x.reshape(-1, 1)
        indx_y = indx_y.reshape(-1, 1)
        object_line = np.concatenate([indx_y, indx_x], axis=1)
        # img_dilate_mask = get_img_dilate_mask(copy.deepcopy(img))
        # img_dilate_mask = img
        #
        # img_value = img_dilate_mask[indx_y, indx_x, :]

        # mask = np.bitwise_or(img_value[:, 0, 0] != 0,
        #                      img_value[:, 0, 1] != 0,
        #                      img_value[:, 0, 2] != 0,)
        # object_line = object_line[mask]
        object_lines.append(object_line)
    return object_lines


def parse_cls_map(cls_map, object_lines):
    cls_lines =[]
    for object_line in object_lines:
        line_y = object_line[:, 0]
        line_x = object_line[:, 1]

        line_cls = cls_map[line_y, line_x]
        cls_lines.append(line_cls)
    return cls_lines


def parse_order_map(order_map, object_lines):
    order_lines =[]
    for object_line in object_lines:
        line_y = object_line[:, 0]
        line_x = object_line[:, 1]

        order_line = order_map[line_y, line_x]
        order_lines.append(order_line)
    return order_lines


def get_curve_line(img, object_map, cls_map, order_map):
    object_lines = parse_object_map(object_map, img)
    cls_lines = parse_cls_map(cls_map, object_lines)
    order_lines = parse_order_map(order_map, object_lines)
    curve_lines = []
    for object_line, cls_line, order_line in zip(object_lines, cls_lines, order_lines):
        cls_id = np.argmax(np.bincount(cls_line)) - 1
        order_index = np.argsort(order_line)
        object_line_order = object_line[order_index]

        # 将点的坐标转为(w, h)的形式，也就是(x, y)
        object_line_order[:, :] = object_line_order[:, ::-1]
        curve_lines.append([object_line_order, cls_id])

    return curve_lines


def draw_curce_line_on_img(img, points, cls_name, color=(0, 255, 0)):
    pre_point = points[0]
    for i, cur_point in enumerate(points[1:]):
        x1, y1 = int(pre_point[0]), int(pre_point[1])
        x2, y2 = int(cur_point[0]), int(cur_point[1])

        # cv2.circle(img, (x1, y1), 1, color, 1)
        cv2.line(img, (x1, y1), (x2, y2), color, 3)
        pre_point = cur_point
        # show order
        # if i % 100 == 0:
        #     img = cv2.putText(img, str(i), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1)

    txt_i = len(points) // 2
    txt_x = int(points[txt_i][0])
    txt_y = int(points[txt_i][1])
    # img = cv2.putText(img, cls_name, (txt_y, txt_x), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 2)

    return img


def parse_calib(calib_path):
    calib_infos = dict()
    with open(calib_path, 'r') as fp:
        calib_info = json.load(fp)

    cam_K = np.array(calib_info['cam_K']).reshape(3, 3)

    baseline = calib_info['baseline']
    dsitort = calib_info["dsitort"]

    calib_infos["cam_K"] = cam_K
    calib_infos["baseline"] = baseline
    calib_infos["dsitort"] = np.array(dsitort)

    if "new_cam_k" in calib_info.keys():
        calib_infos["new_cam_k"] = np.array(calib_info['new_cam_k']).reshape(3, 3)

    return calib_infos