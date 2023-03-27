import os
import shutil
import cv2
import json
import copy
from tqdm import tqdm
import numpy as np

from utils import get_curve_line, get_cls_names, color_list, \
    draw_curce_line_on_img, parse_calib


def convert_2_lld_data(base_infos):
    src_root = base_infos["src_root"]
    dst_root = base_infos["dst_root"]
    dst_w = base_infos["dst_w"]
    dst_h = base_infos["dst_h"]

    if os.path.exists(dst_root):
        shutil.rmtree(dst_root)
    os.mkdir(dst_root)

    for tmp in ['train', 'test']:
        tmp_path = os.path.join(dst_root, tmp)
        os.mkdir(tmp_path)

    for tmp in ['images', 'jsons']:
        os.mkdir(os.path.join(dst_root, 'train', tmp))
        os.mkdir(os.path.join(dst_root, 'test', tmp))

    fish_eye = True if base_infos["type"] == "fisheye" else False
    s_img_root = os.path.join(src_root, "image")
    s_img_show_root = os.path.join(src_root, "image_show")
    s_line_object_root = os.path.join(src_root, "line_object")
    s_line_cls_root = os.path.join(src_root, "line_cls")
    s_line_order_root = os.path.join(src_root, "line_order")
    s_calib_root = os.path.join(src_root, "calib")

    s_distort_img_root = os.path.join(src_root, "distort_image")
    s_distort_img_show_root = os.path.join(src_root, "distort_image_show")
    s_distort_line_object_root = os.path.join(src_root, "distort_line_object")
    s_distort_line_cls_root = os.path.join(src_root, "distort_line_cls")
    s_distort_line_order_root = os.path.join(src_root, "distort_line_order")
    s_distort_calib_root = os.path.join(src_root, "distort_calib")

    img_names = [name for name in os.listdir(s_img_root)
                 if name.split(".")[-1] in ["jpg", "png"]]
    count = 0

    for img_name in tqdm(img_names):
        s_base_name = img_name[:-4]

        if fish_eye:
            img_path = os.path.join(s_distort_img_root, s_base_name + ".jpg")
            object_map_path = os.path.join(s_distort_line_object_root, s_base_name + ".npy")
            cls_map_path = os.path.join(s_distort_line_cls_root, s_base_name + ".npy")
            order_map_path = os.path.join(s_distort_line_order_root, s_base_name + ".npy")
            calib_path = os.path.join(s_distort_calib_root, s_base_name + ".json")

        else:
            img_path = os.path.join(s_img_root, s_base_name + ".jpg")
            object_map_path = os.path.join(s_line_object_root, s_base_name + ".npy")
            cls_map_path = os.path.join(s_line_cls_root, s_base_name + ".npy")
            order_map_path = os.path.join(s_line_order_root, s_base_name + ".npy")
            calib_path = os.path.join(s_calib_root, s_base_name + ".json")

        img = cv2.imread(img_path)
        img_show = copy.deepcopy(img)
        object_map = np.load(object_map_path)
        cls_map = np.load(cls_map_path)
        order_map = np.load(order_map_path)
        calib_info = parse_calib(calib_path)
        curve_lines = get_curve_line(img, object_map, cls_map, order_map)

        cls_names = get_cls_names()
        for curve_id, curve_line in enumerate(curve_lines):
            points, cls = curve_line
            color = color_list[cls]
            cls_name = cls_names[cls]
            img_show = draw_curce_line_on_img(img_show, points, cls_name, color)

        test = True if count % 10 == 0 else False
        lines = []
        clses = []
        for curve_line in curve_lines:
            curve_points, curve_cls = curve_line
            point_start, point_end = curve_points[0], curve_points[-1]
            line = np.array([[point_start[0], point_start[1]],
                             [point_end[0], point_end[1]]])
            lines.append(line)
            clses.append(curve_cls)
        lines = np.array(lines)

        root_name = "test" if test else "train"
        s_root = os.path.join(dst_root, root_name)

        # gt
        img_h, img_w, _ = img.shape
        img_res = cv2.resize(img, (dst_w, dst_h))
        sx, sy = dst_w / img_w, dst_h / img_h

        new_json = {
            'version': '4.5.6',
            'flags': {},
            'shapes': [],
            'imagePath': s_base_name + '.png',
            'imageData': None,
            'imageHeight': dst_w,
            'imageWidth': dst_h,
        }

        for curve_id, curve_line in enumerate(curve_lines):
            # id 用来区分不同的线段是否属于同一连续条线
            # 因为这里是从直线检测那里迁移过来，目前认为就是只包含一条
            # 在曲线中是通过标注出来的，属于同一条线的线段具有相同的id
            id = curve_id

            points, cls = curve_line
            cls_name = cls_names[cls]

            # resize points
            points[:, 0] = np.clip(points[:, 0] * sx, 0, dst_w - 1e-4)
            points[:, 1] = np.clip(points[:, 1] * sy, 0, dst_h - 1e-4)

            item = {
                "label": cls_name,
                "points": points.tolist(),
                "group_id": None,
                "shape_type": "line",
                "flag": {},
                "id": id,
            }
            new_json['shapes'].append(item)

        dst_json_path = os.path.join(s_root, "jsons", s_base_name + ".json")
        with open(dst_json_path, 'w') as sf:
            json.dump(new_json, sf)

        dst_png_path = os.path.join(s_root, "images", s_base_name + ".jpg")
        cv2.imwrite(dst_png_path, img_res)
        # shutil.copy(img_path, dst_png_path)

        count = count + 1


if __name__ == "__main__":
    generate_dataset_infos = {
        "type": 'fisheye',
        "dst_w": 1920,
        "dst_h": 1080,
        "src_root": "/mnt/nas01/algorithm_data/data_disk/liyj/data/training_data/line_2023/pinhole_fisheye_line_raw_20230316",
        "dst_root": "/mnt/data10/liyj/programs/ULSD-ISPRS/dataset/lld_20230316",
    }

    convert_2_lld_data(generate_dataset_infos)