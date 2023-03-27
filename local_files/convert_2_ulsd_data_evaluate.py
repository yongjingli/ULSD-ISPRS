import os
import shutil
import cv2
import copy
from tqdm import tqdm
import numpy as np


from utils import get_curve_line, get_cls_names, color_list, \
    draw_curce_line_on_img, parse_calib
import sys
sys.path.insert(0, "../")
import util.bezier as bez
import util.camera as cam


def save_npz(prefix, image, lines, heatmap_size):
    image_size = (image.shape[1], image.shape[0])
    sx, sy = heatmap_size[0] / image_size[0], heatmap_size[1] / image_size[1]

    lines_mask = lines[:, 0, 1] > lines[:, -1, 1]
    lines[lines_mask] = lines[lines_mask, ::-1]
    lines[:, :, 0] = np.clip(lines[:, :, 0] * sx, 0, heatmap_size[0] - 1e-4)
    lines[:, :, 1] = np.clip(lines[:, :, 1] * sy, 0, heatmap_size[1] - 1e-4)
    juncs = np.concatenate((lines[:, 0], lines[:, -1]))
    juncs = np.round(juncs, 3)
    juncs = np.unique(juncs, axis=0)

    np.savez_compressed(
        f'{prefix}.npz',
        junc=juncs,
        line=lines
    )
    cv2.imwrite(f'{prefix}.png', image)



def convert_2_ulsd_data(base_infos):
    src_root = base_infos["src_root"]
    dst_root = base_infos["dst_root"]
    heatmap_size = base_infos["heatmap_size"]
    order = base_infos["order"]
    fish_eye = True if base_infos["type"] == "fisheye" else False

    vis_root = os.path.join(dst_root, "vis")
    if os.path.exists(dst_root):
        shutil.rmtree(dst_root)
    os.mkdir(dst_root)
    os.mkdir(vis_root)

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
        img_show2 = copy.deepcopy(img)
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

        if fish_eye:
            K_distort = calib_info["new_cam_k"]
            K = calib_info["cam_K"]
            D = calib_info["dsitort"]

            # undistort img
            # img_undistorted = copy.deepcopy(img)
            # img_undistorted = cv2.fisheye.undistortImage(img_undistorted, K_distort, D=D, Knew=K_distort)
            # img_undistorted = img_undistorted.astype(np.uint8)

            lines = []
            clses = []
            for curve_line in curve_lines:
                curve_points, curve_cls = curve_line
                point_start, point_end = curve_points[0], curve_points[-1]
                line = np.array([[point_start[0], point_start[1]],
                                 [point_end[0], point_end[1]]])
                lines.append(line)
                clses.append(curve_cls)

            coeff = {'K': K_distort, 'D': D}
            camera = cam.Fisheye(coeff)
            lines = np.array(lines)
            pts_list = camera.interp_line(lines, resolution=0.01)
            lines = bez.fit_line(pts_list, order=order)[0]
            for line, cls in zip(lines, clses):
                color = color_list[cls]
                start_point = line[0]
                for i, cur_point in enumerate(line[1:]):
                    x1, y1 = int(start_point[0]), int(start_point[1])
                    x2, y2 = int(cur_point[0]), int(cur_point[1])

                    cv2.circle(img_show2, (x1, y1), 8, color, -1)
                    cv2.line(img_show2, (x1, y1), (x2, y2), color, 3)
                    start_point = cur_point
        else:
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
            lines = np.asarray(bez.interp_line(lines, num=7))
            for line, cls in zip(lines, clses):
                color = color_list[cls]
                start_point = line[0]
                for i, cur_point in enumerate(line[1:]):
                    x1, y1 = int(start_point[0]), int(start_point[1])
                    x2, y2 = int(cur_point[0]), int(cur_point[1])

                    cv2.circle(img_show2, (x1, y1), 10, (0, 255, 0), -1)
                    cv2.line(img_show2, (x1, y1), (x2, y2), color, 3)
                    start_point = cur_point

        # save gt
        prefix = os.path.join(dst_root, s_base_name)
        save_npz(prefix, img, lines.copy(), heatmap_size)

        # img_show
        bez.insert_line(img, lines, color=[0, 0, 255])
        bez.insert_point(img, lines, color=[255, 0, 0], thickness=4)

        s_img_path = os.path.join(vis_root, s_base_name + ".jpg")
        cv2.imwrite(s_img_path, img)


if __name__ == "__main__":
    generate_dataset_infos = {
        "type": 'fisheye',   # 'pinhole''fisheye'  'spherical'
        "order": 6,   # 'pinhole''fisheye'  'spherical'
        "heatmap_size": [128, 128],
        "src_root": "/mnt/nas01/algorithm_data/data_disk/liyj/data/training_data/line_2023/pinhole_fisheye_line_raw_20230316",
        "dst_root": "/mnt/nas01/algorithm_data/data_disk/liyj/data/training_data/line_2023/ulsd_20230316",
    }

    convert_2_ulsd_data(generate_dataset_infos)


