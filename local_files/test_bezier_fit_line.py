import os
import cv2
import shutil
import json
import numpy as np
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt

from utils import get_curve_line, get_cls_names, color_list, \
    draw_curce_line_on_img, parse_calib
import sys
sys.path.insert(0, "../")
import util.bezier as bez
import util.camera as cam



def test_bezier_fit_line():
    # root = "./data/indoor_line"
    # root = "./data/indoor_line_no_crop"
    # root = "./data/indoor_line_corner_crop"

    root = "/mnt/data10/liyj/data/debug"
    dst_root = os.path.join(root, "usd_show")
    if not os.path.exists(dst_root):
        os.mkdir(dst_root)

    s_img_root = os.path.join(root, "image")
    s_img_show_root = os.path.join(root, "image_show")
    s_line_object_root = os.path.join(root, "line_object")
    s_line_cls_root = os.path.join(root, "line_cls")
    s_line_order_root = os.path.join(root, "line_order")
    s_calib_root = os.path.join(root, "calib")

    s_distort_img_root = os.path.join(root, "distort_image")
    s_distort_img_show_root = os.path.join(root, "distort_image_show")
    s_distort_line_object_root = os.path.join(root, "distort_line_object")
    s_distort_line_cls_root = os.path.join(root, "distort_line_cls")
    s_distort_line_order_root = os.path.join(root, "distort_line_order")
    s_distort_calib_root = os.path.join(root, "distort_calib")

    img_names = [name for name in os.listdir(s_img_root)
                 if name.split(".")[-1] in ["jpg", "png"]]
    fish_eye = True

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
            lines = bez.fit_line(pts_list, order=6)[0]
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

        s_img_path = os.path.join(dst_root, s_base_name + ".jpg")
        cv2.imwrite(s_img_path, img_show2)
        # plt.imshow(img)
        # plt.imshow(img_show2)
        # plt.imshow(img_undistorted)
        # plt.show()
        # exit(1)


def test_bezier():
    from scipy.special import comb
    order = 1
    num = 7
    resolution = 1.0
    line = np.array([(5, 5), (5, 20)])

    p = comb(order, np.arange(order + 1))
    k = np.arange(0, order + 1)
    t = np.linspace(0, 1, order + 1)[:, None]
    coeff_matrix = p * (t ** k) * ((1 - t) ** (order - k))
    inv_coeff_matrix = np.linalg.inv(coeff_matrix)

    control_points = np.matmul(inv_coeff_matrix, line)
    K = int(round(max(abs(line[-1] - line[0])) / resolution)) + 1 if num is None else num
    t = np.linspace(0, 1, K)[:, None]
    coeff_matrix = p * (t ** k) * ((1 - t) ** (order - k))
    pts = np.matmul(coeff_matrix, control_points)


if __name__ == "__main__":
    print("Start")
    test_bezier_fit_line()
    # test_bezier()
    print("end")









