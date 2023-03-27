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
import util.augment as aug


def save_npz(prefix, image, lines, centers, order, heatmap_size):
    n_pts = order + 1
    image_size = (image.shape[1], image.shape[0])
    heatmap_size = heatmap_size
    sx, sy = heatmap_size[0] / image_size[0], heatmap_size[1] / image_size[1]

    lines_mask = lines[:, 0, 1] > lines[:, -1, 1]
    lines[lines_mask] = lines[lines_mask, ::-1]
    lines[:, :, 0] = np.clip(lines[:, :, 0] * sx, 0, heatmap_size[0] - 1e-4)
    lines[:, :, 1] = np.clip(lines[:, :, 1] * sy, 0, heatmap_size[1] - 1e-4)
    centers[:, 0] = np.clip(centers[:, 0] * sx, 0, heatmap_size[0] - 1e-4)
    centers[:, 1] = np.clip(centers[:, 1] * sy, 0, heatmap_size[1] - 1e-4)

    jmap = np.zeros((1,) + heatmap_size[::-1], dtype=np.float32)
    joff = np.zeros((2,) + heatmap_size[::-1], dtype=np.float32)
    cmap = np.zeros((1,) + heatmap_size[::-1], dtype=np.float32)
    coff = np.zeros((2,) + heatmap_size[::-1], dtype=np.float32)
    eoff = np.zeros(((n_pts // 2) * 2, 2,) + heatmap_size[::-1], dtype=np.float32)
    lmap = np.zeros((1,) + heatmap_size[::-1], dtype=np.float32)

    juncs = np.concatenate((lines[:, 0], lines[:, -1]))
    juncs = np.round(juncs, 3)
    juncs = np.unique(juncs, axis=0)
    lpos = lines.copy()
    lneg = []

    if n_pts % 2 == 1:
        lines = np.delete(lines, n_pts // 2, axis=1)

    def to_int(x):
        return tuple(map(int, x))

    for c, pts in zip(centers, lines):
        v0, v1 = pts[0], pts[-1]

        cint = to_int(c)
        vint0 = to_int(v0)
        vint1 = to_int(v1)
        jmap[0, vint0[1], vint0[0]] = 1
        jmap[0, vint1[1], vint1[0]] = 1
        joff[:, vint0[1], vint0[0]] = v0 - vint0 - 0.5
        joff[:, vint1[1], vint1[0]] = v1 - vint1 - 0.5
        cmap[0, cint[1], cint[0]] = 1
        coff[:, cint[1], cint[0]] = c - cint - 0.5
        eoff[:, :, cint[1], cint[0]] = pts - c

    eoff = eoff.reshape((-1,) + heatmap_size[::-1])
    lmap[0] = bez.insert_line(lmap[0], lpos, color=255) / 255.0

    np.savez_compressed(
        f'{prefix}.npz',
        junc=juncs,
        lpos=lpos,
        lneg=lneg,
        jmap=jmap,
        joff=joff,
        cmap=cmap,
        coff=coff,
        eoff=eoff,
        lmap=lmap
    )
    cv2.imwrite(f'{prefix}.png', image)


def convert_2_ulsd_data(base_infos):
    src_root = base_infos["src_root"]
    dst_root = base_infos["dst_root"] + f'_{base_infos["order"]}'
    train_dst_root = os.path.join(dst_root, "train")
    test_dst_root = os.path.join(dst_root, "test")

    heatmap_size = tuple(base_infos["heatmap_size"])
    order = base_infos["order"]
    fish_eye = True if base_infos["type"] == "fisheye" else False

    vis_root = os.path.join(dst_root, "vis")
    if os.path.exists(dst_root):
        shutil.rmtree(dst_root)

    os.mkdir(dst_root)
    os.mkdir(vis_root)
    os.mkdir(train_dst_root)
    os.mkdir(test_dst_root)

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
    # tfs = [aug.Noop(), aug.HorizontalFlip(), aug.VerticalFlip(),
    #        aug.Compose([aug.HorizontalFlip(), aug.VerticalFlip()])]

    tfs = [aug.Noop(), aug.HorizontalFlip()]

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

        test = True if count % 10 == 0 else False
        if fish_eye:
            K_distort = calib_info["new_cam_k"]
            K = calib_info["cam_K"]
            D = calib_info["dsitort"]

            coeff = {'K': K_distort, 'D': D}
            camera = cam.Fisheye(coeff)
        else:
            camera = cam.Pinhole()

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
        lines = np.array(lines)

        if not test:
            for aug_i in range(len(tfs)):
                img_aug, lines_aug = tfs[aug_i](img, lines)

                pts_list = camera.interp_line(lines_aug)
                lines_aug = bez.fit_line(pts_list, order=2)[0]
                centers = lines_aug[:, 1]
                lines_aug = bez.fit_line(pts_list, order=order)[0]

                # pts_list = camera.interp_line(lines, resolution=0.01)
                # lines = bez.fit_line(pts_list, order=order)[0]
                for line, cls in zip(lines, clses):
                    color = color_list[cls]
                    start_point = line[0]
                    for i, cur_point in enumerate(line[1:]):
                        x1, y1 = int(start_point[0]), int(start_point[1])
                        x2, y2 = int(cur_point[0]), int(cur_point[1])

                        cv2.circle(img_show2, (x1, y1), 8, color, -1)
                        cv2.line(img_show2, (x1, y1), (x2, y2), color, 3)
                        start_point = cur_point

                prefix = os.path.join(train_dst_root, s_base_name + f'_{aug_i}')
                save_npz(prefix, img_aug, lines_aug.copy(), centers, order, heatmap_size)

                # img_show
                bez.insert_line(img_aug, lines_aug, color=[0, 0, 255])
                bez.insert_point(img_aug, lines_aug, color=[255, 0, 0], thickness=4)

                s_img_path = os.path.join(vis_root, s_base_name + f'_{aug_i}' + ".jpg")
                cv2.imwrite(s_img_path, img_aug)

        else:
            pts_list = camera.interp_line(lines)
            lines = bez.fit_line(pts_list, order=2)[0]
            centers = lines[:, 1]
            lines = bez.fit_line(pts_list, order=order)[0]

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
            prefix = os.path.join(test_dst_root, s_base_name)
            save_npz(prefix, img, lines.copy(), centers, order, heatmap_size)

            # img_show
            bez.insert_line(img, lines, color=[0, 0, 255])
            bez.insert_point(img, lines, color=[255, 0, 0], thickness=4)

            s_img_path = os.path.join(vis_root, s_base_name + ".jpg")
            cv2.imwrite(s_img_path, img)

        count = count + 1


if __name__ == "__main__":
    generate_dataset_infos = {
        "type": 'fisheye',   # 'pinhole''fisheye'  'spherical'
        "order": 6,   # 'pinhole''fisheye'  'spherical'
        "heatmap_size": [128, 128],
        "src_root": "/mnt/nas01/algorithm_data/data_disk/liyj/data/training_data/line_2023/pinhole_fisheye_line_raw_20230316",
        "dst_root": "/mnt/nas01/algorithm_data/data_disk/liyj/data/training_data/line_2023/ulsd_20230316",
    }

    convert_2_ulsd_data(generate_dataset_infos)