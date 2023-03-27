import sys
import torch
import cv2
import os
import shutil
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageFont, ImageDraw
from albumentations import Compose, Normalize
# sys.path.insert(0, "../mlsd_pytorch")
from network.lld_repvgg_large import Lld_Repvgg_Large

from local_utils import get_ll_cfg, sigmoid, cv2AddChineseText
from local_utils import curve_class_dict, curve_class_chinese_dict, \
                        color_list, line_class_dict


class Curve_Detector():
    def __init__(self, cfg, model_path=None, class_dict=None, conf_thre=0.5):
        # data basic info setting
        self.src_height = cfg.datasets.src_height
        self.src_width = cfg.datasets.src_width

        self.dst_height = cfg.datasets.input_h
        self.dst_width = cfg.datasets.input_w

        self.grid_size = cfg.train.grid_size
        self.scale_x = cfg.train.scale_x
        self.scale_y = cfg.train.scale_y

        self.pad_left = cfg.train.pad_left
        self.pad_right = cfg.train.pad_right
        self.pad_top = cfg.train.pad_top
        self.pad_bottom = cfg.train.pad_bottom

        self.test_aug = self._test_aug()
        self.map_size = (self.dst_height, self.dst_width)
        self.lane_map_range = cfg.train.lane_map_range
        self.conf_thre = conf_thre

        model = Lld_Repvgg_Large(cfg).cuda().eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)

        self.model = model
        self.cls_dict = class_dict
        self.cls_num = len(class_dict.keys())

        self.pred_confidence = None
        self.pred_offset_x = None
        self.pred_offset_y = None
        self.pred_emb = None
        self.pred_emb_id = None
        self.pred_cls = None
        self.img_debug = None
        self.line_kpi = False

    def _test_aug(self):
        aug = Compose(
            [
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ],
            p=1.0)
        return aug

    def pre_process(self, img):
        img_h, img_w, _ = img.shape
        assert img_h == self.src_height, "img_h not equel to 1080"
        assert img_w == self.src_width, "img_w not equel to 1920"

        res_height = int(self.src_height * self.scale_y)
        res_width = int(self.src_width * self.scale_x)
        remap_img = cv2.resize(img, (res_width, res_height))
        remap_img = cv2.copyMakeBorder(remap_img, self.pad_top, self.pad_bottom, self.pad_left, self.pad_left, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        self.img_debug = remap_img

        remap_img = cv2.cvtColor(remap_img, cv2.COLOR_BGR2RGB)  # !!!!!!!!
        img_norm = self.test_aug(image=remap_img)['image']
        img_norm = img_norm.transpose((2, 0, 1))
        img_norm = np.expand_dims(img_norm, axis=0)
        img_norm = torch.from_numpy(img_norm)
        img_norm = img_norm.to('cuda')
        return img_norm

    def infer(self, img):
        img_norm = self.pre_process(img)
        self.model_forward(img_norm)
        merge_lines = self.decode_curse_line()
        merge_lines = self.post_process(merge_lines)
        merge_lines = self.fitting_line(merge_lines)
        if self.line_kpi:
            merge_lines = self.get_kpi_line(merge_lines)
        return merge_lines

    def get_kpi_line(self, merge_lines):
        kpi_lines = []
        for i, merge_line in enumerate(merge_lines):
            for exact_line in merge_line:
                assert len(exact_line) == 2, "len(exact_line) != 2"
                start_point = exact_line[0]
                end_point = exact_line[1]
                line_cls = int(exact_line[0][4])

                conf = (start_point[2] + end_point[2]) * 0.5

                line = [int(start_point[0]), int(start_point[1]),
                        int(end_point[0]), int(end_point[1]), line_cls, conf]
                kpi_lines.append(line)
        return kpi_lines

    def post_process(self, merge_lines):
        for merge_line in merge_lines:
            for line in merge_line:
                for cur_point in line:
                    cur_point[0] = int(cur_point[0] / self.scale_x)
                    cur_point[1] = int(cur_point[1] / self.scale_y)
        return merge_lines

    def fitting_line(self, merge_lines):
        for i, merge_line in enumerate(merge_lines):
            output_lines = []
            for line in merge_line:
                line = np.array(line)
                # (x, y, conf, emb_id, cls)
                x = line[:, 0]
                y = line[:, 1]
                conf = line[:, 3]
                emb = line[:, 2]
                cls = line[:, 4]

                points = [[_x, _y] for _x, _y in zip(x, y)]
                points = np.array(points)
                output = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)

                # poly = np.polyfit(x, y, deg=1)

                x_min = np.min(x)
                x_max = np.max(x)

                y_min = np.min(y)
                y_max = np.max(y)

                if abs(x_max - x_min) > abs(y_max - y_min):
                    x1 = x_min
                    x2 = x_max
                    # y1 = x_min * poly[0] + poly[1]
                    # y2 = x_max * poly[0] + poly[1]

                    k = output[1] / output[0]
                    b = output[3] - k * output[2]

                    y1 = x_min * k + b
                    y2 = x_max * k + b
                    # print("xmax")
                else:
                    y1 = y_min
                    y2 = y_max

                    k = output[1] / output[0]
                    b = output[3] - k * output[2]

                    x1 = (y1 - b) / k
                    x2 = (y2 - b) / k
                    # print("ymax")
                conf = np.mean(conf)
                emb = np.mean(emb)

                # if cls[0] == 0:
                #     print(x1, x2)
                #     print(poly[0])
                #     print(poly[1])
                #     print(x)
                #     print(y)

                point1 = [int(x1), int(y1), conf, emb, cls[0]]
                point2 = [int(x2), int(y2), conf, emb, cls[0]]
                line = [point1, point2]
                output_lines.append(line)
            merge_lines[i] = output_lines
        return merge_lines

    def model_forward(self, img_norm):
        outputs = self.model(img_norm)
        self.pred_confidence = outputs[0, 0:1].cpu().detach().numpy()
        self.pred_offset_x = outputs[0, 1: 2].cpu().detach().numpy()
        self.pred_offset_y = outputs[0, 2: 3].cpu().detach().numpy()
        self.pred_emb = outputs[0, 3:4].cpu().detach().numpy()
        self.pred_emb_id = outputs[0, 4:5].cpu().detach().numpy()
        self.pred_cls = outputs[0, 5: 5 + self.cls_num].cpu().detach().numpy()

        self.pred_confidence = self.pred_confidence.clip(-20, 20)
        self.pred_offset_x = self.pred_offset_x.clip(-20, 20)
        self.pred_offset_y = self.pred_offset_y.clip(-20, 20)

        self.pred_confidence = sigmoid(self.pred_confidence)
        self.pred_cls = sigmoid(self.pred_cls)
        self.pred_offset_x = sigmoid(self.pred_offset_x) * (self.grid_size - 1)
        self.pred_offset_y = sigmoid(self.pred_offset_y) * (self.grid_size - 1)

        self.pred_offset_x = self.pred_offset_x.round().astype(np.int).clip(0, self.grid_size - 1)
        self.pred_offset_y = self.pred_offset_y.round().astype(np.int).clip(0, self.grid_size - 1)

    def decode_curse_line(self):
        _, h, w = self.pred_offset_x.shape
        pred_grid_x = np.arange(w).reshape(1, 1, w).repeat(h, axis=1) * self.grid_size
        pred_grid_y = np.arange(h).reshape(1, h, 1).repeat(w, axis=2) * self.grid_size
        pred_x = pred_grid_x + self.pred_offset_x
        pred_y = pred_grid_y + self.pred_offset_y

        min_y, max_y = self.lane_map_range
        mask = np.zeros_like(self.pred_confidence, dtype=np.bool)
        mask[:, min_y:max_y, :] = self.pred_confidence[:, min_y:max_y, :] > self.conf_thre

        connect_lines = self.connect_line_points(mask[0], self.pred_emb[0], pred_x[0], pred_y[0],
                                 self.pred_confidence[0], self.pred_emb_id[0], self.pred_cls)

        connect_lines = self.get_line_cls(connect_lines)
        merge_connect_lines = self.merge_same_line(connect_lines)

        return merge_connect_lines

    def get_line_cls(self, connect_lines):
        output_connect_lines = []
        for connect_line in connect_lines:
            points = np.array(connect_line)
            points_cls = points[:, 4]
            cls = np.argmax(np.bincount(points_cls.astype(np.int32)))
            points[:, 4] = cls
            output_connect_lines.append(list(points))
        return output_connect_lines

    def merge_same_line(self, connect_lines, pull_margin=0.5):
        lines = []
        lines_emb_mean = []
        line_numbers = []
        for connect_line in connect_lines:
            points = np.array(connect_line)
            emb_mean = np.mean(points[:, 3])

            id = None
            min_dist = 10000
            for i, line_emb_mean in enumerate(lines_emb_mean):
                distance = abs(line_emb_mean - emb_mean)
                if distance < pull_margin and distance < min_dist:
                    id = i
                    min_dist = distance

            if id == None:
                lines.append([connect_line])
                lines_emb_mean.append(emb_mean)
                line_numbers.append(1)
            else:
                lines[id].append(connect_line)
                lines_emb_mean[id] = (lines_emb_mean[id] * line_numbers[id] + emb_mean) / (
                        line_numbers[id] + 1
                )
                line_numbers[id] += 1

        return lines

    def connect_line_points(self, mask_map, embedding_map, x_map, y_map,
                            confidence_map, pred_emb_id, pred_cls_map, line_maximum=10):
        ys, xs = np.nonzero(mask_map)
        ys, xs = np.flipud(ys), np.flipud(xs)  # 优先将y大的点排在前面
        ebs = embedding_map[ys, xs]
        raw_lines = self.cluster_line_points(xs, ys, ebs)

        raw_lines = self.remove_short_lines(raw_lines)
        raw_lines = self.remove_far_lines(raw_lines)

        exact_lines = []
        for each_line in raw_lines:
            single_line = self.serialize_single_line(each_line, x_map, y_map,
                                                confidence_map, pred_emb_id, pred_cls_map)
            if len(single_line) > 0:
                exact_lines.append(single_line)
        if len(exact_lines) == 0:
            return []
        exact_lines = sorted(exact_lines, key=lambda l: len(l), reverse=True)
        exact_lines = (
            exact_lines[: line_maximum]
            if len(exact_lines) > line_maximum
            else exact_lines
        )
        exact_lines = sorted(exact_lines, key=lambda l: l[0][0], reverse=False)
        # exact_lines = sorted(exact_lines, key=lambda l: np.array(l)[:, 0].mean(), reverse=False)
        return exact_lines

    def cluster_line_points(self, xs, ys, embeddings, pull_margin=0.8):
        lines = []
        embedding_means = []
        point_numbers = []
        for x, y, eb in zip(xs, ys, embeddings):
            id = None
            min_dist = 10000
            for i, eb_mean in enumerate(embedding_means):
                distance = abs(eb - eb_mean)
                if distance < pull_margin and distance < min_dist:
                    id = i
                    min_dist = distance
            if id == None:
                lines.append([(x, y)])
                embedding_means.append(eb)
                point_numbers.append(1)
            else:
                lines[id].append((x, y))
                embedding_means[id] = (embedding_means[id] * point_numbers[id] + eb) / (
                        point_numbers[id] + 1
                )
                point_numbers[id] += 1
        return lines

    def remove_short_lines(self, lines, long_line_thresh=4):
        long_lines = []
        for line in lines:
            if len(line) >= long_line_thresh:
                long_lines.append(line)
        return long_lines

    def remove_far_lines(self, lines, far_line_thresh=10):
        near_lines = []
        for line in lines:
            for point in line:
                if point[1] >= far_line_thresh:
                    near_lines.append(line)
                    break
        return near_lines

    def serialize_single_line(self, single_line, x_map, y_map, confidence_map, pred_emb_id_map, pred_cls_map):
        existing_points = single_line.copy()
        piecewise_lines = []
        while len(existing_points) > 0:
            existing_points = self.remove_isolated_points(existing_points)
            if len(existing_points) == 0:
                break
            y = np.array(existing_points)[:, 1].max()
            selected_points, alternative_points = [], []
            for e_pnt in existing_points:
                if e_pnt[1] == y and len(selected_points) == 0:
                    selected_points.append(e_pnt)
                else:
                    alternative_points.append(e_pnt)
            y -= 1
            while len(alternative_points) > 0:
                near_points, far_points = [], []
                for a_pnt in alternative_points:
                    if a_pnt[1] >= y:
                        near_points.append(a_pnt)
                    else:
                        far_points.append(a_pnt)
                if len(near_points) == 0:
                    break
                selected_points, outliers = self.select_function_points(
                    selected_points, near_points
                )
                if len(outliers) == len(near_points):
                    break
                else:
                    alternative_points = outliers + far_points
                    y -= 1
            selected_points = self.extend_endpoints(selected_points, single_line)
            piecewise_line = self.arrange_points_to_line(
                selected_points, x_map, y_map, confidence_map, pred_emb_id_map, pred_cls_map
            )
            # piecewise_line = self.fit_points_to_line(selected_points, x_map, y_map, confidence_map)  # Curve Fitting
            piecewise_lines.append(piecewise_line)
            existing_points = alternative_points
        if len(piecewise_lines) == 0:
            return []
        elif len(piecewise_lines) == 1:
            exact_lines = piecewise_lines[0]
        else:
            exact_lines = self.connect_piecewise_lines(piecewise_lines)[0]
        if exact_lines[0][1] < exact_lines[-1][1]:
            exact_lines.reverse()
        return exact_lines

    def remove_isolated_points(self, line_points):
        line_points = np.array(line_points)
        valid_points = []
        for point in line_points:
            distance = abs(point - line_points).max(axis=1)
            if np.any(distance == 1):
                valid_points.append(point.tolist())
        return valid_points

    def select_function_points(self, selected_points, near_points):
        while len(near_points) > 0:
            added_points = []
            for n_pnt in near_points:
                for s_pnt in selected_points:
                    distance = max(abs(n_pnt[0] - s_pnt[0]), abs(n_pnt[1] - s_pnt[1]))
                    if distance == 1:
                        vertical_distance = self.compute_vertical_distance(n_pnt, selected_points)
                        if vertical_distance <= 1:
                            selected_points = [n_pnt] + selected_points
                            added_points.append(n_pnt)
                            break
            if len(added_points) == 0:
                break
            else:
                near_points = [n_pnt for n_pnt in near_points if n_pnt not in added_points]
        return selected_points, near_points

    def extend_endpoints(self, selected_points, single_line):
        min_x, max_x = 10000, 0
        left_endpoints, right_endpoints = [], []
        for s_pnt in selected_points:
            if s_pnt[0] == min_x:
                left_endpoints.append(s_pnt)
            elif s_pnt[0] < min_x:
                left_endpoints.clear()
                left_endpoints.append(s_pnt)
                min_x = s_pnt[0]
            if s_pnt[0] == max_x:
                right_endpoints.append(s_pnt)
            elif s_pnt[0] > max_x:
                right_endpoints.clear()
                right_endpoints.append(s_pnt)
                max_x = s_pnt[0]
        for x, y in left_endpoints:
            while (x - 1, y) in single_line:
                selected_points.append((x - 1, y))
                x -= 1
        for x, y in right_endpoints:
            while (x + 1, y) in single_line:
                selected_points.append((x + 1, y))
                x += 1
        return selected_points

    def arrange_points_to_line(self, selected_points, x_map, y_map, confidence_map, pred_emb_id_map, pred_cls_map,
                               map_size=(1920, 1080)):
        selected_points = np.array(selected_points)
        xs, ys = selected_points[:, 0], selected_points[:, 1]
        image_xs = x_map[ys, xs]
        image_ys = y_map[ys, xs]
        confidences = confidence_map[ys, xs]

        pred_cls_map = np.argmax(pred_cls_map, axis=0)
        emb_ids = pred_emb_id_map[ys, xs]
        clses = pred_cls_map[ys, xs]

        indices = image_xs.argsort()
        image_xs = image_xs[indices]
        image_ys = image_ys[indices]
        confidences = confidences[indices]
        h, w = map_size
        line = []
        for x, y, conf, emb_id, cls in zip(image_xs, image_ys, confidences, emb_ids, clses):
            x = min(x, w - 1)
            y = min(y, h - 1)
            line.append((x, y, conf, emb_id, cls))
        return line

    def connect_piecewise_lines(self, piecewise_lines, endpoint_distance=16):
        long_lines = piecewise_lines
        final_lines = []
        while len(long_lines) > 1:
            current_line = long_lines[0]
            current_endpoints = [current_line[0], current_line[-1]]
            other_lines = long_lines[1:]
            other_endpoints = []
            for o_line in other_lines:
                other_endpoints.append(o_line[0])
                other_endpoints.append(o_line[-1])
            point_ids = [None, None]
            min_dist = 10000
            for i, c_end in enumerate(current_endpoints):
                for j, o_end in enumerate(other_endpoints):
                    distance = self.compute_point_distance(c_end, o_end)
                    if distance < min_dist:
                        point_ids[0] = i
                        point_ids[1] = j
                        min_dist = distance
            if min_dist < endpoint_distance:
                adjacent_line = other_lines[point_ids[1] // 2]
                other_lines.remove(adjacent_line)
                if point_ids[0] == 0 and point_ids[1] % 2 == 0:
                    adjacent_line.reverse()
                    left_line = adjacent_line
                    right_line = current_line
                elif point_ids[0] == 0 and point_ids[1] % 2 == 1:
                    left_line = adjacent_line
                    right_line = current_line
                elif point_ids[0] == 1 and point_ids[1] % 2 == 0:
                    left_line = current_line
                    right_line = adjacent_line
                elif point_ids[0] == 1 and point_ids[1] % 2 == 1:
                    left_line = current_line
                    adjacent_line.reverse()
                    right_line = adjacent_line
                long_lines = other_lines + [left_line + right_line]
            else:
                final_lines.append(current_line)
                long_lines = other_lines
        final_lines.append(long_lines[0])
        final_lines = sorted(final_lines, key=lambda l: len(l), reverse=True)
        return final_lines

    def compute_point_distance(self, point_0, point_1):
        distance = np.sqrt((point_0[0] - point_1[0]) ** 2 + (point_0[1] - point_1[1]) ** 2)
        return distance

    def compute_vertical_distance(self, point, selected_points):
        vertical_points = [s_pnt for s_pnt in selected_points if s_pnt[0] == point[0]]
        if len(vertical_points) == 0:
            return 0
        else:
            vertical_distance = 10000
            for v_pnt in vertical_points:
                distance = abs(v_pnt[1] - point[1])
                vertical_distance = distance if distance < vertical_distance else vertical_distance
            return vertical_distance


def show_curve_demo():
    root = "/mnt/data10/liyj/programs/ULSD-ISPRS/dataset/lld_20230316/train/images"
    save_root = "/mnt/data10/liyj/debug"
    if os.path.exists(save_root):
        shutil.rmtree(save_root)
    os.mkdir(save_root)

    model_path = "/mnt/data10/liyj/programs/ULSD-ISPRS/model/lld_20230320/latest.pth"

    img_names = [name for name in os.listdir(root)
                 if name.endswith('.png') or name.endswith('.jpg')]
    # model set
    cfg_path = "/mnt/data10/liyj/programs/ULSD-ISPRS/configs/lld_cfg.yaml"
    cfg = get_cfg(cfg_path)

    classes = curve_class_dict
    curese_lld_detector = Curve_Detector(cfg, model_path, classes, conf_thre=0.5)

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

    show_num = 1
    for i_num, img_name in enumerate(img_names):
        img_name = "0b320388-79d5-4f2d-9b86-a9c9bb1f1a87.jpg"

        img_path = os.path.join(root, img_name)
        img = cv2.imread(img_path)

        merge_lines = curese_lld_detector.infer(img)
        img_show = img.copy()

        count = 0
        for i, merge_line in enumerate(merge_lines):
            count = count % len(color_list)
            color = color_list[count]

            for exact_line in merge_line:
                pre_point = exact_line[0]
                txt_i = len(exact_line) // 2
                txt_x = int(exact_line[txt_i][0])
                txt_y = int(exact_line[txt_i][1])

                line_cls = int(exact_line[txt_i][4])
                # if line_cls not in [0, 1]:
                #     continue

                line_name = line_names[line_cls]
                img_show = cv2AddChineseText(img_show, line_name, (txt_x, txt_y), color[::-1], 30)

                for cur_point in exact_line[1:]:
                    x1, y1 = int(pre_point[0]), int(pre_point[1])
                    x2, y2 = int(cur_point[0]), int(cur_point[1])

                    cv2.circle(img_show, (x1, y1), 5, color, 1)
                    cv2.line(img_show, (x1, y1), (x2, y2), color, 5)
                    pre_point = cur_point
            count += 1

        if 1:
            plt.subplot(2, 3, 1)
            plt.title("input img")
            plt.imshow(img[:, :, ::-1])

            plt.subplot(2, 3, 2)
            plt.title("curve det result")
            plt.imshow(img_show[:, :, ::-1])
            save_path = os.path.join(save_root, img_name)
            cv2.imwrite(save_path, img_show)

            cls_num = 0
            cls_confident = curese_lld_detector.pred_confidence[cls_num, :, :]
            cls_emb = curese_lld_detector.pred_emb[cls_num, :, :]
            pred_emb_id = curese_lld_detector.pred_emb_id[cls_num, :, :]
            mask = cls_confident < 0.3
            cls_emb[mask] = 0
            pred_emb_id[mask] = 0

            plt.subplot(2, 3, 3)
            plt.title("pred confident map")
            plt.imshow(cls_confident)

            plt.subplot(2, 3, 4)
            plt.title("pred embedding map")
            plt.imshow(cls_emb)

            plt.subplot(2, 3, 5)
            plt.title("img_debug")
            # plt.imshow(curese_lld_detector.img_debug[:, :, ::-1])
            plt.imshow(pred_emb_id)

            plt.subplot(2, 3, 6)
            plt.title("pred cls map")

            # debbug cls
            pred_cls = curese_lld_detector.pred_cls[8, :, :]
            pred_cls[mask] = 0
            plt.imshow(pred_cls)
            plt.show()

            plt.subplot(2, 1, 1)
            pred_cls = curese_lld_detector.pred_cls[8, :, :]
            pred_cls[mask] = 0
            plt.imshow(pred_cls)

            plt.subplot(2, 1, 2)
            pred_cls = curese_lld_detector.pred_cls[4, :, :]
            pred_cls[mask] = 0
            plt.imshow(pred_cls)

            vis_img_path = os.path.join(save_root, img_name.split('.')[0] + "_vis.png")
            plt.savefig(vis_img_path)
            plt.show()
        else:
            # plt.imshow(curese_lld_detector.img_debug[:, :, ::-1])
            # plt.imshow(img_show[:, :, ::-1])
            save_path = os.path.join(save_root, img_name)
            cv2.imwrite(save_path, img_show)
        # plt.show()

        if i_num + 1 >= show_num:
            print("Done one")
            exit(1)

    print("Done")



if __name__ == "__main__":
    print("Start...")
    show_curve_demo()
    print("Done...")