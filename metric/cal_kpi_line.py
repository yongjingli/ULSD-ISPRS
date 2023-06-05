from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import torch
import sys
import cv2
import os
import json
import numpy as np
import copy
import shutil

from utils import pred_lines_and_cls

from local_utils import get_cfg, sigmoid, cv2AddChineseText
from local_utils import curve_class_dict, curve_class_chinese_dict, \
                        color_list, line_class_dict
from curve_detoctor2 import Curve_Detector

from line_detector import LineDetector

sys.path.insert(0, "../mlsd_pytorch")
# from models.mbv2_mlsd_large import MobileV2_MLSD_Large
from tmp_models.mbv2_mlsd_large import MobileV2_MLSD_Large


class LineKpiCalculator(object):
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

    def __init__(self):
        self.name_2_chinese = self.get_name_2_chinese()
        self.kpi_names = ["over-all"] + list(self.class_names.keys())
        self.kip_infos = self.init_kpi_infos()

        self.line_thickness = 16      #12
        self.iou = 0.1   # 0.2
        self.count = 0

        self.vis_cls_names = ['concave_wall_line', 'convex_wall_line', 'wall_ground_line',
                              'cabinet_wall_line', 'cabinet_ground_line', 'sofa_ground_line',
                              'stair_line', 'ceiling_line', 'wall_ground_curve', 'stair_curve',
                              'cabinet_ground_curve', 'sofa_ground_curve', 'concave_wall_line_easy',
                              'convex_wall_line_easy', 'door_wall_line', 'line_reserve',
                              'outside_elevator_door_ground_line', 'inside_elevator_door_ground_line',
                              'outside_elevator_door_concave_wall_line', 'inside_elevator_door_concave_line',
                              'inside_elevator_door_convex_line']

        # self.vis_cls_names = ['wall_ground_line', 'cabinet_ground_line', 'sofa_ground_line']
        # self.vis_cls_names = ['door_inside_out_line']
        self.save_vis = False
        self.show_vis = False
        self.img_pred = None
        self.img_gt = None
        self.save_root = ""

    def reset(self):
        self.kpi_names = ["over-all"] + list(self.class_names.keys())
        self.kip_infos = self.init_kpi_infos()
        self.save_vis = False
        self.show_vis = True
        self.img_pred = None
        self.img_gt = None

    def show_kpi_statistics(self):
        grid_num = 12
        string_len = 120

        print("-" * string_len)
        print("**********line kpi calculator infos************")
        print("test img num:", self.count)
        print("-" * string_len)
        print("*cls_name".ljust(grid_num*3), "*acc".ljust(grid_num), "*recall".ljust(grid_num),
              "*F".ljust(grid_num), "*tp".ljust(grid_num), "*fp".ljust(grid_num),
              "*ann_num".ljust(grid_num))
        print("-" * string_len)

        tps = [self.kip_infos[kpi_name]['tp'] for kpi_name in self.kpi_names[1:]]
        fps = [self.kip_infos[kpi_name]['fp'] for kpi_name in self.kpi_names[1:]]
        ann_nums = [self.kip_infos[kpi_name]['ann_num'] for kpi_name in self.kpi_names[1:]]
        tps = sum(tps)
        fps = sum(fps)
        ann_nums = sum(ann_nums)
        accs = tps / (tps + fps + 1e-6)
        recalls = tps / (ann_nums + 1e-6)
        fs = 2 * accs * recalls / (accs + recalls + 1e-6)

        over_all_infos = {"tp": tps, "fp": fps, "ann_num": ann_nums, "acc": accs, "recall": recalls, "f": fs}
        self.kpi_names = ["over-all-with-cls"] + self.kpi_names
        self.kip_infos.update({"over-all-with-cls": over_all_infos})

        for kpi_name in self.kpi_names:
            kpi_info = self.kip_infos[kpi_name]
            kpi_info['acc'] = kpi_info['tp']/(kpi_info['tp'] + kpi_info['fp'] + 1e-6)
            kpi_info['recall'] = kpi_info['tp'] / (kpi_info['ann_num'] + 1e-6)
            kpi_info['f'] = 2 * kpi_info['acc'] * kpi_info['recall'] /\
                            (kpi_info['acc'] + kpi_info['recall'] + 1e-6)

            cls_name = kpi_name
            acc = "" + str(round(kpi_info['acc'], 3))
            recall = "" + str(round(kpi_info['recall'], 3))
            f = "" + str(round(kpi_info['f'], 3))
            tp = "" + str(kpi_info['tp'])
            fp = "" + str(kpi_info['fp'])
            ann_num = "" + str(kpi_info['ann_num'])

            # print("-" * string_len)
            print(cls_name[:grid_num*3].ljust(grid_num*3), acc.ljust(grid_num), recall.ljust(grid_num),
                  f.ljust(grid_num), tp.ljust(grid_num), fp.ljust(grid_num),
                  ann_num.ljust(grid_num))
        print("-" * string_len)

    def get_name_2_chinese(self):
        name_2_chinese = {}
        for key, val in zip(self.class_names.keys(), self.class_names_chinese.keys()):
            name_2_chinese.update({key: val})
        return name_2_chinese

    def get_cls_name(self, value):
        return [k for k, v in self.class_names.items() if v == value]

    def init_kpi_infos(self):
        kip_infos = {}
        record_infos = {"tp": 0, "fp": 0, "ann_num": 0, "acc": 0, "recall": 0, "f": 0}
        for kpi_name in self.kpi_names:
            kip_infos.update({kpi_name: copy.deepcopy(record_infos)})

        return kip_infos

    def draw_chinese(self, img, text, position, textColor=(0, 255, 0), textSize=30):
        textColor = textColor[::-1]
        if (isinstance(img, np.ndarray)):
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        fontStyle = ImageFont.truetype(
            "/userdata/liyj/programs/mlsd_with_cls/local_files/simhei.ttf", textSize, encoding="utf-8")

        draw.text(position, text, textColor, font=fontStyle)
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        return img

    def update(self, img, gt_lines, pred_lines, img_name=None):
        img_h, img_w, _ = img.shape
        self.count += 1

        self.img_pred = copy.deepcopy(img)
        self.img_gt = copy.deepcopy(img)
        all_tp = 0
        all_fp = 0

        for kpi_name in self.kpi_names:
            if kpi_name != self.kpi_names[0]:
                cls_gt_lines = [gt_line for gt_line in gt_lines
                                if gt_line[4] == kpi_name]

                cls_pred_lines = [pred_line for pred_line in pred_lines
                                 if self.get_cls_name(int(pred_line[4]))[0] == kpi_name]

            else:
                cls_gt_lines = gt_lines
                cls_pred_lines = pred_lines

            tp_num = self.kip_infos[kpi_name]["tp"]
            fp_num = self.kip_infos[kpi_name]["fp"]
            ann_num = self.kip_infos[kpi_name]["ann_num"]

            if len(cls_gt_lines) > 0 and len(cls_pred_lines) > 0:
                iou_matrix = self.cal_iou_matrix(cls_gt_lines, cls_pred_lines, img_w, img_h)

                pred_bool = np.max(iou_matrix, axis=0)
                pred_bool = pred_bool > self.iou

                ann_num = ann_num + len(cls_gt_lines)
                tp_num = tp_num + np.sum(pred_bool)
                fp_num = fp_num + len(pred_bool) - np.sum(pred_bool)

            else:
                ann_num = ann_num + len(cls_gt_lines)
                tp_num = tp_num + 0
                fp_num = fp_num + len(cls_pred_lines)
                pred_bool = [False for _ in range(len(cls_pred_lines))]

            self.kip_infos[kpi_name]["tp"] = tp_num
            self.kip_infos[kpi_name]["fp"] = fp_num
            self.kip_infos[kpi_name]["ann_num"] = ann_num

            if kpi_name == self.kpi_names[0]:
                all_tp = tp_num
                all_fp = tp_num

            if (self.save_vis or self.show_vis) and kpi_name in self.vis_cls_names:
                self.vis_pred_result(cls_pred_lines, pred_bool)

        if self.save_vis or self.show_vis:
            self.vis_gt_result(gt_lines)
            img_blank = np.ones((self.img_pred.shape[0], 30, 3), dtype=np.uint8) * 255
            img_show = cv2.hconcat([img, img_blank, self.img_gt, img_blank, self.img_pred])

            if self.save_vis:
                if img_name is None:
                    img_name = str(self.count) + '.jpg'
                s_img_name = img_name.split(".")[0] + "_tp" + str(all_tp) + "fp" + str(all_fp) + ".jpg"
                cv2.imwrite(os.path.join(self.save_root, s_img_name), img_show)
                # cv2.imwrite(os.path.join(self.save_root, s_img_name), self.img_pred)

            if self.show_vis:
                # plt.imshow(img_show[:, :, ::-1])
                # plt.imshow(self.img_gt[:, :, ::-1])
                plt.imshow(self.img_pred[:, :, ::-1])
                plt.show()
                # exit(1)

    def cal_iou_matrix(self, gt_lines, pred_lines, map_width, map_height):
        gt_num = len(gt_lines)
        pred_num = len(pred_lines)
        iou_matrix = np.zeros((gt_num, pred_num), dtype=np.float32)
        for i in range(gt_num):
            gt_line = gt_lines[i]
            gt_x0, gt_y0, gt_x1, gt_y1, gt_cls = gt_line
            gt_x0 = int(round(gt_x0))
            gt_y0 = int(round(gt_y0))
            gt_x1 = int(round(gt_x1))
            gt_y1 = int(round(gt_y1))

            gt_line_map = np.zeros((map_height, map_width), np.uint8)
            cv2.line(gt_line_map, (gt_x0, gt_y0), (gt_x1, gt_y1), (1, 1, 1), self.line_thickness, 8)
            gt_line_map = np.array(gt_line_map, np.float32)

            for j in range(pred_num):
                pred_line = pred_lines[j]
                pred_line_map = np.zeros((map_height, map_width), np.uint8)

                pred_x0 = int(pred_line[0])
                pred_y0 = int(pred_line[1])
                pred_x1 = int(pred_line[2])
                pred_y1 = int(pred_line[3])

                cv2.line(pred_line_map, (pred_x0, pred_y0), (pred_x1, pred_y1), (1, 1, 1), self.line_thickness, 8)

                pred_line_map = np.array(pred_line_map, np.float32)
                intersection = gt_line_map * pred_line_map
                union = gt_line_map + pred_line_map
                union[union > 0] = 1

                eps = 1e-6
                iou = np.sum(intersection)/(np.sum(union) + eps)
                iou_matrix[i, j] = iou

        return iou_matrix

    def vis_pred_result(self, pred_lines, pred_bool):
        self.img_pred = self.draw_chinese(self.img_pred, "ModelPred", (30, 30), (0, 255, 0), 40)
        for l, pred_right in zip(pred_lines, pred_bool):
            if pred_right:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)

            minx, miny = min(int(l[0]), int(l[2])), min(int(l[1]), int(l[3]))
            w, h = int(abs(l[0] - l[2]) / 2), int(abs(l[1] - l[3]) / 2)
            xc, yc = minx + w, miny + h

            conf = str(round(l[5], 5))
            txt = self.name_2_chinese[self.get_cls_name(l[4])[0]] + ":" + conf

            cv2.circle(self.img_pred, (int(xc), int(yc)), 5, color, 5)
            cv2.line(self.img_pred, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), color, 2, 16)
            self.img_pred = self.draw_chinese(self.img_pred, txt, (xc, yc), color, 20)

    def vis_gt_result(self, gt_lines):
        self.img_gt = self.draw_chinese(self.img_gt, "GroundTruth", (30, 30), (0, 255, 0), 40)

        for gt_line in gt_lines:
            x0, y0, x1, y1, cls = gt_line
            cls_name = cls  # str(self.class_names(self.class_names_chinese, cls))
            if cls_name in self.vis_cls_names:
                x0 = int(round(x0))
                y0 = int(round(y0))
                x1 = int(round(x1))
                y1 = int(round(y1))
                xc = int(x0 * 0.5 + x1 * 0.5)
                yc = int(y0 * 0.5 + y1 * 0.5)

                cv2.circle(self.img_gt, (xc, yc), 5, (0, 255, 0), 5)
                cv2.line(self.img_gt, (x0, y0), (x1, y1), (0, 255, 0), 2, 16)
                # txt = str(self.get_cls_name(self.class_names_chinese, cls))
                txt = self.name_2_chinese[cls]

                self.img_gt = self.draw_chinese(self.img_gt, txt, (xc, yc), (0, 255, 0), 20)


def _line_len_fn(l1):
    len1 = np.sqrt((l1[2] - l1[0]) ** 2 + (l1[3] - l1[1]) ** 2)
    return len1

def parse_line_json(json_path):
    with open(json_path, 'r') as fp:
        annos = json.load(fp)

    lines = []
    curves = []
    for s in annos['shapes']:
        cls = s["label"]
        pt = s["points"]
        if len(pt) == 2:
            line = [pt[0][0], pt[0][1], pt[1][0], pt[1][1], cls]
            curve = None

        else:
            line = None
            length = len(pt)
            curve = []
            for i in range(length):
                curve.append([pt[i][0], pt[i][1]])
            curve.append(length)
            curve.append(cls)

        if line is not None:
            min_len = 60
            if _line_len_fn(line) > min_len:
                lines.append(line)

        if curve is not None:
            curves.append(curve)
    return lines, curves


def cal_kpi_line4():
    # root = "/userdata/liyj/data/train_data/edge_indoor/lines_0704/test_kpi"
    root = "/userdata/liyj/data/test_data/line/test_px2_1012/test"
    # root = "/home/dev/data_disk/liyj/data/collect_data/edge_line/edge_lines/0505/overfit_self/train"
    img_root = os.path.join(root, "images")
    json_root = os.path.join(root, "jsons")
    use_mlsd = True

    dst_root = os.path.join(root, "debug_0712")
    if os.path.exists(dst_root):
        shutil.rmtree(dst_root)
    os.mkdir(dst_root)

    # model
    # mlsd model
    if use_mlsd:
        # model_path = "/home/dev/data_disk/liyj/programs/mlsd_with_cls/workdir/models/mlsd_indoor_0419_label_self/latest.pth"
        # best by 0422
        # model_path = "/home/dev/data_disk/liyj/programs/mlsd_with_cls/workdir/models/mlsd_indoor_0420_label_self_displace_1.0_no_sol_no_center/latest.pth"

        # best
        # model_path = "/home/dev/data_disk/liyj/programs/mlsd_with_cls/workdir/models/mlsd_indoor_0422_label_self_displace_1.0_no_sol_no_center_new_displace_auto/latest.pth"

        # model_path = "/home/dev/data_disk/liyj/programs/mlsd_with_cls/workdir/models/mlsd_indoor_0424_label_self_displace_1.0_no_sol_no_center_new_displace_auto_2w/latest.pth"

        # model_path = "/home/dev/data_disk/liyj/programs/mlsd_with_cls/workdir/models/test_change_size_640/latest.pth"
        # model = MobileV2_MLSD_Large().cuda().eval()
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # model.load_state_dict(torch.load(model_path, map_location=device), strict=True)

        # model_path = "/home/dev/data_disk/liyj/programs/mlsd_with_cls/workdir/models/test_change_size_640_convnet/latest.pth"

        # model_path = "/home/dev/data_disk/liyj/programs/mlsd_with_cls/workdir/models/test_change_size_640_convnet_2w/latest.pth"
        # model_path = "/home/dev/data_disk/liyj/programs/mlsd_with_cls/workdir/models/test_change_size_640/latest.pth"
        # model_path = "/home/dev/data_disk/liyj/programs/mlsd_with_cls/workdir/models/test_change_size_640_rep_2w_0429/latest.pth"

        # model_path = "/home/dev/data_disk/liyj/programs/mlsd_with_cls/workdir/models/mlsd_indoor_0507_debug/latest.pth"
        # model_path = "/userdata/liyj/programs/mlsd_with_cls/train_models/mlsd_indoor_kpi_3w_0706/latest.pth"
        # model_path = "/userdata/liyj/programs/mlsd_with_cls/train_models/mlsd_indoor_kpi_2w_0706/latest.pth"
        # model_path = "/userdata/liyj/programs/mlsd_with_cls/train_models/mlsd_indoor_kpi_1w_0706/latest.pth"
        # model_path = "/userdata/liyj/programs/mlsd_with_cls/train_models/mlsd_indoor_px2_0923/line_px2_latest.pth"

        # tmp
        # model_path = "/userdata/liyj/data/tmp/indoor_line_px2_1011_2.pth"
        model_path = "/userdata/liyj/data/tmp/indoor_line_px1_1011.pth"
        model = LineDetector(model_path)

    else:
        # lld model
        model_path = "../workdir/models/0420_indoor_curve_lines_self_label/latest.pth"
        cfg_path = "../mlsd_pytorch/configs/mobilev2_mlsd_curve_large_512_base_bsize24.yaml"
        cfg = get_cfg(cfg_path)
        classes = curve_class_dict
        model = Curve_Detector(cfg, model_path, classes)
        model.line_kpi = True

    # kpi calculator
    line_kpi_calculator = LineKpiCalculator()

    line_kpi_calculator.show_vis = 0
    line_kpi_calculator.save_vis = 0
    line_kpi_calculator.save_root = dst_root

    img_names = [name for name in os.listdir(img_root) if name.endswith('.png')]
    for img_name in tqdm(img_names):
        # img_name = "a7ac1f11-a983-11ec-9d25-7c10c921acb3.png"
        # img_name = "6ea16475-a9ae-11ec-9d25-7c10c921acb3.png"
        # img_name = "b2529808-6982-11ec-95d3-000c293913c8.png"

        # displacement
        # img_name = "5a7287b7-6981-11ec-95d3-000c293913c8.png"
        # img_name = "2bcc2318-6986-11ec-95d3-000c293913c8.png"
        # img_name = "b406c77f-9f16-11ec-9d24-7c10c921acb3.png"
        # img_name = "0b63c31e-c0e7-11ec-acc7-72d8ab8687f9.png"
        # img_name = "0d6c07b4-a92f-11ec-9d25-7c10c921acb3.png"
        # img_name = "0d6c07e5-a92f-11ec-9d25-7c10c921acb3.png"
        # img_name = "0d6c07f9-a92f-11ec-9d25-7c10c921acb3.png"
        img_path = os.path.join(img_root, img_name)
        json_path = os.path.join(json_root, img_name.replace('.png', '.json'))

        img = cv2.imread(img_path)

        gt_lines, gt_curves = parse_line_json(json_path)

        if use_mlsd:
            pred_lines = model.infer(img)

            # model_input_shape = [480, 288]
            # model_input_shape = [640, 352]
            # # model_input_shape = [512, 512]
            # img_proc = copy.deepcopy(img)
            # img_proc = cv2.cvtColor(img_proc, cv2.COLOR_BGR2RGB)
            # pred_lines, centers, centermap, clsmap, linemap, \
            # curvemap, seedMark_list, points_list, \
            # curvelines_list = pred_lines_and_cls(img_proc, model, None, None, model_input_shape, 0.3, 5, cls_num=12)
        else:
            img_proc = copy.deepcopy(img)
            pred_lines = model.infer(img_proc)

        line_kpi_calculator.update(img, gt_lines, pred_lines, img_name=img_name)
        # exit(1)

    line_kpi_calculator.show_kpi_statistics()


if __name__ == "__main__":
    print("Start Proc")
    cal_kpi_line4()
    print("End Proc")
