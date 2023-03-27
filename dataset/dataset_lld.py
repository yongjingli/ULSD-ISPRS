import json

from torch._C import INSERT_FOLD_PREPACK_OPS
#from mlsd_pytorch.utils.comm import create_dir
#from utils.comm import create_dir
import  tqdm
import os
import cv2
import torch
import json
import random
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFont, ImageDraw
import math


from albumentations import (
    RandomBrightnessContrast,
    OneOf,
    HueSaturationValue,
    Compose,
    Normalize,
    Blur,
    GaussianBlur,
    MedianBlur,
    MotionBlur
)


class LLD_Curve_Dataset(Dataset):
    # CLASSES = {
    #     'road_plant_line': 0,
    #     'road_shoulder_line': 1,
    #     'road_parterre_line': 2,
    #     'road_building_line': 3,
    #     'road_fence_line': 4,
    #     'road_stair_line': 5,
    #     'road_sewer_line': 6,
    #     'road_other_canspan_line': 7,
    #     'road_other_cannotspan_line': 8,
    #     'off_the_shoulder_line': 9,
    #     'stair_line': 10,
    #     'uncertain_line': 11
    # }
    #
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

    def __init__(self, cfg, is_train):
        super(LLD_Curve_Dataset, self).__init__()

        # data dir setting
        self.cfg = cfg
        self.min_len = cfg.decode.len_thresh
        self.is_train = is_train

        self.img_dir = cfg.train.img_dir
        self.label_fn = cfg.train.label_fn

        if not is_train:
            self.img_dir = cfg.val.img_dir
            self.label_fn = cfg.val.label_fn

        # data basic info setting
        self.src_height = cfg.datasets.src_height
        self.src_width = cfg.datasets.src_width

        self.dst_height = cfg.datasets.input_h
        self.dst_width = cfg.datasets.input_w

        self.grid_size = cfg.train.grid_size
        # self.scale_x = float(self.dst_width)/self.src_width
        # self.scale_y = float(self.dst_height) / self.src_height
        #
        self.scale_x = cfg.train.scale_x
        self.scale_y = cfg.train.scale_y

        self.pad_left = cfg.train.pad_left
        self.pad_right = cfg.train.pad_right
        self.pad_top = cfg.train.pad_top
        self.pad_bottom = cfg.train.pad_bottom

        self.map_size = (self.dst_height, self.dst_width)
        self.lane_map_range = cfg.train.lane_map_range

        # annos info
        self.annos = None
        self.train_aug = self._aug_train()
        self.test_aug = self._aug_test()

        self.class_dic = LLD_Curve_Dataset.CLASSES
        self.cls_num = len(self.class_dic.keys())

        print("==> load label..")
        self.anns = self.load_curse_anns(self.img_dir, self.label_fn)
        print("==> valid samples: ", len(self.anns))

    def __len__(self):
        return len(self.anns)

    @staticmethod
    def get_cls_num(self):
        return self.cls_num

    def load_curse_anns(self, img_dir, label_fn):
        anns = []
        img_names = [name for name in os.listdir(img_dir)
                     if name.endswith(".png") or name.endswith(".jpg")]
        for img_name in img_names:
            # img_name = "0b320388-79d5-4f2d-9b86-a9c9bb1f1a87.jpg"
            # check data path
            img_path = os.path.join(img_dir, img_name)
            json_path = os.path.join(label_fn,
                                     img_name.replace('.png', '.json').replace('.jpg', '.json'))
            if not os.path.exists(img_path):
                print(" not exist!".format(img_path))
                exit(0)
            if not os.path.exists(json_path):
                print(" not exist!".format(json_path))
                exit(0)

            # gen gt maps
            # remap_img, remap_multi_gt = self.parse_gt_line_maps(img_path, json_path)
            img = cv2.imread(img_path)
            img_h, img_w, _ = img.shape
            assert img_h == self.src_height, "img_h not equel to 1080"
            assert img_w == self.src_width, "img_w not equel to 1920"

            remap_annos = self.parse_json(json_path)

            dst_ann = {
                'img_full_fn': img_path,
                'remap_annos': remap_annos,
            }

            anns.append(dst_ann)
        return anns

    def parse_json(self, json_path):
        with open(json_path, 'r') as f:
            annos = json.load(f)

        remap_annos = []
        for i, shape in enumerate(annos['shapes']):
            line_id = int(shape['id'])
            label = shape['label']

            # need_cls_names = ['concave_wall_line', 'convex_wall_line']
            # if label not in need_cls_names and self.is_train:
            #     continue

            points = shape['points']
            points_remap = []

            for point in points:
                x = point[0] * self.scale_x + self.pad_left
                y = point[1] * self.scale_y + self.pad_top
                points_remap.append([x, y])
            # remap_annos.append({'label': label, 'points': points_remap, 'index': index, 'same_line_id': index})
            remap_annos.append({'label': label, 'points': points_remap, 'index': i, 'line_id': line_id})
        return remap_annos

    def gen_line_map(self, remap_annos):
        line_map = np.zeros(self.map_size, dtype=np.uint8)
        line_map_id = np.zeros(self.map_size, dtype=np.uint8)
        line_map_cls = np.zeros(self.map_size, dtype=np.uint8)
        for remap_anno in remap_annos:
            label = remap_anno['label']
            line_points = remap_anno['points']
            index = remap_anno['index'] + 1
            line_id = remap_anno['line_id'] + 1

            pre_point = line_points[0]
            for cur_point in line_points[1:]:
                x1, y1 = round(pre_point[0]), round(pre_point[1])
                x2, y2 = round(cur_point[0]), round(cur_point[1])
                cv2.line(line_map, (x1, y1), (x2, y2), (index,))
                cv2.line(line_map_id, (x1, y1), (x2, y2), (line_id,))

                cls_value = self.class_dic[label] + 1
                cv2.line(line_map_cls, (x1, y1), (x2, y2), (cls_value,))
                pre_point = cur_point

        return line_map, line_map_id, line_map_cls

    def gen_gt_line_maps(self, line_map, line_map_id, line_map_cls):
        line_map_h, line_map_w = line_map.shape
        gt_map_h, gt_map_w = math.ceil(line_map_h / self.grid_size), math.ceil(
            line_map_w / self.grid_size
        )
        gt_confidence = np.zeros((1, gt_map_h, gt_map_w), dtype=np.float)
        gt_offset_x = np.zeros((1, gt_map_h, gt_map_w), dtype=np.float)
        gt_offset_y = np.zeros((1, gt_map_h, gt_map_w), dtype=np.float)
        gt_line_index = np.zeros((1, gt_map_h, gt_map_w), dtype=np.float)
        gt_line_id = np.zeros((1, gt_map_h, gt_map_w), dtype=np.float)

        gt_line_cls = np.zeros((self.cls_num, gt_map_h, gt_map_w), dtype=np.float)

        for y in range(0, gt_map_h):
            for x in range(0, gt_map_w):
                start_x, end_x = x * self.grid_size, (x + 1) * self.grid_size
                end_x = end_x if end_x < line_map_w else line_map_w
                start_y, end_y = y * self.grid_size, (y + 1) * self.grid_size
                end_y = end_y if end_y < line_map_h else line_map_h
                grid = line_map[start_y:end_y, start_x:end_x]

                grid_id = line_map_id[start_y:end_y, start_x:end_x]
                grid_cls = line_map_cls[start_y:end_y, start_x:end_x]

                confidence = 1 if np.any(grid) else 0
                gt_confidence[0, y, x] = confidence
                if confidence == 1:
                    ys, xs = np.nonzero(grid)
                    offset_y, offset_x = sorted(
                        zip(ys, xs), key=lambda p: (p[0], -p[1]), reverse=True
                    )[0]
                    gt_offset_x[0, y, x] = offset_x / (self.grid_size - 1)
                    gt_offset_y[0, y, x] = offset_y / (self.grid_size - 1)
                    gt_line_index[0, y, x] = grid[offset_y, offset_x]
                    gt_line_id[0, y, x] = grid_id[offset_y, offset_x]

                    cls = grid_cls[offset_y, offset_x]
                    if cls > 0:
                        cls_indx = int(cls - 1)
                        gt_line_cls[cls_indx, y, x] = 1

        foreground_mask = gt_confidence.astype(np.uint8)

        # expand foreground mask
        kernel = np.ones((3, 3), np.uint8)
        foreground_expand_mask = cv2.dilate(foreground_mask[0], kernel)
        foreground_expand_mask = np.expand_dims(foreground_expand_mask.astype(np.uint8), axis=0)

        ignore_mask = np.zeros((1, gt_map_h, gt_map_w), dtype=np.uint8)
        top, bottom = self.lane_map_range
        ignore_mask[0, top:bottom, :] = 1

        return gt_confidence, gt_offset_x, gt_offset_y, \
               gt_line_index, ignore_mask, foreground_mask, \
               gt_line_id, gt_line_cls, foreground_expand_mask

    def __getitem__(self, index):
        # index = 2
        ann = self.anns[index].copy()

        # img process
        img = cv2.imread(ann['img_full_fn'])

        res_height = int(self.src_height * self.scale_y)
        res_width = int(self.src_width * self.scale_x)
        img = cv2.resize(img, (res_width, res_height))

        img = cv2.copyMakeBorder(img, self.pad_top, self.pad_bottom, self.pad_left, self.pad_left, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # img aug
        if self.is_train:
            img = self.train_aug(image=img)['image']

        # label process
        remap_annos = ann['remap_annos']

        # geo aug
        img, remap_annos = self.geo_aug(img, remap_annos)

        # gen train gt
        img_norm = self.test_aug(image=img)['image']

        line_map, line_map_id, line_map_cls = self.gen_line_map(remap_annos)
        gt_line_maps = self.gen_gt_line_maps(line_map, line_map_id, line_map_cls)

        gt_confidence = gt_line_maps[0]
        gt_offset_x = gt_line_maps[1]
        gt_offset_y = gt_line_maps[2]
        gt_line_index = gt_line_maps[3]
        ignore_mask = gt_line_maps[4]
        foreground_mask = gt_line_maps[5]

        gt_line_id = gt_line_maps[6]
        gt_line_cls = gt_line_maps[7]

        foreground_expand_mask = gt_line_maps[8]

        return img_norm, img,  \
               gt_confidence, \
               gt_offset_x, \
               gt_offset_y, \
               gt_line_index, \
               ignore_mask , \
               foreground_mask, \
               gt_line_id, \
               gt_line_cls, \
               foreground_expand_mask, \
               ann['img_full_fn']

    def geo_aug(self, img, remap_annos):
        # flip
        new_remap_annos = []
        if random.random() < 0.5:
            img = np.fliplr(img)
            img_h, img_w, _ = img.shape
            for remap_anno in remap_annos:
                new_anno = {'label': remap_anno['label'], 'index': remap_anno['index'],
                            'line_id': remap_anno["line_id"]}
                points = []
                for point in remap_anno['points']:
                    point[0] = img_w - point[0]
                    points.append(point)
                new_anno['points'] = points
                new_remap_annos.append(new_anno)
        else:
            new_remap_annos = remap_annos
        return img, new_remap_annos

    def _aug_test(self):
        aug = Compose(
            [
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ],
            p=1.0)
        return aug

    def _aug_train(self):
        aug = Compose(
            [
                OneOf(
                    [
                        HueSaturationValue(hue_shift_limit=10,
                                           sat_shift_limit=10,
                                           val_shift_limit=10,
                                           p=0.5),
                        RandomBrightnessContrast(brightness_limit=0.2,
                                                 contrast_limit=0.2,
                                                 p=0.5)
                    ]
                ),
                OneOf(
                    [
                        Blur(blur_limit=3, p=0.5),
                        GaussianBlur(blur_limit=3, p=0.5),
                        MedianBlur(blur_limit=3, p=0.5),
                        MotionBlur(blur_limit=(10, 20), p=0.5)
                    ]
                ),

            ],
            p=1.0)
        return aug


def LLD_Curve_Dataset_collate_fn(batch):
    batch_size = len(batch)
    h, w, c = batch[0][0].shape
    images = np.zeros((batch_size, 3, h, w), dtype=np.float32)

    _, gt_h, gt_w = batch[0][2].shape
    cls_num = len(LLD_Curve_Dataset.CLASSES.keys())
    batch_gt_confidence = np.zeros((batch_size, 1, gt_h, gt_w), dtype=np.float32)
    batch_gt_offset_x = np.zeros((batch_size, 1, gt_h, gt_w), dtype=np.float32)
    batch_gt_offset_y = np.zeros((batch_size, 1, gt_h, gt_w), dtype=np.float32)
    batch_gt_line_index = np.zeros((batch_size, 1, gt_h, gt_w), dtype=np.float32)
    batch_ignore_mask = np.zeros((batch_size, 1, gt_h, gt_w), dtype=np.float32)
    batch_foreground_mask = np.zeros((batch_size, 1, gt_h, gt_w), dtype=np.float32)
    batch_gt_line_id = np.zeros((batch_size, 1, gt_h, gt_w), dtype=np.float32)
    batch_gt_line_cls = np.zeros((batch_size, cls_num, gt_h, gt_w), dtype=np.float32)
    batch_foreground_expand_mask = np.zeros((batch_size, 1, gt_h, gt_w), dtype=np.float32)

    for inx in range(batch_size):
        im, img_origin, \
        gt_confidence, \
        gt_offset_x, \
        gt_offset_y, \
        gt_line_index, \
        ignore_mask, \
        foreground_mask, \
        gt_line_id, \
        gt_line_cls, \
        foreground_expand_mask, \
        img_fn = batch[inx]

        images[inx] = im.transpose((2, 0, 1))

        batch_gt_confidence[inx] = gt_confidence
        batch_gt_offset_x[inx] = gt_offset_x
        batch_gt_offset_y[inx] = gt_offset_y
        batch_gt_line_index[inx] = gt_line_index
        batch_ignore_mask[inx] = ignore_mask
        batch_foreground_mask[inx] = foreground_mask

        batch_gt_line_id[inx] = gt_line_id
        batch_gt_line_cls[inx] = gt_line_cls

        batch_foreground_expand_mask[inx] = foreground_expand_mask

    images = torch.from_numpy(images)
    batch_gt_confidence = torch.from_numpy(batch_gt_confidence)
    batch_gt_offset_x = torch.from_numpy(batch_gt_offset_x)
    batch_gt_offset_y = torch.from_numpy(batch_gt_offset_y)
    batch_gt_line_index = torch.from_numpy(batch_gt_line_index)
    batch_ignore_mask = torch.from_numpy(batch_ignore_mask)
    batch_foreground_mask = torch.from_numpy(batch_foreground_mask)
    batch_gt_line_id = torch.from_numpy(batch_gt_line_id)
    batch_gt_line_cls = torch.from_numpy(batch_gt_line_cls)
    batch_foreground_expand_mask = torch.from_numpy(batch_foreground_expand_mask)

    return {
        "xs": images,
        "batch_gt_confidence": batch_gt_confidence,
        "batch_gt_offset_x": batch_gt_offset_x,
        "batch_gt_offset_y": batch_gt_offset_y,
        "batch_gt_line_index": batch_gt_line_index,
        "batch_ignore_mask": batch_ignore_mask,
        "batch_foreground_mask": batch_foreground_mask,
        "batch_gt_line_id": batch_gt_line_id,
        "batch_gt_line_cls": batch_gt_line_cls,
        "batch_foreground_expand_mask": batch_foreground_expand_mask,
    }


if __name__ == '__main__':
    # root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    root_dir = "../"
    import sys
    sys.path.append(root_dir)
    from configs.lld_default import get_cfg_defaults

    cfg = get_cfg_defaults()

    # cfg.train.img_dir = root_dir + "/data/12_13_100_myself_1/train/images"
    # cfg.train.label_fn = root_dir + "/data/12_13_100_myself_1/train/jsons"

    cfg.train.img_dir = "/mnt/data10/liyj/programs/ULSD-ISPRS/dataset/lld_20230316/train/images"
    cfg.train.label_fn = "/mnt/data10/liyj/programs/ULSD-ISPRS/dataset/lld_20230316/train/jsons"
    print(cfg.train.img_dir)
    print(cfg.train.label_fn)
    cfg.train.batch_size = 1
    cfg.train.data_cache_dir = ""
    cfg.train.with_cache = False
    cfg.datasets.with_centermap_extend = False

    dset = LLD_Curve_Dataset(cfg, True)
    
    color_list = [(0,0,255), (0,255,0), (255,0,0), (10,215,255), (0,255,255), 
                  (230,216,173), (128,0,128), (203,192,255), (238,130,238), (130,0,75),
                  (169,169,169), (0,69,255)] # [纯红、纯绿、纯蓝、金色、纯黄、天蓝、紫色、粉色、紫罗兰、藏青色、深灰色、橙红色]

