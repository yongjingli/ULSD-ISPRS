import os
import cv2
import torch
import json
import random
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFont, ImageDraw

from .dataset_utils import swap_line_pt_maybe, get_ext_lines, gen_TP_mask3, gen_SOL_map2,\
    gen_junction_and_line_mask, gen_curve_mask, cut_line_by_xmin, cut_line_by_xmax


from albumentations import (
    RandomBrightnessContrast,
    OneOf,
    HueSaturationValue,
    Compose,
    Normalize,
    Blur,
    GaussianBlur,
    MedianBlur

)


def parse_labelme_json_file(img_dir, label_file):
    infos = []
    label_list = os.listdir(label_file)
    assert(len(label_list) != 0)

    for i in label_list:
        label_full_path = os.path.join(label_file, i)
        contents = json.load(open(label_full_path, 'r'))
        infos.append(contents)

    random.shuffle(infos)
    return infos


class MLSD_LINE_Dataset(Dataset):
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

    down_scale = 2

    def __init__(self, cfg, is_train):
        super(MLSD_LINE_Dataset, self).__init__()

        self.cfg = cfg
        self.min_len = cfg.decode.len_thresh
        self.is_train = is_train

        self.img_dir = cfg.train.img_dir
        self.label_fn = cfg.train.label_fn

        if not is_train:
            self.img_dir = cfg.val.img_dir
            self.label_fn = cfg.val.label_fn

        self.cache_dir = cfg.train.data_cache_dir
        self.with_cache = cfg.train.with_cache

        print("==> load label..")

        self.anns = self._load_anns(self.img_dir, self.label_fn)

        print("==> valid samples: ", len(self.anns))

        self.input_w = cfg.datasets.input_w
        self.input_h = cfg.datasets.input_h

        self.train_aug = self._aug_train()
        self.test_aug = self._aug_test(input_size=None)

        self.cache_to_mem = cfg.train.cache_to_mem
        self.cache_dict = {}

        assert self.down_scale == cfg.model.down_scale, "self.down_scale == cfg.model.down_scale"

    def __len__(self):
        return len(self.anns)

    def _load_anns(self, img_dir, label_fn):
        infos = parse_labelme_json_file(img_dir, label_fn)
        anns = []
        for c in infos:
            img_full_jpg = os.path.join(img_dir, c['imagePath'][:-4] + ".jpg")
            img_full_png = os.path.join(img_dir, c['imagePath'][:-4] + ".png")
            img_full_fn = ""

            if os.path.exists(img_full_jpg):
                img_full_fn = img_full_jpg
            elif os.path.exists(img_full_png):
                img_full_fn = img_full_png

            if not os.path.exists(img_full_fn):
                print(" not exist!".format(img_full_fn))
                exit(0)

            lines = []
            curves = []
            for s in c['shapes']:
                cls = s["label"]

                pt = s["points"]
                if len(pt) == 2:
                    line = [pt[0][0], pt[0][1], pt[1][0], pt[1][1], cls]
                    line = swap_line_pt_maybe(line)

                    # line length filter
                    if self._line_len_fn(line) < self.min_len:
                        line = None
                    curve = None
                else:
                    # mid = len(pt) // 2
                    # line = [pt[mid][0], pt[mid][1], pt[mid+1][0], pt[mid+1][1], cls]
                    line = [pt[0][0], pt[0][1], pt[1][0], pt[1][1], cls]
                    line = swap_line_pt_maybe(line)
                    length = len(pt)
                    curve = []
                    for i in range(length):
                        curve.append([pt[i][0], pt[i][1]])
                    curve.append(length)
                    curve.append(cls)

                    # line length filter
                    if self._line_len_fn(line) < self.min_len:
                        line = None

                if line is not None:
                    lines.append(line)
                if curve is not None:
                    curves.append(curve)

            dst_ann = {
                'img_full_fn': img_full_fn,
                'lines': lines,
                'curves': curves,
                'img_w': c['imageWidth'],
                'img_h': c['imageHeight']
            }
            anns.append(dst_ann)
        return anns

    def _line_len_fn(self, l1):
        len1 = np.sqrt((l1[2] - l1[0]) ** 2 + (l1[3] - l1[1]) ** 2)
        return len1

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
                        MedianBlur(blur_limit=3, p=0.5)
                    ]
                ),

            ],
            p=1.0)
        return aug

    def _aug_test(self, input_size=384):
        aug = Compose(
            [
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ],
            p=1.0)
        return aug

    def load_label(self, ann, do_aug):
        norm_lines = []
        for l in ann['lines']:
            ll = [
                np.clip(l[0] / ann['img_w'], 0, 1),
                np.clip(l[1] / ann['img_h'], 0, 1),
                np.clip(l[2] / ann['img_w'], 0, 1),
                np.clip(l[3] / ann['img_h'], 0, 1),
                l[4]
            ]

            x0, y0, x1, y1 = self.input_w * ll[0], self.input_h * ll[1], self.input_w * ll[2], self.input_h * ll[3]
            if x0 == x1 and y0 == y1:
                print('fatal err!')
                print(ann['img_w'], ann['img_h'])
                print(ll)
                print(l)
                print(ann)
                exit(0)

            norm_lines.append(ll)

        ann['norm_lines'] = norm_lines

        norm_curves = []
        for c in ann['curves']:
            points_num = c[-2]
            cls = c[-1]
            cc = []
            for i in range(points_num):
                x, y = c[i]
                x = np.clip(x / ann['img_w'], 0, 1)
                y = np.clip(y / ann['img_h'], 0, 1)
                cc.append([x, y])
            cc.append(points_num)
            cc.append(cls)

            norm_curves.append(cc)

        ann['norm_curves'] = norm_curves

        label_cache_path = os.path.basename(ann['img_full_fn'])[:-4] + '.npy'
        label_cache_path = self.cache_dir + '/' + label_cache_path

        can_load = self.with_cache and not do_aug

        if can_load and self.cache_to_mem and label_cache_path in self.cache_dict.keys():
            label = self.cache_dict[label_cache_path]

        elif can_load and os.path.exists(label_cache_path):
            label = np.load(label_cache_path)
            if self.cache_to_mem:
                self.cache_dict[label_cache_path] = label
        else:
            tp_mask = gen_TP_mask3(ann['norm_lines'], self.input_h // self.down_scale, self.input_w // self.down_scale, input_cls_dict=self.CLASSES,
                                   with_ext=self.cfg.datasets.with_centermap_extend)

            sol_mask, ext_lines = gen_SOL_map2(ann['norm_lines'], self.input_h // self.down_scale, self.input_w // self.down_scale,
                                      with_ext=False, input_cls_dict=self.CLASSES)

            junction_map, line_map = gen_junction_and_line_mask(ann['norm_lines'],
                                                                self.input_h // self.down_scale, self.input_w // self.down_scale)

            curve_map = gen_curve_mask(ann['norm_curves'],
                                        self.input_h // self.down_scale, self.input_w // self.down_scale)

            cls_num = len(self.CLASSES.keys())
            sol_tp_num = 1 + 4 + 2 + cls_num

            label = np.zeros((2 * sol_tp_num + 3, self.input_h // self.down_scale, self.input_w // self.down_scale), dtype=np.float32)
            label[0:sol_tp_num, :, :] = sol_mask
            label[sol_tp_num:sol_tp_num * 2, :, :] = tp_mask
            label[sol_tp_num * 2, :, :] = junction_map
            label[sol_tp_num * 2 + 1, :, :] = line_map
            label[sol_tp_num * 2 + 2, :, :] = curve_map

        return label

    def _geo_aug(self, img, ann_origin):
        do_aug = False

        lines = ann_origin['lines'].copy()
        if random.random() < 0.5:
            do_aug = True
            flipped_lines = []
            img = np.fliplr(img) # 图片随机左右翻转
            for l in lines:
                flipped_lines.append(
                    swap_line_pt_maybe([ann_origin['img_w'] - l[0],
                                        l[1], ann_origin['img_w'] - l[2], l[3], l[4]]))
            ann_origin['lines'] = flipped_lines

        if random.random() < 0.5:
            do_aug = True
            img, ann_origin = self._crop_aug(img, ann_origin) # 随机剪裁图片

        ann_origin['img_w'] = img.shape[1]
        ann_origin['img_h'] = img.shape[0]

        return do_aug, img, ann_origin

    def _crop_aug(self, img, ann_origin):
        assert img.shape[1] == ann_origin['img_w']
        assert img.shape[0] == ann_origin['img_h']
        img_w = ann_origin['img_w']
        img_h = ann_origin['img_h']
        lines = ann_origin['lines']
        xmin = random.randint(1, int(0.1 * img_w))

        ## xmin
        xmin_lines = []
        for line in lines:
            flg, line = cut_line_by_xmin(line, xmin)
            line[0] -= xmin
            line[2] -= xmin
            if flg and self._line_len_fn(line) > self.min_len:
                xmin_lines.append(line)
        lines = xmin_lines

        img = img[:, xmin:, :]
        ## xmax
        xmax = img.shape[1] - random.randint(1, int(0.1 * img.shape[1]))
        img = img[:, :xmax, :].copy()
        xmax_lines = []
        for line in lines:
            flg, line = cut_line_by_xmax(line, xmax)
            if flg and self._line_len_fn(line) > self.min_len:
                xmax_lines.append(line)
        lines = xmax_lines

        ann_origin['lines'] = lines
        ann_origin['img_w'] = img.shape[1]
        ann_origin['img_h'] = img.shape[0]

        return img, ann_origin

    def __getitem__(self, index):
        ann = self.anns[index].copy()
        img = cv2.imread(ann['img_full_fn'])

        do_aug = False
        if self.is_train and random.random() < 0.5:
            do_aug, img, ann = self._geo_aug(img, ann)  # flip. crop

        img = cv2.resize(img, (self.input_w, self.input_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label = self.load_label(ann, do_aug)
        ext_lines = get_ext_lines(ann['norm_lines'], self.input_h // self.down_scale, self.input_w // self.down_scale)

        norm_lines = ann['norm_lines']
        norm_lines_512_list = []
        for l in norm_lines:
            norm_lines_512_list.append([
                l[0] * self.input_w,
                l[1] * self.input_h,
                l[2] * self.input_w,
                l[3] * self.input_h,
            ])

        ext_lines_512_list = []
        for l in ext_lines:
            ext_lines_512_list.append([
                l[0] * self.input_w,
                l[1] * self.input_h,
                l[2] * self.input_w,
                l[3] * self.input_h,
            ])

        if self.is_train:
            img = self.train_aug(image=img)['image']
        # img_norm = (img / 127.5) - 1.0
        img_norm = self.test_aug(image=img)['image']

        norm_lines_512_tensor = torch.from_numpy(np.array(norm_lines_512_list, np.float32))
        sol_lines_512_tensor = torch.from_numpy(np.array(ext_lines_512_list, np.float32))

        return img_norm, img, label, \
               norm_lines_512_list, \
               norm_lines_512_tensor, \
               sol_lines_512_tensor, \
               ann['img_full_fn']


def MLSD_LINE_Dataset_collate_fn(batch):
    batch_size = len(batch)
    h, w, c = batch[0][0].shape
    images = np.zeros((batch_size, 3, h, w), dtype=np.float32)

    gt_channels = (len(MLSD_LINE_Dataset.CLASSES.keys()) + 7) * 2 + 3

    down_scale = MLSD_LINE_Dataset.down_scale
    labels = np.zeros((batch_size, gt_channels, h // down_scale, w // down_scale), dtype=np.float32)
    #cls_labels = []
    img_fns = []
    img_origin_list = []
    norm_lines_512_all = []
    norm_lines_512_all_tensor_list = []
    sol_lines_512_all_tensor_list = []

    for inx in range(batch_size):
        im, img_origin, label_mask,\
        norm_lines_512, norm_lines_512_tensor, \
        sol_lines_512, img_fn = batch[inx]

        images[inx] = im.transpose((2, 0, 1))
        labels[inx] = label_mask
        #cls_labels.append(class_label)
        img_origin_list.append(img_origin)
        img_fns.append(img_fn)
        norm_lines_512_all.append(norm_lines_512)
        norm_lines_512_all_tensor_list.append(norm_lines_512_tensor)
        sol_lines_512_all_tensor_list.append(sol_lines_512)

    images = torch.from_numpy(images)
    labels = torch.from_numpy(labels)
    #cls_labels = torch.from_numpy(cls_labels)

    return {
        "xs": images,
        "ys": labels,
        "img_fns": img_fns,
        "origin_imgs": img_origin_list,
        "gt_lines_512": norm_lines_512_all,
        "gt_lines_tensor_512_list": norm_lines_512_all_tensor_list,
        "sol_lines_512_all_tensor_list": sol_lines_512_all_tensor_list
    }


if __name__ == '__main__':
    # root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    root_dir = "../"
    import sys

    sys.path.append(root_dir)
    from configs.mlsd_default import get_cfg_defaults

    cfg = get_cfg_defaults()

    # cfg.train.img_dir = root_dir + "/data/12_13_100_myself_1/train/images"
    # cfg.train.label_fn = root_dir + "/data/12_13_100_myself_1/train/jsons"

    cfg.train.img_dir = "/mnt/data10/liyj/programs/ULSD-ISPRS/dataset/mlsd_20230327/train/images"
    cfg.train.label_fn = "/mnt/data10/liyj/programs/ULSD-ISPRS/dataset/mlsd_20230327/train/jsons"
    print(cfg.train.img_dir)
    print(cfg.train.label_fn)
    cfg.train.batch_size = 1
    cfg.train.data_cache_dir = ""
    cfg.train.with_cache = False
    cfg.datasets.with_centermap_extend = False

    dset = MLSD_LINE_Dataset(cfg, True)

    color_list = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (10, 215, 255), (0, 255, 255),
                  (230, 216, 173), (128, 0, 128), (203, 192, 255), (238, 130, 238), (130, 0, 75),
                  (169, 169, 169), (0, 69, 255)]  # [纯红、纯绿、纯蓝、金色、纯黄、天蓝、紫色、粉色、紫罗兰、藏青色、深灰色、橙红色]

