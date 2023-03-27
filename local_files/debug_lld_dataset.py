import os
import cv2
import json
import numpy as np
import math
import matplotlib.pyplot as plt
import  argparse
from tqdm import tqdm
import sys

from configs.lld_default import get_cfg_defaults
from torch.utils.data import Dataset, DataLoader
from dataset.dataset_lld import LLD_Curve_Dataset, LLD_Curve_Dataset_collate_fn


color_list = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (10, 215, 255), (0, 255, 255),
              (230, 216, 173), (128, 0, 128), (203, 192, 255), (238, 130, 238), (130, 0, 75),
              (169, 169, 169), (0, 69, 255)]  # [纯红、纯绿、纯蓝、金色、纯黄、天蓝、紫色、粉色、紫罗兰、藏青色、深灰色、橙红色]


def get_train_dataloader(cfg, is_train=True):
    dataset = LLD_Curve_Dataset(cfg, is_train=is_train)

    dataloader_loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.sys.num_workers,
        drop_last=True,
        collate_fn=LLD_Curve_Dataset_collate_fn
    )
    return dataloader_loader


def get_val_dataloader(cfg, is_train=False):
    dataset = LLD_Curve_Dataset(cfg, is_train=is_train)

    dataloader_loader = DataLoader(
        dataset,
        batch_size=cfg.val.batch_size,
        shuffle=False,
        num_workers=cfg.sys.num_workers,
        drop_last=False,
        collate_fn=LLD_Curve_Dataset_collate_fn
    )
    return dataloader_loader


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        default= os.path.dirname(os.path.abspath(__file__))+ '/configs/mobilev2_mlsd_large_512_base2_bsize24.yaml',
                        type=str,
                        help="")
    return parser.parse_args()


def debug_lld_dataset_loader(data_iter):
    show_num = 1
    for i, batch in enumerate(data_iter):
        imgs = batch["xs"].cuda()
        batch_gt_confidence = batch["batch_gt_confidence"].cuda()
        batch_gt_offset_x = batch["batch_gt_offset_x"].cuda()
        batch_gt_offset_y = batch["batch_gt_offset_y"].cuda()
        batch_gt_line_index = batch["batch_gt_line_index"].cuda()
        batch_ignore_mask = batch["batch_ignore_mask"].cuda()
        batch_foreground_mask = batch["batch_foreground_mask"].cuda()
        batch_gt_line_id = batch["batch_gt_line_id"].cuda()

        # recovery img
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        img = imgs[0].cpu().detach().numpy().transpose(1, 2, 0)
        img = (img * std + mean) * 255
        img = img.astype(np.uint8)

        # gt recovery
        gt_confidence = batch_gt_confidence[0].cpu().detach().numpy()
        gt_offset_x = batch_gt_offset_x[0].cpu().detach().numpy()
        gt_offset_y = batch_gt_offset_y[0].cpu().detach().numpy()
        gt_line_index = batch_gt_line_index[0].cpu().detach().numpy()
        ignore_mask = batch_ignore_mask[0].cpu().detach().numpy()
        foreground_mask = batch_foreground_mask[0].cpu().detach().numpy()
        gt_line_id = batch_gt_line_id[0].cpu().detach().numpy()

        cls_i = 0  # road_fence_line
        gt_confidence_cls = gt_confidence[cls_i, :, :]
        gt_offset_x_cls = gt_offset_x[cls_i, :, :]
        gt_offset_y_cls = gt_offset_y[cls_i, :, :]
        gt_line_index_cls = gt_line_index[cls_i, :, :]
        ignore_mask_cls = ignore_mask[cls_i, :, :]
        foreground_mask_cls = foreground_mask[cls_i, :, :]
        gt_line_id_cls = gt_line_id[cls_i, :, :]

        # show
        plt.subplot(3, 2, 1)
        plt.title("input img")
        plt.imshow(img)

        plt.subplot(3, 2, 2)
        plt.imshow(gt_confidence_cls)
        plt.title("confident map")
        print(np.unique(gt_confidence_cls))

        plt.subplot(3, 2, 3)
        plt.imshow(gt_offset_x_cls)
        plt.title("x_offset map")

        plt.subplot(3, 2, 4)
        plt.imshow(gt_offset_y_cls)
        plt.title("y_offset map")

        plt.subplot(3, 2, 5)
        plt.imshow(gt_line_index_cls)
        plt.title("embedding map")
        # print(np.unique(gt_line_index_cls))

        plt.subplot(3, 2, 6)
        plt.imshow(ignore_mask_cls)
        # plt.title("ignore map")
        # plt.imshow(foreground_mask)
        # plt.title(gt_line_id_cls)

        plt.show()
        print('Done...')
        exit(1)


def debug_lld_dataset_loader2(data_iter):
    show_num = 1
    for i, batch in enumerate(data_iter):
        print("get in")
        imgs = batch["xs"].cuda()
        batch_gt_confidence = batch["batch_gt_confidence"].cuda()
        batch_gt_offset_x = batch["batch_gt_offset_x"].cuda()
        batch_gt_offset_y = batch["batch_gt_offset_y"].cuda()
        batch_gt_line_index = batch["batch_gt_line_index"].cuda()
        batch_ignore_mask = batch["batch_ignore_mask"].cuda()
        batch_foreground_mask = batch["batch_foreground_mask"].cuda()
        batch_gt_line_id = batch["batch_gt_line_id"].cuda()
        batch_gt_line_cls = batch["batch_gt_line_cls"].cuda()
        batch_foreground_expand_mask = batch["batch_foreground_expand_mask"].cuda()


        # recovery img
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        img = imgs[0].cpu().detach().numpy().transpose(1, 2, 0)
        img = (img * std + mean) * 255
        img = img.astype(np.uint8)

        # gt recovery
        gt_confidence = batch_gt_confidence[0].cpu().detach().numpy()
        gt_offset_x = batch_gt_offset_x[0].cpu().detach().numpy()
        gt_offset_y = batch_gt_offset_y[0].cpu().detach().numpy()
        gt_line_index = batch_gt_line_index[0].cpu().detach().numpy()
        ignore_mask = batch_ignore_mask[0].cpu().detach().numpy()
        foreground_mask = batch_foreground_mask[0].cpu().detach().numpy()
        gt_line_id = batch_gt_line_id[0].cpu().detach().numpy()
        cls_line_id = batch_gt_line_cls[0].cpu().detach().numpy()
        foreground_expand_mask = batch_foreground_expand_mask[0].cpu().detach().numpy()

        cls_i = 0  # road_fence_line
        gt_confidence_cls = gt_confidence[cls_i, :, :]
        gt_offset_x_cls = gt_offset_x[cls_i, :, :]
        gt_offset_y_cls = gt_offset_y[cls_i, :, :]
        gt_line_index_cls = gt_line_index[cls_i, :, :]
        ignore_mask_cls = ignore_mask[cls_i, :, :]
        foreground_mask_cls = foreground_mask[cls_i, :, :]
        gt_line_id_cls = gt_line_id[cls_i, :, :]
        gt_line_cls_cls = cls_line_id[4, :, :]
        foreground_expand_mask_cls = foreground_expand_mask[cls_i, :, :]


        # show
        plt.subplot(3, 2, 1)
        plt.title("input img")
        plt.imshow(img)

        plt.subplot(3, 2, 2)
        plt.imshow(gt_confidence_cls)
        plt.title("confident map")
        print(np.unique(gt_confidence_cls))

        plt.subplot(3, 2, 3)
        plt.imshow(gt_offset_x_cls)
        plt.title("x_offset map")

        plt.subplot(3, 2, 4)
        plt.imshow(gt_offset_y_cls)
        plt.title("y_offset map")

        plt.subplot(3, 2, 5)
        # plt.imshow(gt_line_index_cls)
        plt.title("embedding map")
        print(np.unique(gt_line_index_cls))
        plt.imshow(foreground_mask_cls)

        plt.subplot(3, 2, 6)
        plt.imshow(ignore_mask_cls)
        plt.title("ignore map")
        # plt.imshow(foreground_mask)
        # plt.imshow(gt_line_id_cls)
        # plt.imshow(gt_line_cls_cls)
        # plt.imshow(foreground_expand_mask_cls)

        plt.show()
        print('Done...')
        exit(1)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def debug_lld_dataset_loader3(data_iter):
    show_num = 10
    for i, batch in enumerate(data_iter):
        imgs = batch["xs"].cuda()
        batch_gt_confidence = batch["batch_gt_confidence"].cuda()
        batch_gt_offset_x = batch["batch_gt_offset_x"].cuda()
        batch_gt_offset_y = batch["batch_gt_offset_y"].cuda()
        batch_gt_line_index = batch["batch_gt_line_index"].cuda()
        batch_ignore_mask = batch["batch_ignore_mask"].cuda()
        batch_foreground_mask = batch["batch_foreground_mask"].cuda()

        # recovery img
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        img = imgs[0].cpu().detach().numpy().transpose(1, 2, 0)
        img = (img * std + mean) * 255
        img = img.astype(np.uint8)

        # gt recovery
        gt_confidence = batch_gt_confidence[0].cpu().detach().numpy()
        gt_offset_x = batch_gt_offset_x[0].cpu().detach().numpy()
        gt_offset_y = batch_gt_offset_y[0].cpu().detach().numpy()
        gt_line_index = batch_gt_line_index[0].cpu().detach().numpy()
        ignore_mask = batch_ignore_mask[0].cpu().detach().numpy()
        foreground_mask = batch_foreground_mask[0].cpu().detach().numpy()

        cls_i = 0  # road_fence_line
        pred_confidence = gt_confidence[cls_i:cls_i + 1, :, :]
        pred_offset_x = gt_offset_x[cls_i:cls_i + 1, :, :]
        pred_offset_y = gt_offset_y[cls_i:cls_i + 1, :, :]
        gt_line_index_cls = gt_line_index[cls_i:cls_i + 1, :, :]
        ignore_mask_cls = ignore_mask[cls_i:cls_i + 1, :, :]
        foreground_mask_cls = foreground_mask[cls_i:cls_i + 1, :, :]

        pred_offset_x = pred_offset_x.clip(-20, 20)
        pred_offset_y = pred_offset_y.clip(-20, 20)

        grid_size = 4
        # pred
        # pred_offset_x = sigmoid(pred_offset_x) * (grid_size - 1)
        # pred_offset_y = sigmoid(pred_offset_y) * (grid_size - 1)

        # gt
        pred_offset_x = pred_offset_x * (grid_size - 1)
        pred_offset_y = pred_offset_y * (grid_size - 1)

        pred_offset_x = pred_offset_x.round().astype(np.int).clip(0, grid_size - 1)
        pred_offset_y = pred_offset_y.round().astype(np.int).clip(0, grid_size - 1)

        _, h, w = pred_offset_x.shape
        pred_grid_x = np.arange(w).reshape(1, 1, w).repeat(h, axis=1) * grid_size
        pred_grid_y = np.arange(h).reshape(1, h, 1).repeat(w, axis=2) * grid_size
        pred_x = pred_grid_x + pred_offset_x
        pred_y = pred_grid_y + pred_offset_y

        # pred_x = pred_grid_x
        # pred_y = pred_grid_y

        # mask = np.zeros_like(pred_confidence, dtype=np.bool)
        mask = pred_confidence > 0.3

        img_show = img.copy()

        for _mask, _pred_emb, _pred_x, _pred_y, _pred_confidence, _pred_offset_y in zip(mask, gt_line_index_cls, pred_x, pred_y, pred_confidence, pred_offset_y):

            ys = _pred_y[_mask]
            xs = _pred_x[_mask]
            off_y = _pred_offset_y[_mask]

            ys, xs, off_y = np.flipud(ys), np.flipud(xs), np.flipud(off_y)  # 优先将y大的点排在前面

            for i, (_ys, _xs, _off_y) in enumerate(zip(ys, xs, off_y)):
                cv2.circle(img_show, (int(_xs), int(_ys)), 5, (255, 0, 0), 1)
                if i == 0:
                    print(_off_y)
                    cv2.circle(img_show, (int(_xs), int(_ys - _off_y)), 5, (0, 255, 0), 1)

            # cv2.circle(img_show, (int(ys[0]), int(xs[0])), 5, (0, 255, 0), 1)
            # print(pred_y[0, int(ys[0]), int(xs[0])])

        # plt.subplot(3, 1, 1)
        plt.imshow(img_show)

        # plt.subplot(3, 1, 2)
        # plt.imshow(pred_offset_y[0])

        # plt.subplot(3, 1, 3)
        # plt.imshow(pred_offset_x[0])
        plt.show()
        if i+1 >= show_num:
            exit(1)


if __name__ == "__main__":
    print("Start")
    cfg_path = "/mnt/data10/liyj/programs/ULSD-ISPRS/configs/lld_cfg.yaml"
    cfg = get_cfg_defaults()
    #print(cfg)
    args = get_args()
    args.config = cfg_path
    if args.config.endswith('\r'):
        args.config = args.config[:-1]
    print('using config: ',args.config.strip())
    cfg.merge_from_file(args.config)
    print(cfg)

    train_loader = get_train_dataloader(cfg, is_train=True)
    # val_loader   = get_val_dataloader(cfg, is_train=False)
    data_iter = tqdm(train_loader)


    # debug_lld_dataset_loader(data_iter)
    # debug_lld_dataset_loader2(data_iter)   # debug gt
    debug_lld_dataset_loader3(data_iter)   # debug decode
    print("Done.")