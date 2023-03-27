import os
import cv2
import json
import numpy as np
import math
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import shutil
import sys

from configs.mlsd_default import get_cfg_defaults
from torch.utils.data import Dataset, DataLoader
from dataset.dataset_mlsd import MLSD_LINE_Dataset, MLSD_LINE_Dataset_collate_fn


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        default= os.path.dirname(os.path.abspath(__file__))+ '/configs/mobilev2_mlsd_large_512_base2_bsize24.yaml',
                        type=str,
                        help="")
    return parser.parse_args()


def get_train_dataloader(cfg, is_train=True):
    dataset = MLSD_LINE_Dataset(cfg, is_train=is_train)

    dataloader_loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.sys.num_workers,
        drop_last=True,
        collate_fn=MLSD_LINE_Dataset_collate_fn
    )
    return dataloader_loader


def debug_lld_dataset_loader(data_iter):
    s_root = "/mnt/data10/liyj/data/debug"
    if os.path.exists(s_root):
        shutil.rmtree(s_root)
    os.mkdir(s_root)

    show_num = 1
    for i, batch in enumerate(data_iter):
        imgs = batch["xs"].cuda()
        label = batch["ys"].cuda()
        img_fns = batch["img_fns"]
        origin_imgs = batch["origin_imgs"]
        gt_lines_512 = batch["gt_lines_512"]
        gt_lines_tensor_512_list = batch["gt_lines_tensor_512_list"]
        sol_lines_512_all_tensor_list = batch["sol_lines_512_all_tensor_list"]

        # show one of them
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        img = imgs[0]
        label = label[0]
        img_fns = img_fns[0]
        gt_lines_512 = gt_lines_512[0]
        gt_lines_tensor_512_list = gt_lines_tensor_512_list[0]
        sol_lines_512_all_tensor_list = sol_lines_512_all_tensor_list[0]

        img = img.cpu().detach().numpy().transpose(1, 2, 0)
        img = (img * std + mean) * 255
        img = img.astype(np.uint8)

        sol_mask = label[0:19, :, :]
        #     tp_mask = np.concatenate((centermap, displacement_map, length_map, degree_map, cls_map), axis=0)
        tp_mask = label[19:38, :, :]
        junction_map = label[38, :, :]
        line_map = label[39, :, :]
        curve_map = label[40, :, :]

        sol_centermap = sol_mask[0].cpu().detach().numpy()
        centermap = tp_mask[0].cpu().detach().numpy()
        line_map = line_map.cpu().detach().numpy()
        junction_map = junction_map.cpu().detach().numpy()
        curve_map = curve_map.cpu().detach().numpy()

        img_ori = origin_imgs[0]
        img_raw = cv2.imread(img_fns)

        # draw gt line
        img = img.copy()
        for line in gt_lines_512:
            x1, y1, x2, y2 = [int(tmp) for tmp in line[:4]]
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1, 16)

        plt.subplot(2, 1, 1)
        plt.imshow(img[:, :, ::-1])
        plt.subplot(2, 1, 2)
        plt.imshow(centermap)

        # plt.subplot(3, 2, 1)
        # plt.imshow(img_raw[:, :, ::-1])
        #
        # plt.subplot(3, 2, 2)
        # plt.imshow(img)
        #
        # plt.subplot(3, 2, 3)
        # plt.imshow(centermap)
        #
        # plt.subplot(3, 2, 4)
        # plt.imshow(sol_centermap)
        #
        # plt.subplot(3, 2, 5)
        # plt.imshow(line_map)
        #
        # plt.subplot(3, 2, 6)
        # plt.imshow(junction_map)

        # plt.show()
        s_img_path = os.path.join(s_root, str(i) + ".jpg")
        cv2.imwrite(s_img_path, img)

        if i > show_num:
            exit(1)


if __name__ == "__main__":
    print("Start")
    cfg_path = "/mnt/data10/liyj/programs/ULSD-ISPRS/configs/mlsd_cfg.yaml"
    cfg = get_cfg_defaults()
    #print(cfg)
    args = get_args()
    args.config = cfg_path
    if args.config.endswith('\r'):
        args.config = args.config[:-1]
    print('using config: ', args.config.strip())
    cfg.merge_from_file(args.config)
    print(cfg)

    train_loader = get_train_dataloader(cfg, is_train=True)
    # val_loader   = get_val_dataloader(cfg, is_train=False)
    data_iter = tqdm(train_loader)
    debug_lld_dataset_loader(data_iter)
    print("Done.")