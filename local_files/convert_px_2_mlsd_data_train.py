import os
import shutil
from tqdm import tqdm
import json
import cv2
import copy
import numpy as np
from random import sample


def check_json_path(json_path):
    check_ok = True
    with open(json_path, 'r') as fp:
        contens = json.load(fp)
    objects = contens['objects']
    for i, c in enumerate(objects):
        # proc line, may has point, 'points'
        if 'lines_and_labels' in c.keys():
            lal = c['lines_and_labels']
            cls = lal[1]
            points = lal[0]

            if len(points) == 2:
                line = [points[0][0], points[0][1], points[1][0], points[1][1], cls]
                if line[0] == line[2] and line[1] == line[3]:
                    print("check fail points equal:", line)
                    check_ok = False
            if len(points) > 2:
                if 'line' in cls:
                    print("check fail line has more 2 points:", lal)
                    check_ok = False
        else:
            print("there is no lines:", c)
            check_ok = False

    return check_ok


def convert_big_data_2_dataset(src_root, dst_root_dir):
    b_save_vis = True
    if 1:
        if os.path.exists(dst_root_dir):
            shutil.rmtree(dst_root_dir)
        os.mkdir(dst_root_dir)

        for tmp in ['train', 'test']:
            tmp_path = os.path.join(dst_root_dir, tmp)
            if os.path.exists(tmp_path):
                shutil.rmtree(tmp_path)
            os.mkdir(tmp_path)

        for tmp in ['images', 'jsons', 'vis']:
            os.mkdir(os.path.join(dst_root_dir, 'train', tmp))
            os.mkdir(os.path.join(dst_root_dir, 'test', tmp))

    src_json_root = os.path.join(src_root, 'json')
    src_png_root = os.path.join(src_root, 'png')
    json_names = [name for name in os.listdir(src_json_root) if name.endswith('.json')]

    print("Processing Data ...")
    count = 0
    valid_count = 0
    proc_num = len(json_names)
    all_cls_names = []
    for json_name in tqdm(json_names):
        count = count + 1
        src_json_path = os.path.join(src_json_root, json_name)
        src_png_path = os.path.join(src_png_root, json_name.replace('.json', '.png'))

        if not os.path.exists(src_json_path):
            print("Skip:", src_json_path)
            continue

        if not os.path.exists(src_png_path):
            print("Skip:", src_png_path)
            continue

        b_check_json = check_json_path(src_json_path)
        if not b_check_json:
            print("src_json_pathis check fail: {} ".format(src_json_path))
            continue

        with open(src_json_path, 'r') as fp:
            contens = json.load(fp)

        img = cv2.imread(src_png_path)
        img_vis = copy.deepcopy(img)
        h, w, _ = img.shape
        objects = contens['objects']
        streanID = contens['streamId']

        new_json = {
            'version': '4.5.6',
            'flags': {},
            'shapes': [],
            'imagePath': json_name.replace('.json', '.png'),
            'imageData': None,
            'imageHeight': h,
            'imageWidth': w,
            "streamId": streanID,
        }

        for i, c in enumerate(objects):
            if 'lines_and_labels' in c.keys():
                lal = c['lines_and_labels']
                if 'id' in c:
                    id = c['id']
                else:
                    id = i

                cls = lal[1]
                points = lal[0]
                # in case, more than 2 points in line
                if len(points) > 2:
                    if 'line' in cls:
                        print("proc more 2 points line:", c)
                        points = [points[0], points[-1]]

                # draw line
                if b_save_vis:
                    pt_start = points[0]
                    pt_end = points[1]
                    cv2.line(img_vis, (int(pt_start[0]), int(pt_start[1])),
                             (int(pt_end[0]), int(pt_end[1])), (0, 255, 0), 2, 16)

                item = {
                    "label": cls,
                    "points": points,
                    "group_id": None,
                    "shape_type": "line",
                    "flag": {},
                    "id": id,
                }
                new_json['shapes'].append(item)

                # show cls infos
                if cls not in all_cls_names:
                    all_cls_names.append(cls)
                    # print("all_cls_names:", all_cls_names)
            else:
                print("skip points:", c)

        # save test, fake
        if count % 10 == 0:
            dst_json_path = os.path.join(dst_root_dir, "test", 'jsons', json_name)
            dst_png_path = os.path.join(dst_root_dir, "test", 'images', json_name.replace('.json', '.png'))

            with open(dst_json_path, 'w') as sf:
                json.dump(new_json, sf)
            shutil.copy(src_png_path, dst_png_path)

        else:
            # save train
            dst_json_path = os.path.join(dst_root_dir, "train", 'jsons', json_name)
            dst_png_path = os.path.join(dst_root_dir, "train", 'images', json_name.replace('.json', '.png'))
            dst_vis_path = os.path.join(dst_root_dir, "train", 'vis', json_name.replace('.json', '.jpg'))

        with open(dst_json_path, 'w') as sf:
            json.dump(new_json, sf)
        shutil.copy(src_png_path, dst_png_path)

        if b_save_vis:
            cv2.imwrite(dst_vis_path, img_vis)
        valid_count = valid_count + 1

        # exit(1)

    print("all_cls_names:", all_cls_names)
    print("count:", count)
    print("valid_count:", valid_count)
    print("Processing Data Done...")
    # os.symlink(src, dst)


if __name__ == "__main__":
    print("Start Proc..")
    # px1 data
    # src_root = "/userdata/nas01/training_data/edge_line/edge_line_indoor_liyj__px1_0919"
    # dst_root_dir = "/userdata/nas01/algorithm_data/data_disk/liyj/data/training_data/lines_0921"

    # px2 data
    src_root = "/mnt/nas01/algorithm_data/training_data/edge_line/indoor_edge_line_1219"
    dst_root_dir = "/mnt/data10/liyj/programs/ULSD-ISPRS/dataset/mlsd_20230405"

    convert_big_data_2_dataset(src_root, dst_root_dir)
    print("End Proc..")
