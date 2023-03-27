import cv2
import numpy as np


def F1_score_128(pred_lines_128_list, gt_lines_128_list, thickness=3):
    """
     @brief heat  F1 score, draw the lines to a 128 * 128 img
     @pred_lines_128 [ [x0, y0, x1, y1],  ... ]
     @gt_lines_128_list [ [x0, y0, x1, y1],  ... ]
    """
    pred_heatmap = np.zeros((128, 128), np.uint8)
    gt_heatmap = np.zeros((128, 128), np.uint8)

    for l in pred_lines_128_list:
        x0, y0, x1, y1 = l
        x0 = int(round(x0))
        y0 = int(round(y0))
        x1 = int(round(x1))
        y1 = int(round(y1))
        cv2.line(pred_heatmap, (x0, y0), (x1, y1), (1, 1, 1), thickness, 8)

    for l in gt_lines_128_list:
        x0, y0, x1, y1 = l
        x0 = int(round(x0))
        y0 = int(round(y0))
        x1 = int(round(x1))
        y1 = int(round(y1))
        cv2.line(gt_heatmap, (x0, y0), (x1, y1), (1, 1, 1), thickness, 8)

    pred_heatmap = np.array(pred_heatmap, np.float32)
    gt_heatmap = np.array(gt_heatmap, np.float32)

    intersection = np.sum(gt_heatmap * pred_heatmap)
    # union = np.sum(gt_heatmap) + np.sum(gt_heatmap)
    eps = 0.001
    # dice = (2. * intersection + eps) / (union + eps)

    recall = intersection / (np.sum(gt_heatmap) + eps)
    precision = intersection / (np.sum(pred_heatmap) + eps)

    fscore = (2 * precision * recall) / (precision + recall + eps)
    return fscore, recall, precision


def F1_score_128_curve(pred_lines_128_list, gt_lines_128_list, pred_curve_list, gt_curve_list, thickness=3):
    """
     @brief heat  F1 score, draw the lines to a 128 * 128 img
     @pred_lines_128 [ [x0, y0, x1, y1],  ... ]
     @gt_lines_128_list [ [x0, y0, x1, y1],  ... ]
     @pred_curve_list [[[[x1,y1,x2,y2], [x1,y1,x2,y2]], ...cls]], [[[x1,y1,x2,y2], [x1,y1,x2,y2], ... cls]] 512
     @gt_curve_list [[x1,y1,x2,y2], [x1,y1,x2,y2], ...] 512
    """
    pred_heatmap = np.zeros((128, 128), np.uint8)
    gt_heatmap = np.zeros((128, 128), np.uint8)

    for l in pred_lines_128_list:
        x0, y0, x1, y1 = l
        x0 = int(round(x0))
        y0 = int(round(y0))
        x1 = int(round(x1))
        y1 = int(round(y1))
        cv2.line(pred_heatmap, (x0, y0), (x1, y1), (1, 1, 1), thickness, 8)

    for c in pred_curve_list:
        cl = c[0]
        for cc in cl:
            x0 = int(round(cc[0] / 4))
            y0 = int(round(cc[1] / 4))
            x1 = int(round(cc[2] / 4))
            y1 = int(round(cc[3] / 4))
            cv2.line(pred_heatmap, (x0, y0), (x1, y1), (1, 1, 1), thickness, 8)

    for l in gt_lines_128_list:
        x0, y0, x1, y1 = l
        x0 = int(round(x0))
        y0 = int(round(y0))
        x1 = int(round(x1))
        y1 = int(round(y1))
        cv2.line(gt_heatmap, (x0, y0), (x1, y1), (1, 1, 1), thickness, 8)

    for c in gt_curve_list:
        for cc in c:
            x0 = int(round(cc[0] / 4))
            y0 = int(round(cc[1] / 4))
            x1 = int(round(cc[2] / 4))
            y1 = int(round(cc[3] / 4))
            cv2.line(gt_heatmap, (x0, y0), (x1, y1), (1, 1, 1), thickness, 8)

    pred_heatmap = np.array(pred_heatmap, np.float32)
    gt_heatmap = np.array(gt_heatmap, np.float32)

    intersection = np.sum(gt_heatmap * pred_heatmap)
    # union = np.sum(gt_heatmap) + np.sum(gt_heatmap)
    eps = 0.001
    # dice = (2. * intersection + eps) / (union + eps)

    recall = intersection / (np.sum(gt_heatmap) + eps)
    precision = intersection / (np.sum(pred_heatmap) + eps)

    fscore = (2 * precision * recall) / (precision + recall + eps)
    # (gt - pred) for every 8 x y map
    num = pred_heatmap.shape[1] // 8
    gt_minus_pred = []
    recall_list = []
    precision_list = []
    for i in range(num):
        j = i + 1
        pred = pred_heatmap[(8 * i):(8 * j)]
        gt = gt_heatmap[(8 * i):(8 * j)]
        inter = np.sum(pred * gt)
        union = np.sum(pred) + np.sum(gt)
        relative_error = abs(union - inter) / (union + 0.001)
        # relative_error = 1 - relative_error
        gt_minus_pred.append(relative_error)
        recall = inter / (np.sum(gt) + 0.001)
        precision = inter / (np.sum(pred) + 0.001)
        recall_list.append(recall)
        precision_list.append(precision)

    return fscore, recall, precision, gt_minus_pred, recall_list, precision_list


def F1_score_512(pred_lines_512_list, gt_lines_512_list, thickness=3):
    """
     @brief heat  F1 score, draw the lines to a 128 * 128 img
     @pred_lines_128 [ [x0, y0, x1, y1],  ... ]
     @gt_lines_128_list [ [x0, y0, x1, y1],  ... ]
    """
    pred_heatmap = np.zeros((512, 512), np.uint8)
    gt_heatmap = np.zeros((512, 512), np.uint8)

    for l in pred_lines_512_list:
        x0, y0, x1, y1 = l
        x0 = int(round(x0))
        y0 = int(round(y0))
        x1 = int(round(x1))
        y1 = int(round(y1))
        cv2.line(pred_heatmap, (x0, y0), (x1, y1), (1, 1, 1), thickness, 8)

    for l in gt_lines_512_list:
        x0, y0, x1, y1 = l
        x0 = int(round(x0))
        y0 = int(round(y0))
        x1 = int(round(x1))
        y1 = int(round(y1))
        cv2.line(gt_heatmap, (x0, y0), (x1, y1), (1, 1, 1), thickness, 8)

    pred_heatmap = np.array(pred_heatmap, np.float32)
    gt_heatmap = np.array(gt_heatmap, np.float32)

    intersection = np.sum(gt_heatmap * pred_heatmap)
    # union = np.sum(gt_heatmap) + np.sum(gt_heatmap)
    eps = 0.001
    # dice = (2. * intersection + eps) / (union + eps)

    recall = intersection / (np.sum(gt_heatmap) + eps)
    precision = intersection / (np.sum(pred_heatmap) + eps)

    fscore = (2 * precision * recall) / (precision + recall + eps)
    return fscore, recall, precision


def msTPFP(line_pred, line_gt, threshold):
    line_pred = line_pred.reshape(-1, 2, 2)[:, :, ::-1]  # (topk, 2, 2) eg: (200, 2, 2)
    line_gt = line_gt.reshape(-1, 2, 2)[:, :, ::-1]
    diff = ((line_pred[:, None, :, None] - line_gt[:, None]) ** 2).sum(-1)
    diff = np.minimum(
        diff[:, :, 0, 0] + diff[:, :, 1, 1], diff[:, :, 0, 1] + diff[:, :, 1, 0]
    )

    choice = np.argmin(diff, 1)
    dist = np.min(diff, 1)
    hit = np.zeros(len(line_gt), np.bool)
    tp = np.zeros(len(line_pred), np.float)
    fp = np.zeros(len(line_pred), np.float)
    for i in range(len(line_pred)):
        if dist[i] < threshold and not hit[choice[i]]:
            hit[choice[i]] = True
            tp[i] = 1
        else:
            fp[i] = 1
    return tp, fp


def TPFP(lines_dt, lines_gt, threshold):
    lines_dt = lines_dt.reshape(-1, 2, 2)[:, :, ::-1]
    lines_gt = lines_gt.reshape(-1, 2, 2)[:, :, ::-1]
    diff = ((lines_dt[:, None, :, None] - lines_gt[:, None]) ** 2).sum(-1)
    diff = np.minimum(
        diff[:, :, 0, 0] + diff[:, :, 1, 1], diff[:, :, 0, 1] + diff[:, :, 1, 0]
    )
    choice = np.argmin(diff, 1)
    dist = np.min(diff, 1)
    hit = np.zeros(len(lines_gt), np.bool)
    tp = np.zeros(len(lines_dt), np.float)
    fp = np.zeros(len(lines_dt), np.float)

    for i in range(lines_dt.shape[0]):
        if dist[i] < threshold and not hit[choice[i]]:
            hit[choice[i]] = True
            tp[i] = 1
        else:
            fp[i] = 1
    return tp, fp


def AP(tp, fp):
    recall = tp
    precision = tp / np.maximum(tp + fp, 1e-9)

    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))

    for i in range(precision.size - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])
    i = np.where(recall[1:] != recall[:-1])[0]

    ap = np.sum((recall[i + 1] - recall[i]) * precision[i + 1])

    return ap