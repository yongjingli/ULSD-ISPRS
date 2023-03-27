import math
import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "focal_neg_loss_with_logits",
    "weighted_bce_with_logits",
]

def _sigmoid(x):
  y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
  return y

def focal_neg_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
    '''
    pred = _sigmoid(pred)

    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos  = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss

def focal_neg_loss_with_logits(preds, gt, alpha=2, belta=4):
    """
    borrow from https://github.com/princeton-vl/CornerNet
    """

    preds = torch.sigmoid(preds)
    #preds = _sigmoid(preds)

    pos_inds = gt.eq(1)
    neg_inds = gt.lt(1)

#     pos_inds = gt.gt(0)
#     neg_inds = gt.eq(0)

    neg_weights = torch.pow(1 - gt[neg_inds], belta)

    loss = 0
    pos_pred = preds[pos_inds]
    neg_pred = preds[neg_inds]

    pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, alpha)
    neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, alpha) * neg_weights

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if pos_pred.nelement() == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (3*pos_loss + neg_loss) / num_pos

    return loss


# def weighted_bce_with_logits(out, gt, pos_w=1.0, neg_w=30.0):
#     pos_mask = torch.where(gt == 1, torch.ones_like(gt), torch.zeros_like(gt))
#     neg_mask = torch.ones_like(pos_mask) - pos_mask

#     losses = F.binary_cross_entropy_with_logits(out, gt, reduction='none')

#     loss_neg = (losses * neg_mask).sum() / (torch.sum(neg_mask))
#     loss_v = loss_neg * neg_w

#     pos_sum = torch.sum(pos_mask)
#     if pos_sum != 0:
#         loss_pos = (losses * pos_mask).sum() / pos_sum
#         loss_v += (loss_pos * pos_w)
#     return loss_v


def weighted_bce_with_logits(out, gt, pos_w=1.0, neg_w=30.0):
    pos_mask = torch.where(gt != 0.0, torch.ones_like(gt), torch.zeros_like(gt))
    #pos_mask = torch.where(gt == 1, torch.ones_like(gt), torch.zeros_like(gt))
    neg_mask = torch.ones_like(pos_mask) - pos_mask
    loss = F.binary_cross_entropy_with_logits(out, gt, reduction='none')
    loss_pos = (loss * pos_mask).sum() / ( torch.sum(pos_mask) + 1e-5)
    loss_neg = (loss * neg_mask).sum() / ( torch.sum(neg_mask) + 1e-5)
    loss = loss_pos * pos_w + loss_neg * neg_w
    return loss


def weighted_bce_with_logits(out, gt, pos_w=1.0, neg_w=30.0):
    pos_mask = torch.where(gt != 0, torch.ones_like(gt), torch.zeros_like(gt))
    neg_mask = torch.ones_like(pos_mask) - pos_mask

    loss = F.binary_cross_entropy_with_logits(out, gt, reduction='none')
    loss_pos = (loss * pos_mask).sum() / torch.sum(pos_mask)
    loss_neg = (loss * neg_mask).sum() / torch.sum(neg_mask)

    loss = loss_pos * pos_w + loss_neg * neg_w
    return loss


def displacement_loss_func3(pred_dis, gt_dis, gt_center_mask=None):
    # only consider non zero part
    x0 = gt_dis[:, 0, :, :]
    y0 = gt_dis[:, 1, :, :]
    x1 = gt_dis[:, 2, :, :]
    y1 = gt_dis[:, 3, :, :]

    # if gt_center_mask is not None:
    #     pos_mask = torch.where(gt_center_mask > 0.9, torch.ones_like(x0), torch.zeros_like(x0))
    # else:
    #     pos_v = x0.abs() + y0.abs() + x1.abs() + y1.abs()
    #     pos_mask = torch.where(pos_v != 0, torch.ones_like(x0), torch.zeros_like(x0))
    pos_v = x0.abs() + y0.abs() + x1.abs() + y1.abs()
    pos_mask = torch.where(pos_v != 0, torch.ones_like(x0), torch.zeros_like(x0))
    pos_mask_sum = pos_mask.sum()

    pos_mask = pos_mask.unsqueeze(1) # ???

    pred_dis = pred_dis * pos_mask
    gt_dis = gt_dis * pos_mask

    # self define
    gt_x_len = torch.abs(gt_dis[:, 0, :, :] - gt_dis[:, 2, :, :])
    gt_y_len = torch.abs(gt_dis[:, 1, :, :] - gt_dis[:, 3, :, :])

    mask_x_y = gt_x_len < gt_y_len
    mask_y_x = gt_x_len > gt_y_len

    gt_y_x_ratio = gt_y_len/gt_x_len
    gt_y_x_ratio = torch.clamp(gt_y_x_ratio, min=0, max=4.0)

    gt_x_y_ratio = gt_x_len/gt_y_len
    gt_x_y_ratio = torch.clamp(gt_x_y_ratio, min=0, max=4.0)

    gt_weights = torch.ones_like(gt_dis)
    # gt_weights[:, 0, :, :][mask_x_y] = 4.0
    # gt_weights[:, 2, :, :][mask_x_y] = 4.0
    # gt_weights[:, 1, :, :][mask_y_x] = 4.0
    # gt_weights[:, 3, :, :][mask_y_x] = 4.0

    gt_weights[:, 0, :, :][mask_x_y] = gt_y_x_ratio[mask_x_y]
    gt_weights[:, 2, :, :][mask_x_y] = gt_y_x_ratio[mask_x_y]

    gt_weights[:, 1, :, :][mask_y_x] = gt_x_y_ratio[mask_y_x]
    gt_weights[:, 3, :, :][mask_y_x] = gt_x_y_ratio[mask_y_x]

    displacement_loss1 = F.smooth_l1_loss(pred_dis, gt_dis, reduction='none')
    displacement_loss1 = displacement_loss1 * gt_weights
    displacement_loss1 = displacement_loss1.sum(axis=[1])

    # swap pt
    pred_dis2 = torch.cat((pred_dis[:, 2:, :, :], pred_dis[:, :2, :, :]), dim=1)
    displacement_loss2 = F.smooth_l1_loss(pred_dis2, gt_dis, reduction='none')
    displacement_loss2 = displacement_loss2 * gt_weights
    displacement_loss2 = displacement_loss2.sum(axis=[1])

    # displacement_loss1 = displacement_loss1[:, 0, :, :] + displacement_loss1[:, 1, :, :]

    displacement_loss = displacement_loss1.min(displacement_loss2)

    #if gt_center_mask is not None:
    #    displacement_loss = displacement_loss * gt_center_mask


    displacement_loss = displacement_loss.sum() / pos_mask_sum

    return displacement_loss


def len_and_angle_loss_func(pred_len, pred_angle, gt_len, gt_angle):
    pred_len = torch.sigmoid(pred_len)
    pred_angle = torch.sigmoid(pred_angle)
    # only consider non zero part
    pos_mask = torch.where(gt_len != 0, torch.ones_like(gt_len), torch.zeros_like(gt_len))
    pos_mask_sum = pos_mask.sum()

    len_loss = F.smooth_l1_loss(pred_len, gt_len, reduction='none')
    len_loss = len_loss * pos_mask
    len_loss = len_loss.sum() / pos_mask_sum

    angle_loss = F.smooth_l1_loss(pred_angle, gt_angle, reduction='none')
    angle_loss = angle_loss * pos_mask
    angle_loss = angle_loss.sum() / pos_mask_sum

    return len_loss, angle_loss


def class_loss_func(pred_cls, gt_cls):
    cls_loss = focal_neg_loss_with_logits(pred_cls, gt_cls)
    # cls_loss = weighted_bce_with_logits(pred_cls, gt_cls)
    return cls_loss


def displacement_loss_func(pred_dis, gt_dis, gt_center_mask=None):
    # only consider non zero part
    x0 = gt_dis[:, 0, :, :]
    y0 = gt_dis[:, 1, :, :]
    x1 = gt_dis[:, 2, :, :]
    y1 = gt_dis[:, 3, :, :]

    # if gt_center_mask is not None:
    #     pos_mask = torch.where(gt_center_mask > 0.9, torch.ones_like(x0), torch.zeros_like(x0))
    # else:
    #     pos_v = x0.abs() + y0.abs() + x1.abs() + y1.abs()
    #     pos_mask = torch.where(pos_v != 0, torch.ones_like(x0), torch.zeros_like(x0))
    pos_v = x0.abs() + y0.abs() + x1.abs() + y1.abs()
    pos_mask = torch.where(pos_v != 0, torch.ones_like(x0), torch.zeros_like(x0))
    pos_mask_sum = pos_mask.sum()

    pos_mask = pos_mask.unsqueeze(1) # ???

    pred_dis = pred_dis * pos_mask
    gt_dis = gt_dis * pos_mask

    displacement_loss1 = F.smooth_l1_loss(pred_dis, gt_dis, reduction='none').sum(axis=[1])

    # swap pt
    pred_dis2 = torch.cat((pred_dis[:, 2:, :, :], pred_dis[:, :2, :, :]), dim=1)
    displacement_loss2 = F.smooth_l1_loss(pred_dis2, gt_dis, reduction='none').sum(axis=[1])
    displacement_loss = displacement_loss1.min(displacement_loss2)
    #if gt_center_mask is not None:
    #    displacement_loss = displacement_loss * gt_center_mask

    displacement_loss = displacement_loss.sum() / pos_mask_sum

    return displacement_loss


def deccode_lines_TP(tpMap, score_thresh=0.1, len_thresh=2, topk_n=1000, ksize=3):
    '''
    tpMap:
        center: tpMap[1, 0, :, :]
        displacement: tpMap[1, 1:5, :, :]
    '''
    b, c, h, w = tpMap.shape
    assert b == 1, 'only support bsize==1'
    displacement = tpMap[:, 20:24, :, :]
    center = tpMap[:, 19, :, :]
    heat = torch.sigmoid(center) # 1x256x256
    hmax = F.max_pool2d(heat, (ksize, ksize), stride=1, padding=(ksize - 1) // 2) # 1x256x256
    keep = (hmax == heat).float()
    heat = heat * keep
    heat = heat.reshape(-1, ) # 65536

    scores, indices = torch.topk(heat, topk_n, dim=-1, largest=True) # calculate topk's value and index scores, indices: Torch [topkn]
    valid_inx = torch.where(scores > score_thresh)
    scores = scores[valid_inx]
    indices = indices[valid_inx]  # 筛选出来的scores indices: torch[小于等于topkn]

    yy = torch.floor_divide(indices, w).unsqueeze(-1) # indices // w 求取在特征图中y的坐标点
    xx = torch.fmod(indices, w).unsqueeze(-1)   # indices % w 求取在特征图中x的坐标点
    center_ptss = torch.cat((xx, yy), dim=-1) # cat 组成中心点坐标

    start_point = center_ptss + displacement[0, :2, yy, xx].reshape(2, -1).permute(1, 0) # torch [topk, 2] permute类似numpy中的transpose
    end_point = center_ptss + displacement[0, 2:, yy, xx].reshape(2, -1).permute(1, 0)

    lines = torch.cat((start_point, end_point), dim=-1)

    lines_swap = torch.cat((end_point, start_point), dim=-1)

    all_lens = (end_point - start_point) ** 2
    all_lens = all_lens.sum(dim=-1)
    all_lens = torch.sqrt(all_lens)
    valid_inx = torch.where(all_lens > len_thresh) # 大于线段长度的阈值的线段

    center_ptss = center_ptss[valid_inx]  # torch[topk]
    lines = lines[valid_inx]              # torch[topk, 4]
    lines_swap = lines_swap[valid_inx]    # torch[topk, 4]
    scores = scores[valid_inx]            # torch[topk]

    return center_ptss, lines, lines_swap, scores