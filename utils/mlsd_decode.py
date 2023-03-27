import torch
from torch.nn import functional as F
import torch.nn as nn


def _gather_feat(feat, ind, mask=None):   # feat 1x480x1, ind 1x40
    dim  = feat.size(2) # 1
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def deccode_lines_TP_use_clsmap(tpMap, score_thresh=0.1, len_thresh=2, topk_n=1000, ksize=3, clsnum=21):
    '''
    tpMap:
        clsmap: tpMap[1, 26:38, :, :]
        displacement: tpMap[1, 20:24, :, :]
    '''
    # b, c, h, w = tpMap.shape
    # assert b == 1, 'only support bsize==1'
    cls_num = clsnum
    dis_start = (cls_num + 7) + 1
    cls_start = (cls_num + 7) + 7
    cls_end = cls_start + cls_num
    displacement = tpMap[:, dis_start:dis_start + 4, :, :]
    cls = tpMap[:, cls_start:cls_end, :, :]

    heat = torch.sigmoid(cls) # 1x256x256
    hmax = F.max_pool2d(heat, (ksize, ksize), stride=1, padding=(ksize - 1) // 2) # 1x256x256
    keep = (hmax == heat).float()
    heat = heat * keep
    #heat = heat.reshape(-1, ) # 65536

    b, c, h, w = heat.size()
    assert b == 1, 'only support bsize==1'
    topk_scores, topk_inds = torch.topk(heat.view(b, c, -1), topk_n)

    topk_inds = topk_inds % (h * w)
    topk_ys = (topk_inds / w).int().float()
    topk_xs = (topk_inds % w).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(b, -1), topk_n)
    topk_clses = (topk_ind / topk_n).int()
    topk_ys = _gather_feat(topk_ys.view(b, -1, 1), topk_ind).view(b, topk_n)
    topk_xs = _gather_feat(topk_xs.view(b, -1, 1), topk_ind).view(b, topk_n)

    valid_inx = torch.where(topk_score > score_thresh)
    topk_score = topk_score[valid_inx]
    topk_ys = topk_ys[valid_inx]
    topk_xs = topk_xs[valid_inx]
    xx = topk_xs.unsqueeze(1).long()
    yy = topk_ys.unsqueeze(1).long()
    center_ptss = torch.cat((xx, yy), dim=-1)

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
    scores = topk_score[valid_inx]            # torch[topk]

    return center_ptss, lines, lines_swap, scores