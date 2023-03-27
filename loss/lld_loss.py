import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._func import focal_neg_loss_with_logits, focal_neg_loss  # , weighted_bce_with_logits
# from mlsd_pytorch.utils.decode import deccode_lines_TP
# from utils.decode import deccode_lines_TP

__all__ = [
    "CurveSegmentLoss",
]


def weighted_bce_with_logits(out, gt, pos_w=1.0, neg_w=30.0):
    pos_mask = torch.where(gt != 0, torch.ones_like(gt), torch.zeros_like(gt))
    neg_mask = torch.ones_like(pos_mask) - pos_mask

    loss = F.binary_cross_entropy_with_logits(out, gt, reduction='none')
    loss_pos = (loss * pos_mask).sum() / torch.sum(pos_mask)
    loss_neg = (loss * neg_mask).sum() / torch.sum(neg_mask)

    loss = loss_pos * pos_w + loss_neg * neg_w
    return loss


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

    pos_mask = pos_mask.unsqueeze(1)  # ???

    pred_dis = pred_dis * pos_mask
    gt_dis = gt_dis * pos_mask

    displacement_loss1 = F.smooth_l1_loss(pred_dis, gt_dis, reduction='none').sum(axis=[1])

    # swap pt
    pred_dis2 = torch.cat((pred_dis[:, 2:, :, :], pred_dis[:, :2, :, :]), dim=1)
    displacement_loss2 = F.smooth_l1_loss(pred_dis2, gt_dis, reduction='none').sum(axis=[1])
    displacement_loss = displacement_loss1.min(displacement_loss2)
    # if gt_center_mask is not None:
    #    displacement_loss = displacement_loss * gt_center_mask

    displacement_loss = displacement_loss.sum() / pos_mask_sum

    return displacement_loss


def class_loss_func(pred_cls, gt_cls):
    cls_loss = focal_neg_loss_with_logits(pred_cls, gt_cls)

    return cls_loss


class LldCurveLoss(nn.Module):
    def __init__(self, cfg):
        super(LldCurveLoss, self).__init__()
        self.input_size = cfg.datasets.input_size

        self.line_weight = torch.tensor(cfg.loss.line_weight)
        self.focal_loss_gamma = cfg.loss.focal_loss_gamma
        self.pull_margin = cfg.loss.pull_margin
        self.push_margin = cfg.loss.push_margin

        self.confidence_weight = cfg.loss.confidence_weight
        self.offset_weight = cfg.loss.offset_weight
        self.emb_weight = cfg.loss.emb_weight
        self.emb_id_weight = cfg.loss.emb_id_weight
        self.cls_weight = cfg.loss.cls_weight
        self.cls_loss = torch.nn.BCELoss(reduction='none')
        self.num_classes = cfg.model.num_classes


    def forward(self, outputs, batch_gt_confidence,
                batch_gt_offset_x,
                batch_gt_offset_y,
                batch_gt_line_index,
                batch_ignore_mask,
                batch_foreground_mask,
                batch_gt_line_id,
                batch_gt_line_cls,
                batch_foreground_expand_mask):

        # loss
        zero = torch.tensor(0.0).to(outputs)
        loss = {
            "confidence_loss": zero.clone(),
            "offset_loss": zero.clone(),
            "embedding_loss": zero.clone(),
            "embedding_id_loss": zero.clone(),
            "cls_loss": zero.clone()
        }

        confidence_losses = zero.clone()
        offset_losses = zero.clone()
        embedding_losses = zero.clone()
        embedding_id_losses = zero.clone()
        cls_losses = zero.clone()

        self.line_weight.to(outputs)
        # self.focal_loss_gamma.to(outputs)

        cls_num = batch_gt_confidence.shape[1]
        pred_confs = outputs[:, cls_num * 0:cls_num * 1, :, :]
        # pred_offset_xs = outputs[:, cls_num * 1:cls_num * 2, :, :]
        # pred_offset_ys = outputs[:, cls_num * 2:cls_num * 3, :, :]

        pred_offsets = outputs[:, cls_num * 1:cls_num * 3, :, :]

        pred_embs = outputs[:, cls_num * 3:cls_num * 4, :, :]
        pred_embs_id = outputs[:, cls_num * 4:cls_num * 5, :, :]

        pred_cls = outputs[:, cls_num * 5:cls_num * 5 + self.num_classes, :, :]

        cls_weights = [1.0] * cls_num
        for i in range(cls_num):
            start_p, end_p = i, i + 1
            cls_weight = cls_weights[i]
            pred_conf = pred_confs[:, start_p:end_p, :, :]
            # pred_offset_x = pred_offset_xs[:, start_p:end_p, :, :]
            # pred_offset_y = pred_offset_ys[:, start_p:end_p, :, :]
            pred_offset = pred_offsets[:, start_p:end_p + 1, :, :]

            pred_emb = pred_embs[:, start_p:end_p, :, :]
            pred_emb_id = pred_embs_id[:, start_p:end_p, :, :]

            gt_conf = batch_gt_confidence[:, start_p:end_p, :, :]
            gt_offset_x = batch_gt_offset_x[:, start_p:end_p, :, :]
            gt_offset_y = batch_gt_offset_y[:, start_p:end_p, :, :]

            gt_offset = torch.cat([gt_offset_x, gt_offset_y], 1)

            gt_line_index = batch_gt_line_index[:, start_p:end_p, :, :]
            gt_ignore_mask = batch_ignore_mask[:, start_p:end_p, :, :]
            gt_foreground_mask = batch_foreground_mask[:, start_p:end_p, :, :]
            gt_foreground_expand_mask = batch_foreground_expand_mask[:, start_p:end_p, :, :]

            gt_line_id = batch_gt_line_id[:, start_p:end_p, :, :]

            gt_line_cls = batch_gt_line_cls

            # confidence loss
            gt_ignore_mask = gt_ignore_mask > 0
            if gt_ignore_mask.sum() > 0:
                bce_loss = F.binary_cross_entropy_with_logits(pred_conf, gt_conf, reduction="none")
                p = torch.exp(-1.0 * bce_loss)
                confidence_loss = F.binary_cross_entropy_with_logits(
                    pred_conf, gt_conf, reduction="none", pos_weight=self.line_weight
                )
                confidence_loss = torch.pow(1.0 - p, self.focal_loss_gamma) * confidence_loss

                confidence_loss = confidence_loss[gt_ignore_mask].mean()

                # Line confidence loss (Dice Loss)
                dice_loss = 0.0
                eps = 1.0e-5
                dice_loss_weight = 5.0
                # dice_loss_weight = 0.0
                pred_conf = torch.sigmoid(pred_conf)
                for pred, gt, msk in zip(pred_conf, gt_conf, gt_ignore_mask):
                    if msk.sum() == 0:
                        continue
                    pred = pred[msk]
                    gt = gt[msk]
                    dice_loss += 1 - ((pred * gt).sum() + eps) / (pred.pow(2).sum() + gt.pow(2).sum() + eps)
                    pred = 1 - pred
                    gt = 1 - gt
                    dice_loss += - ((pred * gt).sum() + eps) / (pred.pow(2).sum() + gt.pow(2).sum() + eps)
                confidence_loss += dice_loss_weight * dice_loss / outputs.shape[0]

                confidence_losses += confidence_loss * cls_weight
            else:
                confidence_loss = zero.clone()
                confidence_losses += confidence_loss * cls_weight

            # offset losses
            gt_foreground_mask = gt_foreground_mask > 0
            gt_foreground_expand_mask = gt_foreground_expand_mask > 0
            if gt_foreground_mask.sum() > 0:
                # offset loss
                # pred_offset_x = torch.sigmoid(pred_offset_x)
                # pred_offset_y = torch.sigmoid(pred_offset_y)
                #
                # offset_loss_x = F.mse_loss(pred_offset_x, gt_offset_x, reduction="none")
                # offset_loss_x = offset_loss_x[gt_foreground_mask]
                #
                # offset_loss_y = F.mse_loss(pred_offset_y, gt_offset_y, reduction="none")
                # offset_loss_y = offset_loss_y[gt_foreground_mask]
                # offset_loss = offset_loss_x.mean() + offset_loss_y.mean()

                pred_offset = torch.sigmoid(pred_offset)
                offset_loss = F.mse_loss(pred_offset, gt_offset, reduction="none")
                offset_loss = offset_loss[gt_foreground_mask.expand(-1, 2, -1, -1)]
                offset_loss = offset_loss.mean()

                offset_losses += offset_loss * cls_weight
            else:
                offset_loss = zero.clone()
                offset_losses += offset_loss * cls_weight

            # embedding_losses
            if gt_foreground_mask.sum() > 0:
                num_chn = end_p - start_p

                pull_loss, push_loss = 0.0, 0.0
                for pred, gt, msk in zip(pred_emb, gt_line_index, gt_foreground_mask):
                    N = msk.sum()
                    if N == 0:
                        continue
                    gt = gt[msk]
                    gt_square = torch.zeros(N, N).to(outputs)
                    for i in range(N):
                        gt_row = gt.clone()
                        gt_row[gt == gt[i]] = 1
                        gt_row[gt != gt[i]] = 2
                        gt_square[i] = gt_row
                    msk = msk.expand(num_chn, -1, -1)
                    pred = pred[msk]
                    pred_row = pred.view(num_chn, 1, N).expand(num_chn, N, N)
                    pred_col = pred.view(num_chn, N, 1).expand(num_chn, N, N)
                    pred_sqrt = torch.norm(pred_col - pred_row, dim=0)
                    pred_dist = pred_sqrt[gt_square == 1] - self.pull_margin
                    pred_dist[pred_dist < 0] = 0
                    pull_loss += pred_dist.mean()
                    if gt_square.max() == 2:
                        pred_dist = self.push_margin - pred_sqrt[gt_square == 2]
                        pred_dist[pred_dist < 0] = 0
                        push_loss += pred_dist.mean()
                embedding_losses += (pull_loss + push_loss) / outputs.shape[0]
            else:
                embedding_losses += zero.clone()

            # embedding_losses-id
            if gt_foreground_mask.sum() > 0:
                num_chn = end_p - start_p
                pull_loss, push_loss = 0.0, 0.0
                for pred, gt, msk in zip(pred_emb_id, gt_line_id, gt_foreground_mask):
                    N = msk.sum()
                    if N == 0:
                        continue
                    gt = gt[msk]
                    gt_square = torch.zeros(N, N).to(outputs)
                    for i in range(N):
                        gt_row = gt.clone()
                        gt_row[gt == gt[i]] = 1
                        gt_row[gt != gt[i]] = 2
                        gt_square[i] = gt_row
                    msk = msk.expand(num_chn, -1, -1)
                    pred = pred[msk]
                    pred_row = pred.view(num_chn, 1, N).expand(num_chn, N, N)
                    pred_col = pred.view(num_chn, N, 1).expand(num_chn, N, N)
                    pred_sqrt = torch.norm(pred_col - pred_row, dim=0)
                    pred_dist = pred_sqrt[gt_square == 1] - self.pull_margin
                    pred_dist[pred_dist < 0] = 0
                    pull_loss += pred_dist.mean()
                    if gt_square.max() == 2:
                        pred_dist = self.push_margin - pred_sqrt[gt_square == 2]
                        pred_dist[pred_dist < 0] = 0
                        push_loss += pred_dist.mean()
                embedding_id_losses += (pull_loss + push_loss) / outputs.shape[0]
            else:
                embedding_id_losses += zero.clone()

            # cls loss
            if gt_foreground_mask.sum() > 0:
                pred_cls = torch.sigmoid(pred_cls)
                cls_loss = self.cls_loss(pred_cls, gt_line_cls)
                cls_loss = cls_loss[gt_foreground_mask.expand(-1, self.num_classes, -1, -1)]
                cls_loss = cls_loss.mean()

                cls_losses += cls_loss * cls_weight
            else:
                cls_losses += zero.clone()

        loss["confidence_loss"] = confidence_losses * self.confidence_weight
        loss["offset_loss"] = offset_losses * self.offset_weight
        loss["embedding_loss"] = embedding_losses * self.emb_weight
        loss["embedding_id_loss"] = embedding_id_losses * self.emb_id_weight
        loss["cls_loss"] = cls_losses * self.cls_weight

        loss['loss'] = loss["confidence_loss"] + loss["offset_loss"] \
                       + loss["embedding_loss"] + loss["embedding_id_loss"] \
                       + loss["cls_loss"]
        return loss


