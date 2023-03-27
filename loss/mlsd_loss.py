import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._func import focal_neg_loss_with_logits, weighted_bce_with_logits,\
    displacement_loss_func3, focal_neg_loss, len_and_angle_loss_func, class_loss_func, \
    displacement_loss_func, deccode_lines_TP# , weighted_bce_with_logits


class MLSDLoss(nn.Module):
    def __init__(self, cfg):
        super(MLSDLoss, self).__init__()

        self.cls_num = 21
        self.input_size = cfg.datasets.input_size

        self.with_SOL_loss = cfg.loss.with_sol_loss
        self.with_match_loss = cfg.loss.with_match_loss
        self.with_focal_loss = cfg.loss.with_focal_loss

        # 0: only in tp
        # 1: in tp and sol
        # 2: in tp, sol, junc
        # 3: in tp, sol, junc, line
        self.focal_loss_level = cfg.loss.focal_loss_level
        self.match_sap_thresh = cfg.loss.match_sap_thresh  # 5

        self.decode_score_thresh = cfg.decode.score_thresh  # 0.01
        self.decode_len_thresh = cfg.decode.len_thresh  # 10
        self.decode_top_k = cfg.decode.top_k  # 200

        self.loss_w_dict = {
            'tp_center_loss': 1.0,
            'tp_displacement_loss': 1.0,
            'tp_len_loss': 1.0,
            'tp_angle_loss': 1.0,
            'tp_match_loss': 1.0,
            'tp_centerness_loss': 1.0,  # current not support
            'tp_class_loss': 10.0,

            'sol_center_loss': 1.0,
            'sol_displacement_loss': 1.0,
            'sol_len_loss': 1.0,
            'sol_angle_loss': 1.0,
            'sol_match_loss': 1.0,
            'sol_centerness_loss': 1.0,  # current not support
            'sol_class_loss': 1.0,

            'line_seg_loss': 1.0,
            'junc_seg_loss': 1.0,
            'curve_seg_loss': 1.0,

        }

        if len(cfg.loss.loss_weight_dict_list) > 0:
            self.loss_w_dict.update(cfg.loss.loss_weight_dict_list[0])

        print("===> loss weight: ", self.loss_w_dict)

    def _m_gt_matched_n(self, p_lines, gt_lines, thresh):
        gt_lines = gt_lines.cuda()
        distance1 = torch.cdist(gt_lines[:, :2], p_lines[:, :2], p=2)
        distance2 = torch.cdist(gt_lines[:, 2:], p_lines[:, 2:], p=2)

        distance = distance1 + distance2
        near_inx = torch.argsort(distance, 1)[:, 0]  # most neared pred one

        matched_pred_lines = p_lines[near_inx]

        distance1 = F.pairwise_distance(gt_lines[:, :2], matched_pred_lines[:, :2], p=2)
        distance2 = F.pairwise_distance(gt_lines[:, 2:], matched_pred_lines[:, 2:], p=2)

        # print("distance1: ",distance1.shape)

        inx = torch.where((distance1 < thresh) & (distance2 < thresh))[0]
        return len(inx)

    def _m_match_loss_fn(self, p_lines, p_centers, p_scores, gt_lines, thresh):
        gt_lines = gt_lines.cuda()
        distance1 = torch.cdist(p_lines[:, :2], gt_lines[:, :2], p=2)  # 算两条线距离
        distance2 = torch.cdist(p_lines[:, 2:], gt_lines[:, 2:], p=2)

        distance = distance1 + distance2
        near_inx = torch.argsort(distance, 1)[:, 0]  # most neared one

        matched_gt_lines = gt_lines[near_inx]

        distance1 = F.pairwise_distance(matched_gt_lines[:, :2], p_lines[:, :2], p=2)
        distance2 = F.pairwise_distance(matched_gt_lines[:, 2:], p_lines[:, 2:], p=2)

        # print("distance1: ",distance1.shape)

        inx = torch.where((distance1 < thresh) & (distance2 < thresh))[0]

        # center_distance = F.pairwise_distance( (matched_gt_lines[:, :2] + matched_gt_lines[:, 2:])/2,
        #                                        (p_lines[:, :2] + p_lines[:, 2:])/2, p=2)
        # unmached_inx = torch.where( (distance1 > 3*thresh) &
        #                             (distance2 > 3*thresh) &
        #                              (center_distance > 2 * thresh) )[0]

        # print(inx)
        # print(unmached_inx)

        match_n = len(inx)
        # n_gt = gt_lines.shape[0]
        loss = 4 * thresh
        # loss = 0.0
        # match_ratio = inx[0].shape[0] / n_gt
        # match_ratio = np.clip(match_ratio, 0, 1.0)
        if match_n > 0:
            mathed_gt_lines = matched_gt_lines[inx]
            mathed_pred_lines = p_lines[inx]
            mathed_pred_centers = p_centers[inx]
            # mathed_pred_scores = p_scores[inx]

            endpoint_loss = F.l1_loss(mathed_pred_lines, mathed_gt_lines, reduction='mean')  # * 2

            gt_centers = (mathed_gt_lines[:, :2] + mathed_gt_lines[:, 2:]) / 2
            # print("gt_centers: ", gt_centers.shape)
            # print("mathed_pred_centers: ", mathed_pred_centers.shape)
            center_dis_loss = F.l1_loss(mathed_pred_centers, gt_centers, reduction='mean')

            # center_dis_loss = torch.where(center_dis_loss< 1.0, torch.zeros_like(center_dis_loss), center_dis_loss - 1.0)
            # endpoint_loss = torch.where(endpoint_loss< 1.0, torch.zeros_like(endpoint_loss), endpoint_loss - 1.0)
            # center_dis_loss = center_dis_loss.mean()
            # endpoint_loss = endpoint_loss.mean()
            # print("mean score: ", mathed_pred_scores.mean())

            # larger is better
            # mean_score = mathed_pred_scores.mean()
            # print(mean_score)
            loss = 1.0 * endpoint_loss + 1.0 * center_dis_loss  # - 1.0* mean_score

            # if len(unmached_inx) >0:
            #     unmathed_pred_scores = p_scores[unmached_inx]
            #     unmatch_mean_score = unmathed_pred_scores.mean()
            #     #print(unmatch_mean_score)
            #     # small is better
            #     loss = loss + 2.0 * unmatch_mean_score

            # print("endpoint_loss： ", endpoint_loss/ mathed_gt_lines.shape[0])
            # print("center_dis_loss： ", center_dis_loss/ mathed_gt_lines.shape[0])

            # loss = loss / mathed_pred_lines.shape[0]

        ## match ratio large is good
        # loss = loss - 5 * match_ratio
        # print("loss: ", loss)
        # print("match_n: ", match_n)
        return loss, match_n

    def matching_loss_func(self, pred_tp_mask, gt_line_512_tensor_list):
        match_loss_all = 0.0
        match_ratio_all = 0.0
        for pred, gt_line_512 in zip(pred_tp_mask, gt_line_512_tensor_list):
            gt_line_128 = gt_line_512 / 4
            n_gt = gt_line_128.shape[0]

            pred_center_ptss, pred_lines, pred_lines_swap, pred_scores = \
                deccode_lines_TP(pred.unsqueeze(0),
                                 score_thresh=self.decode_score_thresh,
                                 len_thresh=self.decode_len_thresh,
                                 topk_n=self.decode_top_k,
                                 ksize=3)
            n_pred = pred_center_ptss.shape[0]
            if n_pred == 0:
                match_loss_all += 4 * self.match_sap_thresh
                match_ratio_all += 0.0
                continue
            # print("pred_center_ptsssss: ",pred_center_ptss.shape)
            # print("gt_line_128: ", gt_line_128.shape)
            pred_lines_128 = 128 * pred_lines / (self.input_size / 2)
            pred_lines_128_swap = 128 * pred_lines_swap / (self.input_size / 2)
            pred_center_ptss_128 = 128 * pred_center_ptss / (self.input_size / 2)

            pred_lines_128 = torch.cat((pred_lines_128, pred_lines_128_swap), dim=0)
            pred_center_ptss_128 = torch.cat((pred_center_ptss_128, pred_center_ptss_128), dim=0)
            pred_scores = torch.cat((pred_scores, pred_scores), dim=0)

            mloss, match_n_pred = self._m_match_loss_fn(pred_lines_128,
                                                        pred_center_ptss_128,
                                                        pred_scores, gt_line_128, self.match_sap_thresh)

            match_n = self._m_gt_matched_n(pred_lines_128, gt_line_128, self.match_sap_thresh)
            match_ratio = match_n / n_gt

            match_loss_all += mloss
            match_ratio_all += match_ratio

        return match_loss_all / pred_tp_mask.shape[0], match_ratio_all / pred_tp_mask.shape[0]

    def tp_mask_loss(self, out, gt, gt_lines_tensor_512_list):
        tp_cls_ind_start = (self.cls_num + 7)
        tp_cls_ind_end = (self.cls_num + 7) * 2

        out_center = out[:, tp_cls_ind_start, :, :]
        gt_center = gt[:, tp_cls_ind_start, :, :]

        if self.loss_w_dict['tp_center_loss'] != 0:
            if self.with_focal_loss:
                center_loss = focal_neg_loss_with_logits(out_center, gt_center)
                # center_loss += weighted_bce_with_logits(out_center, gt_center, 1.0, 10.0)
            else:
                center_loss = weighted_bce_with_logits(out_center, gt_center, 1.0, 30.0)
        else:
            center_loss = torch.tensor(0)

        out_displacement = out[:, tp_cls_ind_start + 1:tp_cls_ind_start + 5, :, :]
        gt_displacement = gt[:, tp_cls_ind_start + 1:tp_cls_ind_start + 5, :, :]
        displacement_loss = displacement_loss_func3(out_displacement, gt_displacement, gt_center)

        len_loss, angle_loss = len_and_angle_loss_func(
            pred_len=out[:, tp_cls_ind_start + 5, :, :],
            pred_angle=out[:, tp_cls_ind_start + 6, :, :],
            gt_len=gt[:, tp_cls_ind_start + 5, :, :],
            gt_angle=gt[:, tp_cls_ind_start + 6, :, :]
        )
        match_loss, match_ratio = 0, 0
        # if self.with_match_loss:
        #     match_loss, match_ratio = self.matching_loss_func(out[:, 19:24],
        #                                                       gt_lines_tensor_512_list)

        out_cls = out[:, tp_cls_ind_start + 7:tp_cls_ind_end, :, :]
        gt_cls = gt[:, tp_cls_ind_start + 7:tp_cls_ind_end, :, :]
        cls_loss = class_loss_func(out_cls, gt_cls)

        return {
            'tp_center_loss': center_loss,
            'tp_displacement_loss': displacement_loss,
            'tp_len_loss': len_loss,
            'tp_angle_loss': angle_loss,
            'tp_match_loss': match_loss,
            'tp_match_ratio': match_ratio,  # not included in loss, only for log
            'tp_class_loss': cls_loss
        }

    def sol_mask_loss(self, out, gt, sol_lines_512_all_tensor_list):
        out_center = out[:, 0, :, :]
        gt_center = gt[:, 0, :, :]

        if self.with_focal_loss and self.focal_loss_level >= 1:
            center_loss = focal_neg_loss_with_logits(out_center, gt_center)
        else:
            center_loss = weighted_bce_with_logits(out_center, gt_center, 1.0, 30.0)

        out_displacement = out[:, 1:5, :, :]
        gt_displacement = gt[:, 1:5, :, :]
        displacement_loss = displacement_loss_func(out_displacement, gt_displacement, gt_center)

        len_loss, angle_loss = len_and_angle_loss_func(
            pred_len=out[:, 5, :, :],
            pred_angle=out[:, 6, :, :],
            gt_len=gt[:, 5, :, :],
            gt_angle=gt[:, 6, :, :]
        )
        match_loss, match_ratio = 0, 0
        # if self.with_match_loss:
        #     match_loss, match_ratio = self.matching_loss_func(out[:, 0:5],
        #                                                       sol_lines_512_all_tensor_list)

        sol_cls_ind = (self.cls_num + 7)
        out_cls = out[:, 7:sol_cls_ind, :, :]
        gt_cls = gt[:, 7:sol_cls_ind, :, :]
        cls_loss = class_loss_func(out_cls, gt_cls)

        return {
            'sol_center_loss': center_loss,
            'sol_displacement_loss': displacement_loss,
            'sol_len_loss': len_loss,
            'sol_angle_loss': angle_loss,
            'sol_match_loss': match_loss,
            'sol_class_loss': cls_loss
        }

    def line_and_juc_seg_loss(self, out, gt):
        line_seg_ind = (self.cls_num + 7) * 2 + 1
        out_line_seg = out[:, line_seg_ind, :, :]
        gt_line_seg = gt[:, line_seg_ind, :, :]
        if self.with_focal_loss and self.focal_loss_level >= 3:
            line_seg_loss = focal_neg_loss_with_logits(out_line_seg, gt_line_seg)
        else:
            line_seg_loss = weighted_bce_with_logits(out_line_seg, gt_line_seg, 1.0, 1.0)

        junc_seg_ind = (self.cls_num + 7) * 2
        out_junc_seg = out[:, junc_seg_ind, :, :]
        gt_junc_seg = gt[:, junc_seg_ind, :, :]
        if self.with_focal_loss and self.focal_loss_level >= 2:
            junc_seg_loss = focal_neg_loss_with_logits(out_junc_seg, gt_junc_seg)
        else:
            junc_seg_loss = weighted_bce_with_logits(out_junc_seg, gt_junc_seg, 1.0, 30.0)

        return line_seg_loss, junc_seg_loss

    def curve_loss(self, out, gt):
        curve_seg_ind = (self.cls_num + 7) * 2 + 2
        out_curve_seg = out[:, curve_seg_ind, :, :]
        gt_curve_seg = gt[:, curve_seg_ind, :, :]
        if self.with_focal_loss and self.focal_loss_level >= 3:
            curve_seg_loss = focal_neg_loss_with_logits(out_curve_seg, gt_curve_seg)
        else:
            curve_seg_loss = weighted_bce_with_logits(out_curve_seg, gt_curve_seg, 1.0, 1.0)

        return {'curve_seg_loss': curve_seg_loss}

    def forward(self, preds, gts,
                tp_gt_lines_512_list,
                sol_gt_lines_512_list):

        line_seg_loss, junc_seg_loss = self.line_and_juc_seg_loss(preds, gts)
        # curve_seg_loss_flag = self.curve_loss(preds, gts)['curve_seg_loss']
        # curve_seg_loss_dict = self.curve_loss(preds, gts)

        loss_dict = {
            'line_seg_loss': line_seg_loss,
            'junc_seg_loss': junc_seg_loss,
        }
        # if curve_seg_loss_flag !=0:
        #     loss_dict.update(curve_seg_loss_dict)

        if self.with_SOL_loss:
            sol_loss_dict = self.sol_mask_loss(preds, gts,
                                               sol_gt_lines_512_list)
            loss_dict.update(sol_loss_dict)

        tp_loss_dict = self.tp_mask_loss(preds, gts,
                                         tp_gt_lines_512_list)

        loss_dict.update(tp_loss_dict)

        loss = 0.0
        for k, v in loss_dict.items():
            if not self.with_SOL_loss and 'sol_' in k:
                continue
            if k in self.loss_w_dict.keys():
                v = v * self.loss_w_dict[k]
                loss_dict[k] = v
                loss += v
        loss_dict['loss'] = loss

        if self.with_SOL_loss:
            loss_dict['center_loss'] = loss_dict['sol_center_loss'] + loss_dict['tp_center_loss']
            loss_dict['displacement_loss'] = loss_dict['sol_displacement_loss'] + loss_dict['tp_displacement_loss']
            loss_dict['match_loss'] = loss_dict['tp_match_loss'] + loss_dict['sol_match_loss']
            loss_dict['match_ratio'] = loss_dict['tp_match_ratio']
            loss_dict['class_loss'] = loss_dict['tp_class_loss'] + loss_dict['sol_class_loss']
        else:
            loss_dict['center_loss'] = loss_dict['tp_center_loss']
            loss_dict['displacement_loss'] = loss_dict['tp_displacement_loss']
            loss_dict['match_loss'] = loss_dict['tp_match_loss']
            loss_dict['match_ratio'] = loss_dict['tp_match_ratio']
            loss_dict['class_loss'] = loss_dict['tp_class_loss']

        # print("center_loss", loss_dict['center_loss'])
        # print("displacement_loss", loss_dict['displacement_loss'])
        # print("match_loss", loss_dict['match_loss'])
        # print("match_ratio", loss_dict['match_ratio'])
        # print("class_loss", loss_dict['class_loss'])

        return loss_dict