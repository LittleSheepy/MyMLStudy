#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import math
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolox.utils import bboxes_iou, meshgrid

from .losses import IOUloss
from .network_blocks import BaseConv, DWConv


class YOLOXHead(nn.Module):
    def __init__(
        self,
        num_classes,
        width=1.0,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        act="silu",
        depthwise=False,
    ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): whether apply depthwise conv in conv branch. Defalut value: False.
        """
        super().__init__()

        self.n_anchors = 1
        self.num_classes = num_classes
        self.decode_in_inference = True  # for deploy, set to False

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()
        Conv = DWConv if depthwise else BaseConv

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(256 * width),
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )
            self.cls_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * 1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)

    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
    # [torch.Size([4, 128, 128, 128]), torch.Size([4, 256, 64, 64]), torch.Size([4, 512, 32, 32])]  torch.Size([4, 120, 5])   torch.Size([4, 3, 1024, 1024])
    def forward(self, xin, labels=None, imgs=None):
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.strides, xin)
        ):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)        # torch.Size([4, 8, 128, 128])

            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)        # torch.Size([4, 4, 128, 128])
            obj_output = self.obj_preds[k](reg_feat)        # torch.Size([4, 1, 128, 128])

            if self.training:
                output = torch.cat([reg_output, obj_output, cls_output], 1)     # torch.Size([4, 13=n_class8+box4+obj1, 128, 128])
                output, grid = self.get_output_and_grid(
                    output, k, stride_this_level, xin[0].type()
                )   # output: torch.Size([4, 16384=128*128, 13])   grid: torch.Size([1, 16384, 2])
                x_shifts.append(grid[:, :, 0])      # [torch.Size([1, 16384])]
                y_shifts.append(grid[:, :, 1])      # [torch.Size([1, 16384])]
                expanded_strides.append(
                    torch.zeros(1, grid.shape[1])
                    .fill_(stride_this_level)
                    .type_as(xin[0])
                )   # [torch.Size([1, 16384])]      [[8] [16] [32]]
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.view(
                        batch_size, self.n_anchors, 4, hsize, wsize
                    )
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                        batch_size, -1, 4
                    )
                    origin_preds.append(reg_output.clone())

            else:
                output = torch.cat(
                    [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
                )

            outputs.append(output)      # [torch.Size([4, 16384=128*128, 13=n_class8+box4+obj1]), torch.Size([4, 4096=64*64, 13]), torch.Size([4, 1024=32*32, 13])]

        if self.training:
            return self.get_losses(
                imgs,
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                torch.cat(outputs, 1),
                origin_preds,
                dtype=xin[0].dtype,
            )
        else:
            self.hw = [x.shape[-2:] for x in outputs]
            # [batch, n_anchors_all, 85]
            outputs = torch.cat(
                [x.flatten(start_dim=2) for x in outputs], dim=2
            ).permute(0, 2, 1)
            if self.decode_in_inference:
                return self.decode_outputs(outputs, dtype=xin[0].type())
            else:
                return outputs

    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]

        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
            self.grids[k] = grid

        output = output.view(batch_size, self.n_anchors, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(
            batch_size, self.n_anchors * hsize * wsize, -1
        )
        grid = grid.view(1, -1, 2)
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        return output, grid

    def decode_outputs(self, outputs, dtype):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)

        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        return outputs

    def get_losses(
        self,
        imgs,
        x_shifts,
        y_shifts,
        expanded_strides,       # [torch.Size([1, 16384]) torch.Size([1, 4096]) torch.Size([1, 1024])]
        labels,                 # torch.Size([4, 120, 5])
        outputs,
        origin_preds,
        dtype,
    ):
        bbox_preds = outputs[:, :, :4]              # [batch, n_anchors_all, 4]     torch.Size([4, 21504, 4])
        obj_preds = outputs[:, :, 4].unsqueeze(-1)  # [batch, n_anchors_all, 1]     torch.Size([4, 21504, 1])
        cls_preds = outputs[:, :, 5:]               # [batch, n_anchors_all, n_cls] torch.Size([4, 21504, 8])

        # calculate targets
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects    tensor([ 2,  3, 10,  9], device='cuda:0')

        total_num_anchors = outputs.shape[1]                        # 21504
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]     torch.Size([1, 21504])
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]     torch.Size([1, 21504])
        expanded_strides = torch.cat(expanded_strides, 1)   # torch.Size([1, 21504])
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)

        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.0
        num_gts = 0.0

        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])             # 2
            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]       # torch.Size([2, 4])
                gt_classes = labels[batch_idx, :num_gt, 0]                  # torch.Size(2)
                bboxes_preds_per_image = bbox_preds[batch_idx]              # torch.Size([21504, 4])
                # [0 0] (21504,)  [0.0143, 0.4808] [1, 0] 2
                try:
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                    )
                except RuntimeError as e:
                    # TODO: the string might change, consider a better way
                    if "CUDA out of memory. " not in str(e):
                        raise  # RuntimeError might not caused by CUDA OOM

                    logger.error(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    torch.cuda.empty_cache()
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                        "cpu",
                    )

                torch.cuda.empty_cache()
                num_fg += num_fg_img        # 2

                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                ) * pred_ious_this_matching.unsqueeze(-1)                   # torch.Size([2, 8])
                obj_target = fg_mask.unsqueeze(-1)                          # torch.Size([21504, 1])
                reg_target = gt_bboxes_per_image[matched_gt_inds]           # torch.Size([2, 4])
                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )

            cls_targets.append(cls_target)              # [torch.Size([2, 8]), torch.Size([1, 8]) torch.Size([3, 8]) torch.Size([9, 8])]
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)                    # [21504 21504 21504 21504 ]
            if self.use_l1:
                l1_targets.append(l1_target)

        cls_targets = torch.cat(cls_targets, 0)         # torch.Size([15, 8])
        reg_targets = torch.cat(reg_targets, 0)         # torch.Size([15, 4])
        obj_targets = torch.cat(obj_targets, 0)         # torch.Size([86016, 1])
        fg_masks = torch.cat(fg_masks, 0)               # (86016,)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        num_fg = max(num_fg, 1)     # 15
        loss_iou = (
            self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)
        ).sum() / num_fg
        loss_obj = (
            self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)
        ).sum() / num_fg
        loss_cls = (
            self.bcewithlog_loss(
                cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
            )
        ).sum() / num_fg
        if self.use_l1:
            loss_l1 = (
                self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)
            ).sum() / num_fg
        else:
            loss_l1 = 0.0

        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1

        return (
            loss,
            reg_weight * loss_iou,
            loss_obj,
            loss_cls,
            loss_l1,
            num_fg / max(num_gts, 1),
        )

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target

    @torch.no_grad()
    def get_assignments(
        self,
        batch_idx,
        num_gt,
        total_num_anchors,
        gt_bboxes_per_image,
        gt_classes,
        bboxes_preds_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        cls_preds,      # torch.Size([4, 21504, 8])
        bbox_preds,
        obj_preds,
        labels,
        imgs,
        mode="gpu",
    ):

        if mode == "cpu":
            print("------------CPU Mode for This Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()         # torch.Size([2, 4]) [[920.5000, 242.6250,   8.9141,  10.3203], [902.0000,  93.8125,   4.6914,   9.3828]]
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            gt_bboxes_per_image,        # torch.Size([2, 4])
            expanded_strides,           # torch.Size([1, 21504])
            x_shifts,                   # torch.Size([1, 21504=32*32+64*64+128*128])
            y_shifts,                   # torch.Size([1, 21504])
            total_num_anchors,          # 21504
            num_gt,                     # 2
        )   # (21504,) sum=150  (21504,) sum=2

        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]        # torch.Size([150, 4])  [[880 72.2 7.8 7.4]]
        cls_preds_ = cls_preds[batch_idx][fg_mask]                      # torch.Size([150, 8])
        obj_preds_ = obj_preds[batch_idx][fg_mask]                      # torch.Size([150, 1])
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]           # 150

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()
        # 计算每个gt和每个pre的iou
        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)     # torch.Size([2, 150])

        gt_cls_per_image = (
            F.one_hot(gt_classes.to(torch.int64), self.num_classes)
            .float()
            .unsqueeze(1)
            .repeat(1, num_in_boxes_anchor, 1)
        )       # torch.Size([2, 150, 8])
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)     # torch.Size([2, 150])

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        with torch.cuda.amp.autocast(enabled=False):
            cls_preds_ = (
                cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                * obj_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )       # torch.Size([2, 150, 8])
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.sqrt_(), gt_cls_per_image, reduction="none"
            ).sum(-1)       # torch.Size([2, 150])
        del cls_preds_
        # 计算损失 所有的 此刻还没有做gt和pred的匹配
        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + 100000.0 * (~is_in_boxes_and_center)
        )
        # 2 [0 0]   [0.0143, 0.4808]    [1, 0]
        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()
        # [0 0] (21504,)  [0.0143, 0.4808] [1, 0] 2
        return (
            gt_matched_classes,         # [0 0]
            fg_mask,                    # (21504,)
            pred_ious_this_matching,    # [0.0143, 0.4808]
            matched_gt_inds,            # [1, 0]
            num_fg,                     # 2
        )

    def get_in_boxes_info(
        self,
        gt_bboxes_per_image,        # torch.Size([2, 4])
        expanded_strides,           # torch.Size([1, 21504])
        x_shifts,                   # torch.Size([1, 21504=32*32+64*64+128*128])
        y_shifts,                   # torch.Size([1, 21504])
        total_num_anchors,          # 21504
        num_gt,                     # 2
    ):
        expanded_strides_per_image = expanded_strides[0]                # (21504,)  [8 8 8 ]
        x_shifts_per_image = x_shifts[0] * expanded_strides_per_image   # (21504,)  [0 8 16 ]
        y_shifts_per_image = y_shifts[0] * expanded_strides_per_image   # (21504,)  [0 8 16 ]
        x_centers_per_image = (
            (x_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )  # [n_anchor] -> [n_gt, n_anchor] torch.Size([2, 21504])  # (2, 21504)  [[4 12 20 ]]
        y_centers_per_image = (
            (y_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )   # (2, 21504) [4 4 4...1008]

        gt_bboxes_per_image_l = (
            (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )       # torch.Size([2, 21504])  gt_bboxes_per_image_l.T = [[916, 899]]
        gt_bboxes_per_image_r = (
            (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )       # torch.Size([2, 21504])  gt_bboxes_per_image_l.T = [[925 904]]
        gt_bboxes_per_image_t = (
            (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )       # torch.Size([2, 21504])  gt_bboxes_per_image_l.T = [[237, 89]]
        gt_bboxes_per_image_b = (
            (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )       # torch.Size([2, 21504])  gt_bboxes_per_image_l.T = [[247, 98]]
        # 得到锚框中心点与gt框的差值 用来判断这个中心点是否在gt中
        b_l = x_centers_per_image - gt_bboxes_per_image_l       # torch.Size([2, 21504])    [[-912 -904 -896]]
        b_r = gt_bboxes_per_image_r - x_centers_per_image       # torch.Size([2, 21504])    [[921 913 905]]
        b_t = y_centers_per_image - gt_bboxes_per_image_t       # torch.Size([2, 21504])    [[-233.5 -233.5]]
        b_b = gt_bboxes_per_image_b - y_centers_per_image       # torch.Size([2, 21504])    [[243.75 243.75]]
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)      # torch.Size([2, 21504, 4]) [[-912 -233.5 921 243.75]]
        # 总的来说，这个代码块用于过滤掉不在任何真实框内的锚框，这可以提高目标检测模型的效率。
        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0      # torch.Size([2, 21504]) is_in_boxes是一个布尔张量，指示每个锚框是否在任何真实框内。
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0            # (21504, ) is_in_boxes_all是一个布尔张量，指示任何锚框中心点是否在任何真实框内。
        # in fixed center

        center_radius = 2.5
        # gt 放大
        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)     # torch.Size([2, 21504]) [[900.5 900.5 ]]
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)

        c_l = x_centers_per_image - gt_bboxes_per_image_l           # torch.Size([2, 21504]) [[-896.5 -888.5]]
        c_r = gt_bboxes_per_image_r - x_centers_per_image           #
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # in boxes and in centers
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all    # sum 150

        is_in_boxes_and_center = (
            is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        )   # sum = 2
        return is_in_boxes_anchor, is_in_boxes_and_center
    # cost torch.Size([2, 150]),  pair_wise_ious torch.Size([2, 150]) gt_classes [0 0]   fg_mask:(21504,)
    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        # Dynamic K
        # ---------------------------------------------------------------
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)     # torch.Size([2, 150])

        ious_in_boxes_matrix = pair_wise_ious                           # torch.Size([2, 150])
        n_candidate_k = min(10, ious_in_boxes_matrix.size(1))           # 10
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)       # torch.Size([2, 10])
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)                     # tensor([1, 1], device='cuda:0', dtype=torch.int32)
        dynamic_ks = dynamic_ks.tolist()        # [1 1]
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx], largest=False
            )       # tensor([37], device='cuda:0')
            matching_matrix[gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)         # (150,)
        if (anchor_matching_gt > 1).sum() > 0:      # 这里检查是否有anchor boxes被多个ground truth box匹配。如果有，就选择与这些anchor boxes匹配的代价最小的ground truth box，并将其他ground truth box的匹配标记为0。
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1
        fg_mask_inboxes = matching_matrix.sum(0) > 0        # (150, )
        num_fg = fg_mask_inboxes.sum().item()               # 2

        fg_mask[fg_mask.clone()] = fg_mask_inboxes          # (21504,) sum = 2
        # 判断和哪个gt匹配
        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)     # tensor([1, 0], device='cuda:0')
        gt_matched_classes = gt_classes[matched_gt_inds]                    # [0, 0]
        # (matching_matrix=torch.Size([2, 150]) * pair_wise_ious=torch.Size([2, 150])) shape = torch.Size([2, 150])
        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]       # tensor([0.0143, 0.4808], device='cuda:0')
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds
        # 2 [0 0]   [0.0143, 0.4808]    [1, 0]
