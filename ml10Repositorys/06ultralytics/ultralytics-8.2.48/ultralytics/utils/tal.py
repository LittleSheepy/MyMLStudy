# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn

from .checks import check_version
from .metrics import bbox_iou, probiou
from .ops import xywhr2xyxyxyxy
import matplotlib.pyplot as plt   # 直接选择落在GT框中的锚点 torch.Size([1, 2, 8400])

TORCH_1_10 = check_version(torch.__version__, "1.10.0")
import matplotlib.pyplot as plt
import numpy as np
def normalize_to_255(arr):
    arr_min = arr.min()
    arr_max = arr.max()
    normalized_arr = (arr - arr_min) / (arr_max - arr_min + 1e-8) * 255
    return normalized_arr

def show_one(img_data, title=""):
    data_min = img_data.min()
    data_max = img_data.max()
    plt.title(title +",".join([" ", str(data_min), str(data_max)]))
    # img_data = normalize_to_255(img_data)
    plt.imshow(img_data, cmap="gray")
    plt.show()


def show8400(data8400, title=""):
    data6400 = data8400[:6400].reshape((80, 80)).cpu().numpy()
    show_one(data6400, title + "80*80")

    data1600 = data8400[6400:8000].reshape((40, 40)).cpu().numpy()
    show_one(data1600, title + "40*40")

    data400 = data8400[8000:].reshape((20, 20)).cpu().numpy()
    show_one(data400, title + "20*20")


class TaskAlignedAssigner(nn.Module):
    """
    A task-aligned assigner for object detection.

    This class assigns ground-truth (gt) objects to anchors based on the task-aligned metric, which combines both
    classification and localization information.

    Attributes:
        topk (int): The number of top candidates to consider.
        num_classes (int): The number of object classes.
        alpha (float): The alpha parameter for the classification component of the task-aligned metric.
        beta (float): The beta parameter for the localization component of the task-aligned metric.
        eps (float): A small value to prevent division by zero.
    """

    def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=6.0, eps=1e-9):
        """Initialize a TaskAlignedAssigner object with customizable hyperparameters."""
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        Compute the task-aligned assignment. Reference code is available at
        https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py.
        pred=pd
        Args:
            pd_scores (Tensor): shape(bs, num_total_anchors, num_classes)   torch.Size([1, 8400, 80])
            pd_bboxes (Tensor): shape(bs, num_total_anchors, 4)             torch.Size([1, 8400, 4])
            anc_points (Tensor): shape(num_total_anchors, 2)                torch.Size([8400, 2])
            gt_labels (Tensor): shape(bs, n_max_boxes, 1)                   torch.Size([1, 2, 1])
            gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)                   torch.Size([1, 2, 4])
            mask_gt (Tensor): shape(bs, n_max_boxes, 1)                     torch.Size([1, 2, 1])

        Returns:
            target_labels (Tensor): shape(bs, num_total_anchors)
            target_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            target_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            fg_mask (Tensor): shape(bs, num_total_anchors)
            target_gt_idx (Tensor): shape(bs, num_total_anchors)
        """
        self.bs = pd_scores.shape[0]            # 1
        self.n_max_boxes = gt_bboxes.shape[1]   # 2

        if self.n_max_boxes == 0:
            device = gt_bboxes.device
            return (
                torch.full_like(pd_scores[..., 0], self.bg_idx).to(device),
                torch.zeros_like(pd_bboxes).to(device),
                torch.zeros_like(pd_scores).to(device),
                torch.zeros_like(pd_scores[..., 0]).to(device),
                torch.zeros_like(pd_scores[..., 0]).to(device),
            )
        # mask_pos torch.Size([1, 1, 8400])
        # align_metric  对齐度       torch.Size([1, 1, 8400])
        # overlaps      IOU         torch.Size([1, 1, 8400])
        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )

        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)
        show8400(target_gt_idx[0, :], "target_gt_idx")
        show8400(fg_mask[0, :], "fg_mask")
        show8400(mask_pos[0, 0, :], "mask_pos")
        # Assigned target 指派目标
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        # Normalize
        align_metric *= mask_pos    # 根据有效anchor的标记来保留有效的对齐得分  [1, 1, 8400]
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)  # b, max_num_obj # 获取所有anchor与每个gt对应的最大对齐得分 [1, 1, 1]
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)  # b, max_num_obj # 获取所有anchor与每个gt的最大iou [1, 1, 1]
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        """Get in_gts mask, (b, max_num_obj, h*w)."""
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes)

        show8400(mask_in_gts[0, 0, :], "mask_in_gts")

        # Get anchor_align metric, (b, max_num_obj, h*w)  计算预测边界框和真实边界框的对齐度指标。 torch.Size([1, 1, 8400])
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        # Get topk_metric mask, (b, max_num_obj, h*w)
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())

        show8400(mask_topk[0, 0, :], "mask_topk")

        # Merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_topk * mask_in_gts * mask_gt

        show8400(mask_pos[0, 0, :], "mask_pos")

        return mask_pos, align_metric, overlaps
    # pd_scores torch.Size([1, 8400, 80])
    # pd_bboxes torch.Size([1, 8400, 4])
    # gt_labels torch.Size([1, 8, 1])
    # gt_bboxes torch.Size([1, 8, 4])
    # mask_gt torch.Size([1, 8, 8400])
    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """Compute alignment metric given predicted and ground truth bounding boxes. 计算给定预测边界框和真实边界框的对齐度量"""
        na = pd_bboxes.shape[-2]        # 8400
        mask_gt = mask_gt.bool()  # b, max_num_obj, h*w torch.Size([1, 8, 8400])
        # torch.Size([1, 1, 8400])
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        # torch.Size([1, 1, 8400])
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)
        # torch.Size([2, 1, 1])
        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj   torch.Size([2, 1, 2])
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)  # b, max_num_obj
        ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj
        # Get the scores of each grid for each gt cls
        # 把初选anchor对应的正确类别的 cls_score 记录下来
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]  # b, max_num_obj, h*w torch.Size([1, 8, 8400])
        # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)  torch.Size([3590, 4])
        pd_bboxes_expand = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)  # torch.Size([1, 1, 8400, 4])
        pd_boxes = pd_bboxes_expand[mask_gt]     # torch.Size([2046, 4])
        gt_bboxes_expand = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)    # torch.Size([1, 1, 8400, 4])
        gt_boxes = gt_bboxes_expand[mask_gt]                   # torch.Size([2046, 4])
        iou = self.iou_calculation(gt_boxes, pd_boxes)  # torch.Size([2046])
        overlaps[mask_gt] = iou

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        show8400(align_metric[0, 0, :], "align_metric")
        show8400(overlaps[0, 0, :], "overlaps")
        return align_metric, overlaps       # 对齐度 和 IOU

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """IoU calculation for horizontal bounding boxes."""
        return bbox_iou(gt_bboxes, pd_bboxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0)

    def select_topk_candidates(self, metrics, largest=True, topk_mask=None):
        """
        Select the top-k candidates based on the given metrics.

        Args:
            metrics (Tensor): A tensor of shape (b, max_num_obj, h*w), where b is the batch size,
                              max_num_obj is the maximum number of objects, and h*w represents the
                              total number of anchor points.
            largest (bool): If True, select the largest values; otherwise, select the smallest values.
            topk_mask (Tensor): An optional boolean tensor of shape (b, max_num_obj, topk), where
                                topk is the number of top candidates to consider. If not provided,
                                the top-k values are automatically computed based on the given metrics.

        Returns:
            (Tensor): A tensor of shape (b, max_num_obj, h*w) containing the selected top-k candidates.
        """

        # (b, max_num_obj, topk)
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=largest)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)
        # (b, max_num_obj, topk)
        topk_idxs.masked_fill_(~topk_mask, 0)

        # (b, max_num_obj, topk, h*w) -> (b, max_num_obj, h*w)
        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device)
        ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8, device=topk_idxs.device)
        for k in range(self.topk):
            # Expand topk_idxs for each value of k and add 1 at the specified positions
            count_tensor.scatter_add_(-1, topk_idxs[:, :, k : k + 1], ones)
        # count_tensor.scatter_add_(-1, topk_idxs, torch.ones_like(topk_idxs, dtype=torch.int8, device=topk_idxs.device))
        # Filter invalid bboxes
        count_tensor.masked_fill_(count_tensor > 1, 0)

        return count_tensor.to(metrics.dtype)

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """
        Compute target labels, target bounding boxes, and target scores for the positive anchor points.
        计算正锚点的目标标签、目标边界框和目标分数。
        Args:
            gt_labels (Tensor): Ground truth labels of shape (b, max_num_obj, 1), where b is the
                                batch size and max_num_obj is the maximum number of objects.
            gt_bboxes (Tensor): Ground truth bounding boxes of shape (b, max_num_obj, 4).
            target_gt_idx (Tensor): Indices of the assigned ground truth objects for positive
                                    anchor points, with shape (b, h*w), where h*w is the total
                                    number of anchor points.
            fg_mask (Tensor): A boolean tensor of shape (b, h*w) indicating the positive
                              (foreground) anchor points.

        Returns:
            (Tuple[Tensor, Tensor, Tensor]): A tuple containing the following tensors:
                - target_labels (Tensor): Shape (b, h*w), containing the target labels for
                                          positive anchor points.
                - target_bboxes (Tensor): Shape (b, h*w, 4), containing the target bounding boxes
                                          for positive anchor points.
                - target_scores (Tensor): Shape (b, h*w, num_classes), containing the target scores
                                          for positive anchor points, where num_classes is the number
                                          of object classes.
        """

        # Assigned target labels, (b, 1)
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]    # torch.Size([1, 1])
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes  # (b, h*w) torch.Size([1, 8400])
        # 8400中的格子对应的类别是哪一个
        target_labels = gt_labels.long().flatten()[target_gt_idx]  # (b, h*w) torch.Size([1, 8400])
        # Assigned target boxes, (b, max_num_obj, 4) -> (b, h*w, 4)
        gt_bboxes_view = gt_bboxes.view(-1, gt_bboxes.shape[-1])    # torch.Size([1, 4])
        target_bboxes = gt_bboxes_view[target_gt_idx]  # torch.Size([1, 8400, 4])   每个格子对应的真实框，没有就用第一个替代，后面会使用mask去除
        # Assigned target scores 分配的目标分数
        target_labels.clamp_(0)

        # 10x faster than F.one_hot() torch.Size([1, 8400, 80])
        target_scores = torch.zeros(
            (target_labels.shape[0], target_labels.shape[1], self.num_classes),
            dtype=torch.int64,
            device=target_labels.device,
        )  # (b, h*w, 80)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)
        show8400(target_scores[0, :, 22], "target_scores")
        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)  # (b, h*w, 80) torch.Size([1, 8400, 80])
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        return target_labels, target_bboxes, target_scores

    @staticmethod
    def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9):
        """
        Select the positive anchor center in gt.

        Args:
            xy_centers (Tensor): shape(h*w, 2)              # torch.Size([8400, 2])
            gt_bboxes (Tensor): shape(b, n_boxes, 4)        # torch.Size([1, 2, 4])

        Returns:
            (Tensor): shape(b, n_boxes, h*w)
        """
        n_anchors = xy_centers.shape[0]     # 8400
        bs, n_boxes, _ = gt_bboxes.shape    # 1    2
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom
        bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors, -1) # torch.Size([1, 2, 8400, 4])
        # return (bbox_deltas.min(3)[0] > eps).to(gt_bboxes.dtype)
        return bbox_deltas.amin(3).gt_(eps)

    @staticmethod
    def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
        """
        If an anchor box is assigned to multiple gts, the one with the highest IoU will be selected.
        如果一个锚框被分配给多个gts，那么IoU最高的那个将被选择。
        Args:
            mask_pos (Tensor): shape(b, n_max_boxes, h*w)
            overlaps (Tensor): shape(b, n_max_boxes, h*w)

        Returns:
            target_gt_idx (Tensor): shape(b, h*w)
            fg_mask (Tensor): shape(b, h*w)
            mask_pos (Tensor): shape(b, n_max_boxes, h*w)
        """
        # (b, n_max_boxes, h*w) -> (b, h*w)
        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)  # (b, n_max_boxes, h*w)
            max_overlaps_idx = overlaps.argmax(1)  # (b, h*w)

            is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)

            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()  # (b, n_max_boxes, h*w)
            fg_mask = mask_pos.sum(-2)
        # Find each grid serve which gt(index)
        target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)
        return target_gt_idx, fg_mask, mask_pos


class RotatedTaskAlignedAssigner(TaskAlignedAssigner):
    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """IoU calculation for rotated bounding boxes."""
        return probiou(gt_bboxes, pd_bboxes).squeeze(-1).clamp_(0)

    @staticmethod
    def select_candidates_in_gts(xy_centers, gt_bboxes):
        """
        Select the positive anchor center in gt for rotated bounding boxes.

        Args:
            xy_centers (Tensor): shape(h*w, 2)
            gt_bboxes (Tensor): shape(b, n_boxes, 5)

        Returns:
            (Tensor): shape(b, n_boxes, h*w)
        """
        # (b, n_boxes, 5) --> (b, n_boxes, 4, 2)
        corners = xywhr2xyxyxyxy(gt_bboxes)
        # (b, n_boxes, 1, 2)
        a, b, _, d = corners.split(1, dim=-2)
        ab = b - a
        ad = d - a

        # (b, n_boxes, h*w, 2)
        ap = xy_centers - a
        norm_ab = (ab * ab).sum(dim=-1)
        norm_ad = (ad * ad).sum(dim=-1)
        ap_dot_ab = (ap * ab).sum(dim=-1)
        ap_dot_ad = (ap * ad).sum(dim=-1)
        return (ap_dot_ab >= 0) & (ap_dot_ab <= norm_ab) & (ap_dot_ad >= 0) & (ap_dot_ad <= norm_ad)  # is_in_box


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_10 else torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)

# 将每一个anchor point和预测的回归参数通过dist2bbox做解码，生成anchor box与gt计算iou
def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


def bbox2dist(anchor_points, bbox, reg_max):
    """Transform bbox(xyxy) to dist(ltrb)."""
    x1y1, x2y2 = bbox.chunk(2, -1)  # 拆分
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp_(0, reg_max - 0.01)  # dist (lt, rb)


def dist2rbox(pred_dist, pred_angle, anchor_points, dim=-1):
    """
    Decode predicted object bounding box coordinates from anchor points and distribution.

    Args:
        pred_dist (torch.Tensor): Predicted rotated distance, (bs, h*w, 4).
        pred_angle (torch.Tensor): Predicted angle, (bs, h*w, 1).
        anchor_points (torch.Tensor): Anchor points, (h*w, 2).
    Returns:
        (torch.Tensor): Predicted rotated bounding boxes, (bs, h*w, 4).
    """
    lt, rb = pred_dist.split(2, dim=dim)
    cos, sin = torch.cos(pred_angle), torch.sin(pred_angle)
    # (bs, h*w, 1)
    xf, yf = ((rb - lt) / 2).split(1, dim=dim)
    x, y = xf * cos - yf * sin, xf * sin + yf * cos
    xy = torch.cat([x, y], dim=dim) + anchor_points
    return torch.cat([xy, lt + rb], dim=dim)
