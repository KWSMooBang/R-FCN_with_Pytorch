import numpy as np
import torch

def transform_bbox(anchors, bboxes):
    """
    transform bbox format
    (x_min, y_min, x_max, y_max) -> (t_x, t_y, t_w, t_h)
    """
    if bboxes.dim() == 2:
        anchors_w = anchors[:, 2] - anchors[:, 0] + 1.0
        anchors_h = anchors[:, 3] - anchors[:, 1] + 1.0
        anchors_ctr_x = anchors[:, 0] + 0.5 * anchors_w
        anchors_ctr_y = anchors[:, 1] + 0.5 * anchors_h

        bboxes_w = bboxes[:, 2] - bboxes[:, 0] + 1.0
        bboxes_h = bboxes[:, 3] - bboxes[:, 1] + 1.0
        bboxes_ctr_x = bboxes[:, 0] + 0.5 * bboxes_w
        bboxes_ctr_y = bboxes[:, 1] + 0.5 * bboxes_h

        targets_x = (bboxes_ctr_x - anchors_ctr_x) / anchors_w
        targets_y = (bboxes_ctr_y - anchors_ctr_y) / anchors_h
        targets_w = torch.log(bboxes_w / anchors_w)
        targets_h = torch.log(bboxes_h / anchors_h)

        targets = torch.stack(
            (targets_x, targets_y, targets_w, targets_h), 1
        )

        return targets
    else:
        anchors_w = anchors[:, :, 2] - anchors[:, :, 0] + 1.0
        anchors_h = anchors[:, :, 3] - anchors[:, :, 1] + 1.0
        anchors_ctr_x = anchors[:, :, 0] + 0.5 * anchors_w
        anchors_ctr_y = anchors[:, :, 1] + 0.5 * anchors_h

        bboxes_w = bboxes[:, :, 2] - bboxes[:, :, 0] + 1.0
        bboxes_h = bboxes[:, :, 3] - bboxes[:, :, 1] + 1.0
        bboxes_ctr_x = bboxes[:, :, 0] + 0.5 * bboxes_w
        bboxes_ctr_y = bboxes[:, :, 1] + 0.5 * bboxes_h

        targets_x = (bboxes_ctr_x - anchors_ctr_x) / anchors_w
        targets_y = (bboxes_ctr_y - anchors_ctr_y) / anchors_h
        targets_w = torch.log(bboxes_w / anchors_w)
        targets_h = torch.log(bboxes_h / anchors_h)

        targets = torch.stack(
            (targets_x, targets_y, targets_w, targets_h), 2
        )

        return targets

def inverse_transform_bbox(anchors, targets):
    anchors_w = anchors[:, :, 2] - anchors[:, :, 0] + 1.0
    anchors_h = anchors[:, :, 3] - anchors[:, :, 1] + 1.0
    anchors_ctr_x = anchors[:, :, 0] + 0.5 * anchors_w
    anchors_ctr_y = anchors[:, :, 1] + 0.5 * anchors_h

    targets_x = targets[:, :, 0]
    targets_y = targets[:, :, 1]
    targets_w = targets[:, :, 2]
    targets_h = targets[:, :, 3]

    bboxes_ctr_x = targets_x * anchors_w + anchors_ctr_x
    bboxes_ctr_y = targets_y * anchors_h + anchors_ctr_y
    bboxes_w = torch.exp(targets_w) * anchors_w
    bboxes_h = torch.exp(targets_h) * anchors_h

    bboxes_xmin = bboxes_ctr_x - 0.5 * bboxes_w
    bboxes_ymin = bboxes_ctr_y - 0.5 * bboxes_h
    bboxes_xmax = bboxes_ctr_x + 0.5 * bboxes_w
    bboxes_ymax = bboxes_ctr_y + 0.5 * bboxes_h

    bboxes = torch.stack(
        (bboxes_xmin, bboxes_ymin, bboxes_xmax, bboxes_ymax), 2
    )

    return bboxes

def iou_bbox(anchors, gt_bboxes):
    """
    calculate iou between anchors and ground truth bboxes
    Args:
        anchors: (N, 4)
        gt_bboxes: (K, 4)

        overlaps: (N, K)
    """
    N = anchors.size(0)
    K = gt_bboxes.size(0)

    gt_bboxes_area = ((gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1) * (gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1)).view(1, K)
    anchors_area = ((anchors[:, 2] - anchors[:, 0] + 1) * (anchors[:, 3] - anchors[:, 1] + 1)).view(N, 1)
    
    anchors_expanded = anchors.view(N, 1, 4).expand(N, K, 4)
    gt_bboxes_expanded = gt_bboxes.view(1, K, 4).expand(N, K, 4)

    intersects_w = (torch.min(anchors_expanded[:, :, 2], gt_bboxes_expanded[:, :, 2]) - 
                torch.max(anchors_expanded[:, :, 0], anchors_expanded[:, :, 0]) + 1)
    intersects_h = (torch.min(anchors_expanded[:, :, 3], gt_bboxes_expanded[:, :, 2]) -
                torch.max(anchors_expanded[:, :, 0], anchors_expanded[:, :, 0]) + 1)
    intersects_w[intersects_w < 0] = 0
    intersects_h[intersects_h < 0] = 0

    unions_area = anchors_area + gt_bboxes_area - (intersects_w * intersects_h)
    ious = (intersects_w * intersects_h) / unions_area
    
    return ious


