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

def clip_bbox(bboxes, image_shape, batch_size):
    for i in range(batch_size):
        bboxes[i, :, 0::4].clamp_(0, image_shape[i, 1]-1)
        bboxes[i, :, 1::4].clamp_(0, image_shape[i, 0]-1)
        bboxes[i, :, 2::4].clamp_(0, image_shape[i, 1]-1)
        bboxes[i, :, 3::4].clamp_(0, image_shape[i, 0]-1)
        
    return bboxes

def iou_bbox(anchors, gt_bboxes):
    """
    calculate iou between anchors and ground truth bboxes
    Args:
        anchors: (N, 4)
        gt_bboxes: (batch_size, K, 4)

        overlaps: (N, K)
    """
    batch_size = gt_bboxes.size(0)

    N = anchors.size(0)
    K = gt_bboxes.size(1)

    anchors = anchors.view(1, N, 4).expand(batch_size, N, 4).contiguous()
    gt_bboxes = gt_bboxes[:, :, :4].contiguous()

    gt_bboxes_x = (gt_bboxes[:, :, 2] - gt_bboxes[:, :, 0] + 1)
    gt_bboxes_y = (gt_bboxes[:, :, 3] - gt_bbpxes[:, :, 1] + 1)
    gt_bboxes_area = (gt_bboxes_x * gt_bboxes_y).view(batch_size, 1, K)

    anchors_x = (anchors[:, :, 2] - anchors[:, :, 0] + 1)
    anchors_y = (anchors[:, :, 3] - anchors[:, :, 1] + 1)
    anchors_area = (anchors_x * anchors_y).view(batch_size, N, 1)

    gt_area_zero = (gt_bboxes_x == 1) & (gt_bboxes_y == 1)
    anchors_area_zero = (anchors_x == 1) & (anchors_y == 1)

    boxes = anchors.view(batch_size, N, 1, 4).expand(batch_size, N, K, 4)
    query_boxes = gt_bboxes.view(batch_size, 1, K, 4).expand(batch_size, N, K, 4)

    intersects_w = (torch.min(boxes[:, :, :, 2], query_boxes[:, :, :, 2]) - 
                torch.max(boxes[:, :, :, 0], query_boxes[:, :, :, 0]) + 1)
    intersects_h = (torch.min(boxes[:, :, :, 3], query_boxes[:, :, :, 2]) -
                torch.max(boxes[:, :, :, 1], query_boxes[:, :, :, 1]) + 1)
    intersects_w[intersects_w < 0] = 0
    intersects_h[intersects_h < 0] = 0

    unions_area = anchors_area + gt_bboxes_area - (intersects_w * intersects_h)
    ious = (intersects_w * intersects_h) / unions_area

    ious.masked_fill_(gt_area_zero.view(batch_size, 1, K).expand(batch_size, N, K), 0)
    ious.masked_fill_(anchors_area_zero.view(batch_size, N, 1).expand(batch_size, N, K), 0)
    
    return ious