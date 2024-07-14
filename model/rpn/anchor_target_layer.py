import numpy as np
import torch
import torch.nn as nn

from generate_anchors import generate_anchors
from bbox_utils import clip_bbox, transform_bbox, iou_bbox

class AnchorTargetLayer(nn.Module):
    """
    Assign anchors to groung-truth targets.
    Produces anchor classification labels and bounding-box regression targets
    """
    def __init__(self, scales, ratios, feat_stride):
        super(AnchorTargetLayer, self).__init__()

        self.feat_stride = feat_stride
        self.scales = scales
        self.ratios = ratios
        self.anchors = torch.from_numpy(generate_anchors(scales=np.array(scales),
                                                         ratios=np.array(ratios))).float()
        self.num_anchors = self.anchors.size(0)

        self.allowed_border = 0
    
    def forward(self, input):
        rpn_cls_score = input[0]
        gt_bboxes = input[1]
        image_info = input[2]
        num_boxes = input[3]

        batch_size = rpn_cls_score.size(0)

        height, width = rpn_cls_score.size(2), rpn_cls_score.size(3)
        
        feat_height, feat_width = rpn_cls_score.size(2), rpn_cls_score.size(3)
        shift_x = np.arange(0, feat_width) * self.feat_stride
        shift_y = np.arange(0, feat_height) * self.feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                             shift_x.ravel(), shift_y.ravel())).transpose())
        shifts = shifts.contiguous().type_as(rpn_cls_score).float()

        A = self.num_anchors
        K = shifts.size(0)

        self.anchors = self.anchors.type_as(gt_bboxes)
        anchors = self.anchors.view(1, A, 4) + shifts.view(K, 1, 4)
        anchors = anchors.view(K * A, 4)

        total_anchors = int(K * A)

        keep = ((anchors[:, 0] >= -self.allowed_border) &
                (anchors[:, 1] >= -self.allowed_border) &
                (anchors[:, 2] < image_info[0][1] + self.allowed_border) &
                (anchors[:, 3] < image_info[0][0] + self.allowed_border))
        
        keep_index = torch.nonzero(keep).view(-1)

        anchors = anchors[keep_index, :]

        labels = gt_bboxes.new(batch_size, keep_index.size(0)).fill_(-1)
        bbox_inside_weights = gt_bboxes.new(batch_size, keep_index.size(0)).zero_()
        bbox_outside_weights = gt_bboxes.new(batch_size, keep_index.size(0)).zero_()

        ious = iou_bbox(anchors, gt_bboxes)

        max_ious, argmax_ious = torch.max(ious, 2)
        gt_max_ious, _ = torch.max(ious, 1)

        gt_max_ious[gt_max_ious==0] = 1e-5
        keep = torch.sum(ious.eq(gt_max_ious.view(batch_size, 1, -1).expand_as(ious)), 2)

        if torch.sum(keep) > 0:
            labels[keep > 0] = 1

        labels[max_ious >= 0.7] = 1
        labels[max_ious < 0.3] = 0

        num_pos = 64

        sum_pos = torch.sum((labels == 1).int(), 1)
        sum_neg = torch.sum((labels == 0).int(), 1)

        for i in range(batch_size):
            if sum_pos[i] > num_pos:
                pos_index = torch.nonzero(labels[i] == 1).view(-1)

                random_num = torch.from_numpy(np.random.permutation(pos_index.size(0))).type_as(gt_bboxes)
                disable_index = pos_index[random_num[:pos_index.size(0) - num_pos]]
                labels[i][disable_index] = -1

            num_neg = 128 - torch.sum((labels == 1).int(), 1)[i]

            if sum_neg[i] > num_neg:
                neg_index = torch.nonzero(labels[i] == 0).view(-1)

                random_num = torch.from_numpy(np.random.permutation(neg_index.size(0))).type_as(gt_boxes)
                disable_index = neg_index[random_num[:neg_index.size(0) - num_neg]]
                labels[i][disable_index] = -1
        
        offset = torch.arange(0, batch_size) * gt_bboxes.size(1)

        argmax_ious = argmax_ious + offset.view(batch_size, 1).type_as(argmax_ious)
        bbox_targets = transform_bbox(anchors, gt_bboxes.view(-1, 5)[argmax_ious.view(-1), :].view(batch_size, -1, 5))

        bbox_inside_weights[labels == 1] = 1

        num_examples = torch.sum(labels >= 0)
        weights= 1.0 / num_examples

        bbox_outside_weights[labels == 1] = weights
        bbox_outside_weights[labels == 0] = weights

        labels = unmap(labels, total_anchors, keep_index, batch_size, fill=-1)
        bbox_targets = unmap(bbox_targets, total_anchors, keep_index, batch_size, fill=0)
        bbox_inside_weights = unmap(bbox_inside_weights, total_anchors, )

    def unmap(self, data, count, index, batch_size, fill=0):
        if data.dim() == 2:
            ret = torch.Tensor(batch_size, count).fill_(fill).type_as(data)
            ret[:, index] = data
        else:
            ret = torch.Tensor(batch_size, count, data.size(2)).fill_(fill).type_as(data)
            ret[:, index, :] = data
        return ret