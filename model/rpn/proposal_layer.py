import torch
import torch.nn as nn
import numpy as np
import math

from .generate_anchors import generate_anchors
from .bbox_utils import inverse_transform_bbox, clip_bbox
from torchvision.ops import nms

class ProposalLayer(nn.Module):
    """
    Outputs object detection proposals(roi) by applying estimated bbox
    transformations to a set of regular boxes 
    """

    def __init__(self, feat_stride, scales, ratios,):
        super(ProposalLayer, self).__init__()

        self.train_pre_nms_n = 12000
        self.train_post_nms_n = 2000
        self.train_nms_thresh = 0.7
        self.train_min_size = 8

        self.test_pre_nms_n = 6000
        self.test_post_nms_n = 300
        self.test_nms_thresh = 0.7
        self.test_min_size = 16

        self.feat_stride = feat_stride
        self.anchors = torch.from_numpy(generate_anchors(scales=np.array(scales),
                                    ratios=np.array(ratios))).float()
        self.num_anchors = self.anchors.size(0)


    def forward(self, input):
        scores = input[0][:, self.num_anchors:, :, :] # (batch_size, 9, H, W)
        bbox_targets = input[1]
        image_info = input[2]
        training = input[3]

        if training:
            pre_nms_n = self.train_pre_nms_n
            post_nms_n = self.train_post_nms_n
            nms_thresh = self.train_nms_thresh
            min_size = self.train_min_size
        else:
            pre_nms_n = self.test_pre_nms_n
            post_nms_n = self.test_post_nms_n
            nms_thresh = self.test_nms_thresh
            min_size = self.test_min_size

        batch_size = bbox_targets.size(0)

        feat_height, feat_width = scores.size(2), scores.size(3)
        shift_x = np.arange(0, feat_width) * self.feat_stride
        shift_y = np.arange(0, feat_height) * self.feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                             shift_x.ravel(), shift_y.ravel())).transpose())
        shifts = shifts.contiguous().type_as(scores).float()

        A = self.num_anchors
        K = shifts.size(0)

        self.anchors = self.anchors.type_as(scores)
        anchors = self.anchors.view(1, A, 4) + shifts.view(K, 1, 4)
        anchors = anchors.view(1, K * A, 4).expand(batch_size, K * A, 4)

        bbox_targets = bbox_targets.permute(0, 2, 3, 1).contiguous() # (batch_size, H, W, ( * 4))
        bbox_targets = bbox_targets.view(batch_size, -1, 4) # (batch_size, K * A, 4)

        scores = scores.premute(0, 2, 3, 1).contiguous() # (batch_size, H, W, 9)
        scores = scores.view(batch_size, -1) # (batch_size, K * A)

        proposals = inverse_transform_bbox(anchors, bbox_targets)

        proposals = clip_bbox(proposals, image_info, batch_size)

        scores_keep = scores
        proposals_keep = proposals
        _, order = torch.sort(scores_keep, 1, True)

        output = scores.new(batch_size, post_nms_n, 5).zeros_()
        for i in range(batch_size):
            proposal_single = proposals_keep[i]
            scores_single = scores_keep 

            order_single= order[i]

            if pre_nms_n > 0 and pre_nms_n < scores_keep.numel():
                order_single = order_single[:pre_nms_n]
            
            proposal_single = proposal_single[order_single, :]
            scores_single = scores_single[order_single]

            keep_index = nms(proposal_single, scores_single, nms_thresh)

            if post_nms_n > 0"
                keep_index = keep_index[:post_nms_n]

            proposal_single = proposal_single[keep_index, :]
            scores_single = scores_single[keep_index]

            num_proposal = proposal_single.size(0)
            output[i, :, 0] = i
            output[i, :num_proposal, 1:] = proposal_single

        return output