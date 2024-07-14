import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from .proposal_layer import ProposalLayer
from .anchor_target_layer import AnchorTargetLayer

class RPN(nn.Module):
    """
    Region Proposal Network
    """
    def __init__(self, in_channels, scales=[8, 16, 32], ratios=[0.5, 1, 2],
                 feat_stride=16):
        self.in_channels = in_channels
        self.anchors_scales = scales
        self.anchors_ratios = ratios
        self.feat_stride = feat_stride

        self.RPN_conv = nn.Conv2d(self.in_channels, 512, 3, 1, 1, bias=True)
        
        # background / foreground score layer
        self.score_out_channels = len(self.anchors_scales) * len(self.anchors_ratios) * 2
        self.RPN_cls_score = nn.Conv2d(512, self.score_out_channels, 1, 1, 0) 

        # anchor box offset prediction layer
        self.bbox_out_channels = len(self.anchor_scales) * len(self.anchors_ratios) * 4
        self.RPN_reg_bbox = nn.Conv2d(512, self.bbox_out_channels, 1, 1, 0)

        # proposal layer
        self.RPN_proposal = ProposalLayer(self.feat_stride, self.anchors_scales, self.anchors_ratios)

        # anchor target layer
        self.RPN_anchor_target = AnchorTargetlayer(self.feat_stride, self.anchor_scales, self.anchors_ratios)

        self.rpn_loss_cls = 0
        self.rpn_loss_reg = 0

    @staticmethod
    def reshape(x, dim):
        input_shape = x.size()
        x = x.view(
            input_shape[0],
            int(dim),
            int(float(input_shape[1] * input_shape[2]) / float(dim)),
            input_shape[3]
        )
        return x

    def forward(self, base_feat, image_info, gt_bboxes, num_bboxes):
        batch_size = base_feat.size(0)

        rpn_conv1 = F.relu(self.RPN_conv(base_feat), inplace=True)
        rpn_cls_score = self.RPN_cls_score(rpn_conv1) # (batch_size, 2 * 9, H, W)

        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2) # (batch_size, 2, 9 * H, W)
        rpn_cls_prob_reshpae = self.F.softmax(rpn_cls_score_reshape, dim=1) 
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshpae, self.score_out_channels)

        rpn_reg_bbox = self.RPN_reg_bbox(rpn_conv1) # (batch_size, 4 * 9, H, W)

        rois = self.RPN_proposal((rpn_cls_prob.data, rpn_reg_bbox.data, image_info, self.training))
        
        self.rpn_loss_cls = 0
        self.rpn_loss_reg = 0
    
        if self.training:
            assert gt_bboxes is not None

            rpn_data = self.RPN_anchor_target((rpn_cls_score.data, gt_bboxes, image_info, num_bboxes))

            rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
            rpn_label = rpn_data[0].view(batch_size, -1)

            rpn_keep = rpn_label.view(-1).ne(-1).nonzeros().view(-1)
            rpn_cls_score = torch.index_select(rpn_cls_score.view(-1, 2), 0, rpn_keep)
            rpn_label = torch.index_select(rpn_label.view(-1), 0, rpn_keep.data)
            rpn_label = rpn_label.long()
            self.rpn_loss_cls = F.cross_entropy(rpn_cls_score, rpn_label)
            pos_count = torch.sum(rpn_label.data.ne(0))

            rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]

            self.rpn_loss_reg = F.smooth_l1_loss

        return rois, self.rpn_loss_cls, self.rpn_loss_reg
