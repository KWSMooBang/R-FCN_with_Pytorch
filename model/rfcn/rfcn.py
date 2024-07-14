import torch
import torch.nn as nn
import torch.nn.functional as F

from model.rpn.proposal_target_layer import ProposalTargetLayer
from model.rpn.rpn import RPN

class RFCN(nn.Module):
    """
    R-FCN
    """
    def __init__(self, base_model, classes, class_agnostic):
        super(RFCN, self).__init__()
        self.RFCN_base = base_model

        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RFCN_loss_cls = 0
        self.RFCN_loss_reg = 0
        
        self.bbox_num_classes = 1 if class_agnostic else self.n_classes

        # rpn
        self.RFCN_rpn = RPN(self.base_model.out_channels)
        self.RFCN_proposal_target = ProposalTargetLayer(self.n_classes)
        self.RFCN_psroi_pool_cls = 
        self.RFCN_psroi_pool_reg =
        self.pooling = 
        self.grid_size

    def forward(self, image_data, image_info, gt_bboxes, num_bboxes):
        self.batch_size = image_data.size(0)

        image_info = image_info.data
        gt_bboxes = gt_bboxes.data
        num_bboxes = num_bboxes.data

        base_feat = self.RFCN_base(image_data)

        rois, rpn_loss_cls, rpn_loss_reg = self.RFCN_rpn(base_feat, image_info, gt_bboxes, num_bboxes)

