import time
import pdb
import random
import numpy as np
import torch
import torch.utils.data as data

from PIL import Image
from roi_data_layer.minibatch import get_minibatch, get_image_blob


class roibatchLoader(data.Dataset):
    def __init__(self, roidb, ratio_list, ratio_index, batch_size, num_classes,
                 training=True, normalize=None):
        self.roidb = roidb
        self.num_classes = num_classes
        self.trim_height = 600
        self.trim_width = 600
        self.max_num_box = 20
        self.training = training
        self.normalize = normalize
        self.ratio_list = ratio_list
        self.ratio_index = ratio_index
        self.batch_size = batch_size
        self.data_size = len(self.ratio_list)
        self.num_batch = int(np.ceil(len(ratio_index) / batch_size))

        # given the ratio_list, we want to make the ratio same for each batch
        self.ratio_list_batch = torch.zeros(self.data_size)
        self.target_size_batch = torch.zeros(self.data_size)

        for i in range(self.num_batch):
            left_index = i * batch_size
            right_index = min((i+1)*batch_size-1, self.data_size-1)

            if ratio_list[right_index] < 1:
                # for ratio < 1, we preserve the leftmost in each batch
                target_ratio = ratio_list[left_index]
            elif ratio_list[left_index] > 1:
                # for ratio > 1, we preserve the rightmost in each batch
                target_ratio = ratio_list[right_index]
            else:
                # for ratio cross 1, we make it to be 1
                target_ratio = 1
            
            self.ratio_list_batch[left_index:(right_index+1)] = target_ratio
        
        
        
    def resize_batch(self):
        scales = [400, 500, 600, 700, 800]
        for i in range(self.num_batch):
            left_index = i * self.batch_size
            right_index = min((i + 1) * self.batch_size - 1, self.data_size-1)

            if self.training:
                scale_index = np.random.randint(0, high=5, size=1)
                target_size = scales[scale_index[0]]
            else:
                target_size = 600 # test scale
            self.target_size_batch[left_index: (right_index + 1)] = target_size

    def __getitem__(self, index):
        if self.training:
            index_ratio = int(self.ratio_index[index])
        else:
            index_ratio = index

        # get the anchor index for current sample index
        # here we set the anchor index to the last one
        # sample in this group
        minibatch_db = [self.roidb[index_ratio]]
        target_size = [self.target_size_batch[index]]

        blobs = get_minibatch(minibatch_db, self.num_classes, target_size)
        data = torch.from_numpy(blobs['data'])
        image_info = torch.from_numpy(blobs['image_info'])

        data_height, data_width = data.size(1), data.size(2)

        if self.training:
            np.random.shuffle(blobs['gt_boxes'])
            gt_boxes = torch.from_numpy(blobs['gt_boxes'])

            # padding the input image to fixed size for each group

            # NOTE1: need to cope with the case where a group cover both conditions
            # NOTE2: need to consider the situation for the tail samples
            # NOTE3: need to implement a parralel data loader

            ratio = self.ratio_list_batch[index]

            if self.roidb[index_ratio]['need_crop']:
                if ratio < 1: # width << height -> crop height
                    min_y = int(torch.min(gt_boxes[:, 1]))
                    max_y = int(torch.max(gt_boxes[:, 3]))
                    trim_size = int(np.floor(data_width / ratio))
                    if trim_size > data_height:
                        trim_size = data_height
                    box_region = max_y - min_y + 1
                    if min_y == 0:
                        y_s = 0
                    else:
                        if (box_region - trim_size) < 0:
                            y_s_min = max(max_y - trim_size, 0)
                            y_s_max = min(min_y, data_height-trim_size)
                            if y_s_min == y_s_max:
                                y_s = y_s_min
                            else:
                                y_s = np.random.choice(range(y_s_min, y_s_max))
                        else:
                            y_s_add = int((box_region - trim_size) / 2)
                            if y_s_add == 0:
                                y_s = min_y
                            else:
                                y_s = np.random.choice(range(min_y, min_y + y_s_add))
                    # crop the image
                    data = data[:, y_s:(y_s + trim_size), :, :]

                    # shift y coordinate of gt_boxes
                    gt_boxes[:, 1] = gt_boxes[:, 1] - float(y_s)
                    gt_boxes[:, 3] = gt_boxes[:, 3] - float(y_s)

                    # update gt bouding box according the trip
                    gt_boxes[:, 1].clamp_(0, trim_size - 1) 
                    gt_boxes[:, 3].clamp_(0, trim_size - 1)
                else: # width >> height -> crop width
                    min_x = int(torch.min(gt_boxes[:, 0]))
                    max_x = int(torch.max(gt_boxes[:, 2]))
                    trim_size = int(np.ceil(data_height * ratio))
                    if trim_size > data_width:
                        trim_size = data_width
                    box_region = max_x - min_x + 1
                    if min_x == 0:
                        x_s = 0
                    else:
                        if (box_region - trim_size) < 0:
                            x_s_min = max(max_x - trim_size, 0)
                            x_s_max = min(min_x, data_width - trim_size)
                            if x_s_min == x_s_max:
                                x_s = x_s_min
                            else:
                                x_s = np.random.choice(range(x_s_min, x_s_max))
                        else:
                            x_s_add = int((box_region - trim_size) / 2)
                            if x_s_add == 0:
                                x_s = min_x
                            else:
                                x_s = np.random.choice(range(min_x, min_x + x_s_add))
                    
                    # crop the image
                    data = data[:, :, x_s:(x_s + trim_size), :]

                    # shift x coordinate of gt_boxes
                    gt_boxes[:, 0] = gt_boxes[:, 0] - float(x_s)
                    gt_boxes[:, 2] = gt_boxes[:, 2] - float(x_s)
                    gt_boxes[:, 0].clamp_(0, trim_size - 1)
                    gt_boxes[:, 2].clamp_(0, trim_size - 1)

            # based on ratio, padding the image
            if ratio < 1:
                trim_size = int(np.floor(data_width / ratio))

                padding_data = torch.zeros(int(np.ceil(data_width / ratio)), data_width, 3, dtype=torch.float32)
                padding_data[:data_height, :, :] = data[0]
                # update image_info
                image_info[0, 0] = padding_data.size(0)
            elif ratio > 1:
                padding_data = torch.zeros(data_height, int(np.ceil(data_height * ratio)), 3, dtype=torch.float32)
                padding_data[:, :data_width, :] = data[0]
                image_info[0, 1] = padding_data.size(1)
            else:
                trim_size = min(data_height, data_width)
                padding_data = torch.zeros(trim_size, trim_size, 3, dtype=torch.float32)
                padding_data = data[0][:trim_size, :trim_size, :]
                gt_boxes[:, :4].clamp_(0, trim_size)
                image_info[0, 0] = trim_size
                image_info[0, 0] = trim_size
            
            not_keep = (gt_boxes[:, 0] == gt_boxes[:, 2]) | (gt_boxes[:, 1] == gt_boxes[:, 3])
            keep = torch.nonzero(not_keep == 0).view(-1)
            
            gt_boxes_padding = torch.zeros(self.max_num_box, gt_boxes.size(1), dtype=torch.float32)
            if keep.numel() != 0:
                gt_boxes = gt_boxes[keep]
                num_boxes = min(gt_boxes.size(0), self.max_num_box)
                gt_boxes_padding[:num_boxes,:] = gt_boxes[:num_boxes]
            else:
                num_boxes = 0

                # permute trim_data to adapt to downstream processing
            padding_data = padding_data.permute(2, 0, 1).contiguous() # (C, H, W)
            image_info = image_info.view(3)

            return padding_data, image_info, gt_boxes_padding, num_boxes
        else:
            data = data.permute(0, 3, 1, 2).contiguous().view(3, data_height, data_width)
            image_info = image_info.view(3)

            gt_boxes = torch.FloatTensor([1,1,1,1,1])
            num_boxes = 0

            return data, image_info, gt_boxes, num_boxes
    
    def __len__(self):
        return len(self.roidb)
            