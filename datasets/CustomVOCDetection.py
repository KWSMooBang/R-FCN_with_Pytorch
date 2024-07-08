import os
import xml.etree.ElementTree as ET
import pickle
import numpy as np
import scipy.sparse
import torch

from PIL import Image
from torchvision.datasets import VOCDetection
from torchvision.ops import box_iou

class CustomVOCDetection(VOCDetection):
    def __init__(self, root, year, image_set, download=False, 
                 transform=None, target_transform=None, transforms=None):
        super().__init__(root, year, image_set, download, 
                         transform, target_transform, transforms)
        
        self.cache_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'cache'))
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

        self.name = 'pascal_voc_' + year + '_' + image_set
        self.classes = ('background',  # always index 0
                         'aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor')
        self.num_classes = len(self.classes)
        self.class_to_index = dict(zip(self.classes, range(self.num_classes)))
        self.num_images = len(self.images)
        self.obj_proposer = 'gt'
        self.roidb_handler = self.gt_roidb
        self.roidb = self.roidb_handler()

    def get_widths(self):
        return [Image.open(self.images[i]).size[0]
                for i in range(self.num_images)]

    def gt_roidb(self):
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                roidb = pickle.load(f)
            print(f"{self.name} gt roidb loaded from {cache_file}")
            return roidb

        gt_roidb = [self.get_annotation(index)
                    for index in range(self.num_images)]
        with open(cache_file, 'wb') as f:
            pickle.dump(gt_roidb, f, pickle.HIGHEST_PROTOCOL)
        print(f"wrote gt roidb to {cache_file}")

        return gt_roidb
    
    def rpn_roidb(self):    
        cache_file = os.path.join(self.cache_path, self.name + '_rpn_roidb.pkl')
        assert os.path.exists(cache_file), \
            f"rpn roidb no found at {cache_file}"
        with open(cache_file, 'rb') as f:
            box_list = pickle.load(f)
        print(f"{self.name} rpn roidb loaded from {cache_file}")

        if int(self.year) == 2007 or self.image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self.create_roidb_from_box_list(box_list, gt_roidb)
            roidb = self.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self.create_roidb_from_box_list(box_list, None)
        
        return roidb

    def get_annotation(self, index):
        """
        Get image and bounding boxes info from XML file in the PASCAL VOC format.
        """
        tree = ET.parse(self.annotations[index])
        objs = tree.findall('object')
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        ious = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        areas = np.zeros((num_objs), dtype=np.float32)
        difficult = np.zeros((num_objs), dtype=np.int32)

        # Get object bouding boxes into a data frame
        for i, obj in enumerate(objs):
            box = obj.find('bndbox')
            xmin = float(box.find('xmin').text) - 1
            ymin = float(box.find('ymin').text) - 1
            xmax = float(box.find('xmax').text) - 1
            ymax = float(box.find('ymax').text) - 1

            diffc = obj.find('difficult')
            ishard = 0 if diffc is None else int(diffc.text)
            difficult[i] = ishard

            cls = self.class_to_index[obj.find('name').text.lower().strip()]
            boxes[i, :] = [xmin, ymin, xmax, ymax]
            gt_classes[i] = cls
            ious[i, cls] = 1.0
            areas[i] = (xmax - xmin + 1) * (ymax - ymin + 1)

        ious = scipy.sparse.csr_matrix(ious)

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_difficult': difficult,
                'gt_ious': ious,
                'flipped': False,
                'areas': areas} 
    
    def create_roidb_from_box_list(self, box_list, gt_roidb):
        assert len(box_list) == self.num_images, \
            "Number of boxes must match number of ground-truth images"
        roidb = []
        for i in range(self.num_images):
            boxes = box_list[i]
            num_boxes = boxes.shape[0]
            ious = np.zeros((num_boxes, self.num_classes), dtype=np.float32)

            if gt_roidb is not None and gt_roidb[i]['boxes'].size > 0:
                gt_boxes = gt_roidb[i]['boxes']
                gt_classes = gt_roidb[i]['gt_classes']
                gt_ious = box_iou(torch.tensor(boxes.astype(np.float)),
                                  torch.tensor(gt_boxes.astype(np.float))).to_numpy()
                argmaxes= gt_ious.argmax(axis=1)
                maxes = gt_ious.max(axis=1)
                I = np.where(maxes > 0)[0]
                ious[I, gt_classes[argmaxes[I]]] = maxes[I]

            ious = scipy.sparse.csr_matrix(ious)
            roidb.append({
                'boxes': boxes,
                'gt_classes': np.zeros((num_boxes,), dtype=np.int32),
                'gt_ious': ious,
                'flipped': False,
                'areas': np.zeros((num_boxes,), dtype=np.float32),
            })
        return roidb
    
    def append_flipped_images(self):
        num_images = self.num_images
        widths = self.get_widths()
        for i in range(num_images):
            boxes = self.roidb[i]['boxes'].copy()
            oldxmin = boxes[:, 0].copy()
            oldxmax = boxes[:, 2].copy()
            boxes[:, 0] = widths[i] - oldxmax - 1
            boxes[:, 2] = widths[i] - oldxmin - 1
            assert (boxes[:, 2] >= boxes[:, 0]).all()
            entry = {'boxes': boxes,
                     'gt_ious': self.roidb[i]['gt_ious'],
                     'gt_classes': self.roidb[i]['gt_classes'],
                     'flipped': True,
                     'areas': self.roidb[i]['areas']}
            self.roidb.append(entry)
        self.images = self.images * 2

    @staticmethod
    def merge_roidbs(a, b):
        assert len(a) == len(b)
        for i in range(len(a)):
            a[i]['boxes'] = np.vstack((a[i]['boxes'], b[i]['boxes']))
            a[i]['gt_classes'] = np.hstack((a[i]['gt_classes'],
                                            b[i]['gt_classses']))
            a[i]['gt_ious'] = scipy.sparse.vstack([a[i]['gt_ious'],
                                                   b[i]['gt_ious']])
            a[i]['areas'] = np.hstack((a[i]['areas'],
                                       b[i]['areas']))
        return a