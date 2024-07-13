import sys
sys.path.append('..')
import numpy as np
import datasets 

from PIL import Image
from datasets.CustomVOCDetection import CustomVOCDetection

def prepare_roidb(dataset: CustomVOCDetection):
    roidb = dataset.roidb
    sizes = [Image.open(dataset.images[i]).size
             for i in range(dataset.num_images)]
    
    for i in range(dataset.num_images):
        roidb[i]['image_id'] = i
        roidb[i]['image'] = dataset.images[i]
        roidb[i]['width'] = sizes[i][0]
        roidb[i]['height'] = sizes[i][1]
        gt_ious = roidb[i]['gt_ious'].toarray()
        max_ious = gt_ious.max(axis=1)
        max_classes = gt_ious.argmax(axis=1)
        roidb[i]['max_classes'] = max_classes
        roidb[i]['max_ious'] = max_ious
        zero_index = np.where(max_ious == 0)[0]
        assert all(max_classes[zero_index] == 0)
        nonzero_index = np.where(max_ious > 0)[0]
        assert all(max_classes[nonzero_index] != 0)


def rank_roidb_ratio(roidb):
    ratio_large = 2 # largest ratio to preserve
    ratio_small = 0.5 # smallest ratio to preserve

    ratio_list = []
    for i in range(len(roidb)):
        width = roidb[i]['width']
        height = roidb[i]['height']
        ratio = width / float(height)

        if ratio > ratio_large:
            roidb[i]['need_crop'] = 1
            ratio = ratio_large
        elif ratio < ratio_small:
            roidb[i]['need_crop'] = 1
            ratio = ratio_small
        else:
            roidb[i]['need_crop'] = 0

        ratio_list.append(ratio)
    
    ratio_list = np.array(ratio_list)
    ratio_index = np.argsort(ratio_list)
    return ratio_list[ratio_index], ratio_index

def filter_roidb(roidb):
    print("before filtering, there are %d images..." %(len(roidb)))
    i = 0
    while i < len(roidb):
        if len(roidb[i]['boxes']) == 0:
            del roidb[i]
            i -= 1
        i += 1
    
    print("after filtering, there are %d images..." %(len(roidb)))
    return roidb

def combined_roidb(dataset: CustomVOCDetection, training=True, isFlip=False):
    """
    Combine multiple roidbs
    """

    def get_training_roidb(dataset: CustomVOCDetection):
        if isFlip:
            print("Appending horizontally-flipped training examples...")
            dataset.append_flipped_images()
            print("Done")
        
        print("Preparing training data...")
        prepare_roidb(dataset)
        print("Done")
        
        return dataset.roidb

    roidb = get_training_roidb(dataset)
    
    if training:
        roidb = filter_roidb(roidb)

    ratio_list, ratio_index = rank_roidb_ratio(roidb)

    return roidb, ratio_list, ratio_index