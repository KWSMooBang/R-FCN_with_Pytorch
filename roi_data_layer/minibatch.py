import pdb
import numpy as np
import cv2

from model.utils.blob import prepare_image_for_blob, image_list_to_blob

def get_minibatch(roidb, num_classes, target_size):
    """
    Given a roidb, construct a minibatch sampled from it
    """
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    assert(128 % num_images == 0), \
        "num_images ({}) must divide batch_size ({})".format(num_images, 128)
    
    image_blob, image_scales = get_image_blob(roidb, target_size)

    blobs = {'data': image_blob}

    assert len(image_scales) == 1, "Single batch only"
    assert len(roidb) == 1, "Single batch only"

    # gt boxes: (xmin, ymin, xmax, ymax, cls)
    gt_index = np.where(roidb[0]['gt_classes'] != 0)[0]
    gt_boxes = np.empty((len(gt_index), 5), dtype=np.float32)
    gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_index, :] * image_scales[0]
    gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_index]
    blobs['gt_boxes'] = gt_boxes
    blobs['image_info'] = np.array(
        [[image_blob.shape[1], image_blob.shape[2], image_scales[0]]],
        dtype=np.float32
    )

    blobs['image_id'] = roidb[0]['image_id']

    """
    blob: {
        data,
        image_id,
        image_info: [image_height, image_width, image_scales]
        gt_boxes
    }
    """
    return blobs


def get_image_blob(roidb, target_size):
    """
    Build an input blob from the images in the roidb at the specified scales
    """
    num_images = len(roidb)

    processed_images = []
    image_scales = []
    for i in range(num_images):
        image = cv2.imread(roidb[i]['image'])

        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
            image = np.concatenate((image, image, image), axis=2)
        image = image[:, :,::-1]

        if roidb[i]['flipped']:
            image = image[:, ::-1, :]
        image, image_scale = prepare_image_for_blob(image, np.array([[[102.9801, 115.9465, 122.7717]]]), 
                                                 target_size[i], 1000)
        image_scales.append(image_scale)
        processed_images.append(image)
    
    image_blob = image_list_to_blob(processed_images)

    return image_blob, image_scales
