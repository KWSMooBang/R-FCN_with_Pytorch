import numpy as np
import cv2 

def image_list_to_blob(images):
    """
    Convert a list of images into a network input
    Assume images are already prepared 
    """
    max_shape = np.array([image.shape for image in images]).max(axis=0)
    num_images = len(images)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
    for i in range(num_images):
        image = images[i]
        blob[i, 0:image.shape[0], 0:image.shape[1], :] = image
    
    return blob

def prepare_image_for_blob(image, pixel_means, target_size, max_size):
    """
    Mean substract and scale an image for use in a blob
    """
    image = image.astype(np.float32, copy=False)
    image -= pixel_means
    image_shape= image.shape
    image_size_min = np.min(image_shape[0:2])
    image_size_max = np.max(image_shape[0:2])
    image_scale = float(target_size) / float(image_size_min)
    if np.round(image_scale * image_size_max) > max_size:
        image_scale = float(max_size) / float(image_size_max)
    image = cv2.resize(image, dsize=(0, 0), fx=image_scale, fy=image_scale,
                       interpolation=cv2.INTER_LINEAR)
    return image, image_scale