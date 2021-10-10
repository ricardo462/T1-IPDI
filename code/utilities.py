import numpy as np
from numpy.random import randint

def convolution_dimensions(img, kernel):
    """Computes the final dimensions of the convolution with the image and kernel provided"""
    img_dimensions = np.array(list(img.shape))
    kernel_dimensions = np.array(list(kernel.shape))
    conv_image_dimensions = img_dimensions-kernel_dimensions + 1
    return conv_image_dimensions 

def convolution(img, kernel, keepdim:bool = True):
    """Computes the convolution of the image and the kernel provided"""
    dimensions = convolution_dimensions(img, kernel)
    conv = np.zeros(tuple(dimensions.tolist()))
    kernel = np.fliplr(np.flipud(kernel))
    for i in range(dimensions[0]):
        for j in range(dimensions[1]):
            # Here i gotta compute every posirion of the convolution
            img_extracted = extract(img, kernel, i, j)
            conv[i, j] = np.sum(img_extracted * kernel)
    if keepdim:
        final_dimensions = np.array(list(img.shape))
        index = final_dimensions - dimensions + 1
        result  = np.zeros(img.shape)
        x = 2
        y = 2
        result[index[0] -x :index[0] + dimensions[0] - y, index[1] - x :index[1] + dimensions[1] - y ] = conv
        return result
    return conv

def extract(img, kernel, i, j):
    """ Extracts a block on the i,j position with kernel dimension"""
    kernel_dim = list(kernel.shape)
    return img[i:i + kernel_dim[0], j: j + kernel_dim[1]]

def balanced_kernel(center, borders, dim:tuple):
    """Returns a balanced kernel with the dimensions, center value and borders values provided"""
    kernel = np.zeros(dim)
    # Vertical borders
    kernel[0, :] = borders
    kernel[dim[1] - 1, :] = borders
    # Horizontal borders
    kernel[:, 0] = borders
    kernel[:, dim[0] - 1] = borders
    # Center
    kernel[int(dim[0]/2), int(dim[1]/2)] = center
    return kernel


def find_deprecated(detection, percentage_threshold = 0.8):
    """Returns an array with the indexes that contains a value detection greater than the threshold.
     The threshold is calculated as percentage_threshold * max(detection) """
    maximum_detection = np.max(detection)
    threshold = percentage_threshold * maximum_detection
    detection_shape = detection.shape
    indexes = []
    for i in range(detection_shape[1]-1):
        for j in range(detection_shape[0]-1):
            if detection[j,i] > threshold:
                indexes.append([j,i])
    return np.array(indexes)


def find_by_percentage(detection, percentage_threshold=0.8):
    """Returns an array with the indexes that contains a value detection greater than the threshold.
    The threshold is calculated as percentage_threshold * max(detection) """
    maximum_value = np.max(detection)
    threshold = maximum_value * percentage_threshold
    return np.nonzero(detection > threshold)
    
def find_by_value(detection, threshold=100):
    """Returns an array with the indexes that contains a value detection greater than the threshold.
    The threshold is calculated as percentage_threshold * max(detection) """
    return np.nonzero(detection > threshold)