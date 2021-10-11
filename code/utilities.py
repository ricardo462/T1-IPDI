import numpy as np
from PIL import Image

def convolution_dimensions(img, kernel):
    """Computes the final dimensions of the convolution with the image and kernel provided"""
    img_dimensions = np.array(list(img.shape))
    kernel_dimensions = np.array(list(kernel.shape))
    conv_image_dimensions = img_dimensions-kernel_dimensions + 1
    return conv_image_dimensions 

def convolution(img, kernel, keepdim:bool = False, invert:bool= False):
    """Computes the convolution of the image and the kernel provided"""
    dimensions = convolution_dimensions(img, kernel)
    conv = np.zeros(tuple(dimensions.tolist()))
    if invert:
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

def binarize_img(image_paht:str):
    """Binarizes the image in the path provided"""
    imagen = np.array(Image.open(image_paht))
    dimensions = imagen.shape
    binary_image = np.zeros((dimensions[0], dimensions[1]))
    threshold = 255*3/2
    for i in range(dimensions[0]):
        for j in range(dimensions[1]):
            binary_image[i, j] = 0 if np.sum(imagen[i, j, :]) > threshold else 1
    return binary_image

def eliminate_non_maximum(indices:tuple, neighborhood:int = 5):
    """Saves only one index if there is more than one in the neighborhood provided"""
    indices_list = []
    x = indices[0]
    y = indices[1]
    for i in range(len(x)):
        indices_list.append([x[i], y[i]])
    indices_list = np.array(indices_list)
    normalized_indices = []
    normalized_indices.append(indices_list[0,:])
    for p in indices_list:
        if not is_in(p, normalized_indices, neighborhood):
            normalized_indices.append(p)
    normalized_indices = np.array(normalized_indices)
    return normalized_indices


def is_in(point, points, neighborhood):
    """Checks if the point is in points, considering the points can be in a neighborhood with the value provided"""
    x = point[0]
    y = point[1]
    for p in points:
        if abs(x - p[0]) <= neighborhood:
            if abs(y - p[1]) <= neighborhood:
                return True
    return False

    

