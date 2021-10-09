import numpy as np
from numpy.random import randint

def convolution_dimensions(img, kernel):
    """Computes the final dimensions of the convolution with the image and kernel provided"""
    img_dimensions = np.array(list(img.shape))
    kernel_dimensions = np.array(list(kernel.shape))
    conv_image_dimensions = img_dimensions-kernel_dimensions + 1
    return conv_image_dimensions 

def convolution(img, kernel):
    """Computes the convolution of the image and the kernel provided"""
    dimensions = convolution_dimensions(img, kernel)
    conv = np.zeros(tuple(dimensions.tolist()))
    for i in range(dimensions[0]):
        for j in range(dimensions[1]):
            # Here i gotta compute every posirion of the convolution
            img_extracted = extract(img, kernel, i, j)
            conv[i, j] = np.sum(img_extracted * kernel)
    return conv

def extract(img, kernel, i, j):
    """ Extracts a block on the i,j position with kernel dimension"""
    kernel_dim = list(kernel.shape)
    return img[i:i + kernel_dim[0], j: j + kernel_dim[1]]