#####################################################################################
#
# Visión por Computador 2024 - Práctica 1
#
#####################################################################################
#
# Authors: Alejandro Perez, Jesus Bermudez, JMM Montiel
#
# Version: 1.0
#
#####################################################################################

import numpy as np

def convolve(image, kernel):
    # Get the dimensions of the image and kernel
    img_height, img_width = image.shape
    kernel_size = len(kernel)

    # Calculate the padding size to maintain the original image size
    pad_size = kernel_size // 2

    # Create a padded image with zeros
    padded_image = np.zeros((img_height + 2 * pad_size, img_width + 2 * pad_size))
    padded_image[pad_size:pad_size + img_height, pad_size:pad_size + img_width] = image

    # Create an output image
    output_image = np.zeros_like(image, dtype=float)

    # Convolution
    for i in range(img_height):
        for j in range(img_width):
            output_image[i, j] = np.sum(padded_image[i:i + kernel_size, j:j + kernel_size] * kernel)

    return output_image