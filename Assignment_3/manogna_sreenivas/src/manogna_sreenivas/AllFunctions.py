# This file should contain all the functions required by Wrapper.py

import numpy as np
import skimage
from skimage.filters import rank
from skimage.util import img_as_float

from manogna_sreenivas.helper import get_padded_image, get_distance_matrix, freq_filter


def average_filter(image: np.ndarray, kernel: int) -> np.ndarray:
    padded_im = get_padded_image(image, kernel)
    sum_image=np.zeros(image.shape)
    h,w=image.shape
    for k_h in range(kernel):
        for k_w in range(kernel):
            sum_image += padded_im[k_h:k_h + h, k_w:k_w + w]
    avg_image = sum_image/(kernel**2)
    return np.uint8(avg_image)


def highboost_filter(image: np.ndarray, k: float) -> np.ndarray:
    image = img_as_float(image)
    im_blur = skimage.filters.gaussian(image, sigma=3)
    im_filtered = (1 + k) * image - k * im_blur
    im_filtered = np.clip(im_filtered, 0, 1)
    return im_filtered


def get_sinusoidal_image(size: tuple, freq: tuple):
    indices = np.indices(size)
    M, N = size
    u_0, v_0 = freq
    theta = 2 * np.pi * ((u_0 * indices[0]) / M + (v_0 * indices[1]) / N)
    im_sine = np.sin(theta)
    return im_sine


def freq_domain_filter(image: np.ndarray, type='Gaussian', D0=100) -> np.ndarray:
    im_size = image.shape
    D = get_distance_matrix(im_size)

    # Get filter coefficients H(u,v)
    H = np.zeros(im_size)
    if type == 'Ideal_lowpass':
        H[np.where(D <= D0)] = 1
    elif type == 'Gaussian':
        H = np.exp(-1 * (D ** 2) / (2 * (D0 ** 2)))

    # Get image after filtering in frequency domain
    im_filtered = freq_filter(image, H)
    im_filtered = np.clip(im_filtered, 0, 255)
    return im_filtered


def homomorphic_filter(image: np.ndarray, param=[2, 0.5, 50]) -> np.ndarray:
    D = get_distance_matrix(image.shape)

    # Get filter coefficients H(u,v)
    gH, gL, D0 = param
    H = (gH - gL) * (1 - np.exp(-1 * (D ** 2) / (2 * (D0 ** 2)))) + gL

    # Transform image to log domain
    image = np.array(image, dtype='float')
    log_image = np.log(image + 1)

    # Get image after filtering in frequency domain
    filtered_im = freq_filter(log_image, H)

    # Get actual pixel values from log domain using exp
    filtered_im = np.exp(filtered_im)
    filtered_im = (filtered_im - np.min(filtered_im)) / (np.max(filtered_im) - np.min(filtered_im)) * 255
    return filtered_im
