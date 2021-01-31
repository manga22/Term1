# This file should contain all the functions required by Wrapper.py

import numpy as np
import skimage
from numpy.fft import fft2, ifft2
from skimage.filters import gaussian
from skimage.filters.rank import median
from skimage.morphology import square
import matplotlib.pyplot as plt

from manogna_sreenivas.helper import get_padded_image, get_distance_matrix, freq_filter, convolve


def decimate(image: np.ndarray, d=2, filter=None, D0=None) -> np.ndarray:
    im_size = image.shape
    if filter == 'Gaussian':
        D = get_distance_matrix(im_size)
        H = np.zeros(im_size)
        H = np.exp(-1 * (D ** 2) / (2 * (D0 ** 2)))

        # Get image after filtering in frequency domain
        im_filtered = freq_filter(image, H)
        image = np.clip(im_filtered, 0, 255)
        image = np.array(image, dtype=np.uint8)

    deci_image = np.zeros((int(im_size[0] / 2), int(im_size[1] / 2)))
    y, x = np.indices(deci_image.shape) * d
    deci_image = image[y, x]
    return deci_image


def wiener_filter(image: np.ndarray, h: np.ndarray, noise_var=5) -> np.ndarray:
    image = skimage.img_as_float(image)
    M, N = image.shape
    H = fft2(h, s=image.shape)                  # FFT of given blur filter
    nsr = 0.01
    D = np.conj(H) / (np.abs(H) ** 2 + nsr)     # Weiner filter
    f_image = fft2(image)                       # get fft of image
    im_filtered = ifft2(f_image * D)            # Filter image in frequency domain
    im_filtered = np.abs(im_filtered)
    deblur_im = np.uint8(np.clip(im_filtered, 0, 1) * 255)
    return deblur_im


def inverse_filter(image: np.ndarray, h: np.ndarray) -> np.ndarray:
    image = skimage.img_as_float(image)
    M, N = image.shape
    H = fft2(h, s=image.shape)                  # FFT of given blur filter
    D = 1 / H                                   # Wiener filter
    f_image = fft2(image)                       # get fft of image
    im_filtered = ifft2(f_image * D)            # Filter image in frequency domain
    im_filtered = np.abs(im_filtered)
    deblur_im = (im_filtered - np.min(im_filtered)) / (np.max(im_filtered) - np.min(im_filtered)) * 255
    return deblur_im


def cls_filter(image: np.ndarray, h: np.ndarray, gamma=10**(-2)) -> np.ndarray:
    image = skimage.img_as_float(image)
    M, N = image.shape
    H = fft2(h, s=image.shape)                  # FFT of given blur filter

    p = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])     # Get CLS filter
    P = fft2(p, s=image.shape)
    D = np.conj(H) / (np.abs(H) ** 2 + gamma * np.abs(P) ** 2)

    f_image = fft2(image)                       # get fft of image
    im_filtered = ifft2(f_image * D)            # Filter image in frequency domain
    im_filtered = np.abs(im_filtered)
    deblur_im = np.uint8(np.clip(im_filtered, 0, 1) * 255)
    return deblur_im


def image_denoise(image: np.ndarray, type='Gaussian', param=1) -> np.ndarray:
    if type == 'Gaussian':
        denoised_img = gaussian(image, sigma=param)
        denoised_img = np.uint8(denoised_img * 255)
    if type == 'Median':
        denoised_img = median(image, square(param))
    return denoised_img


def bilateral_filter(image: np.ndarray, param: list) -> np.ndarray:
    image = skimage.img_as_float(image)
    sigma_g, sigma_h, k_size = param

    k_h, k_w = k_size
    h, w = image.shape
    d = get_distance_matrix((k_h, k_w))
    G = np.exp(-d / (2 * sigma_g ** 2))  # spatial smoothing Gaussian filter
    padded_image = get_padded_image(image, (k_h, k_w))
    conv_image = np.zeros(image.shape, dtype=float)

    K = 0
    for i in range(k_h):
        for j in range(k_w):
            I_diff = padded_image[i:i + h, j:j + w] - image
            H = np.exp(-I_diff ** 2 / (2 * sigma_h ** 2))  # Gaussian using Luminance distance
            conv_image += padded_image[i:i + h, j:j + w] * G[i, j] * H
            K += G[i, j] * H  # Collect weights to normalize

    filtered_im = np.uint8(conv_image / K * 255)
    return filtered_im


def nonlocalmeans_filter(image: np.ndarray, param: list) -> np.ndarray:
    image = skimage.img_as_float(image)
    sigma_g, sigma_h, k_size = param

    k_h, k_w = k_size
    h, w = image.shape
    d = get_distance_matrix((k_h, k_w))
    padded_image = get_padded_image(image, (k_h, k_w))
    conv_image = np.zeros(image.shape, dtype=float)

    K = 0
    for i in range(k_h):
        for j in range(k_w):
            I_diff = padded_image[i:i + h, j:j + w] - image
            H = np.exp(-I_diff ** 2 / (2 * sigma_h ** 2))  # Gaussian using Luminance distance
            conv_image += padded_image[i:i + h, j:j + w] * H
            K += H  # Collect weights to normalize

    filtered_im = np.uint8(conv_image / K * 255)
    return filtered_im


def edge_detect(image: np.ndarray, thresh=70, type='Sobel', plot_grad=False) -> np.ndarray:
    image = gaussian(image, sigma=3)
    if type == 'Sobel':
        Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        Gy = Gx.T
    elif type == 'Prewitt':
        Gx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        Gy = Gx.T
    I_x = convolve(image, Gx)  # Horizontal gradients
    I_y = convolve(image, Gy)  # Vertical gradients
    grad = np.sqrt(I_x ** 2 + I_y ** 2)
    grad = (grad - np.min(grad)) / (np.max(grad) - np.min(grad)) * 255
    if plot_grad:
        plt.imshow(grad, cmap='gray')
        plt.title('Gradient image')
        plt.show()
    binary = grad > thresh
    binary = np.uint8(binary * 255)
    return binary
