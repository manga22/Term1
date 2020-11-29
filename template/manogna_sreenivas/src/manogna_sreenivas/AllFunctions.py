# This file should contain all the functions required by Wrapper.py

from pathlib import Path
import time
import skimage
import math
import numpy as np
from matplotlib import pyplot
from matplotlib.colors import NoNorm
from manogna_sreenivas.helper import image_hist, clip_histogram, interpolate, interpolate_vec


def linear_contrast(image: np.ndarray, gain: float) -> np.ndarray:
    enhanced_image = image * gain
    return enhanced_image.astype(np.uint8)


def powerlaw_contrast(image: np.ndarray, exp: float) -> np.ndarray:
    image = image / 255
    enhanced_image = image ** exp
    enhanced_image = np.array(enhanced_image * 255, dtype=np.uint8)
    return enhanced_image


def hist_equalization(image: np.ndarray, clip=False, plot=False, clahe=False) -> np.ndarray:
    bins_vec, freq_vec, prob_vec = image_hist(image)
    valid_hist = np.where(freq_vec > 0)
    bins_vec = bins_vec[valid_hist]
    prob_vec = prob_vec[valid_hist]
    freq_vec = freq_vec[valid_hist]
    if clip:
        prob_vec = clip_histogram(prob_vec, clip)
    cdf_image = np.cumsum(prob_vec)
    enhanced_image = np.zeros(image.shape)
    for k in range(bins_vec.shape[0]):
        enhanced_image[np.where(image == bins_vec[k])] = cdf_image[k]
    if clahe:
        return enhanced_image
    enhanced_image = np.array(enhanced_image * 255, dtype=np.uint8)
    if plot:
        pyplot.subplot(221, title='Original image')
        pyplot.imshow(image, cmap='gray', norm=NoNorm())
        pyplot.subplot(222, title='After equalization')
        pyplot.imshow(enhanced_image, cmap='gray', norm=NoNorm())
        pyplot.subplot(223)
        pyplot.hist(image.flatten(), bins=255, range=(0, 256), density=True)
        pyplot.subplot(224)
        pyplot.hist(enhanced_image.flatten(), bins=255, range=(0, 256), density=True)
        pyplot.show()
    return enhanced_image


def clahe(image: np.ndarray, clip=0.015, overlap=0.25) -> np.ndarray:
    k = 8
    image = image
    h, w = image.shape
    clahe_im = np.zeros(image.shape)
    if overlap:
        n_x = 4 * int(w / (3 * k + 1) + 1)
        n_y = 4 * int(h / (3 * k + 1) + 1)
        overlap_x = int(overlap * n_x)
        overlap_y = int(overlap * n_y)
        shift_x = int(3 / 4 * n_x)
        shift_y = int(3 / 4 * n_y)
        for i in range(k):
            for j in range(k):
                tile = image[i * shift_y:i * shift_y + n_y, j * shift_x:j * shift_x + n_x]
                enhanced_tile = hist_equalization(tile, clip=clip, clahe=True)
                if j > 0:
                    enhanced_tile[:, :overlap_x] = (enhanced_tile[:, :overlap_x] + \
                                                    clahe_im[i * shift_y:i * shift_y + n_y,
                                                    j * shift_x:j * shift_x + overlap_x]) / 2
                if i > 0:
                    enhanced_tile[:overlap_y, :] = (enhanced_tile[:overlap_y, :] + \
                                                    clahe_im[i * shift_y:i * shift_y + overlap_y,
                                                    j * shift_x:j * shift_x + n_x]) / 2
                clahe_im[i * shift_y:i * shift_y + n_y, j * shift_x:j * shift_x + n_x] = enhanced_tile
    else:
        n_x = int(w / k)
        n_y = int(h / k)
        for i in range(k):
            for j in range(k):
                tile = image[i * n_y:(i + 1) * n_y, j * n_x:(j + 1) * n_x]
                clahe_im[i * n_y:(i + 1) * n_y, j * n_x:(j + 1) * n_x] = hist_equalization(tile, clip=clip, clahe=True)
    return clahe_im


def saturated_contrast(image: np.ndarray, percent=0.15, plot=True, powerlaw=False) -> np.ndarray:
    ch = ['R', 'G', 'B']
    enhanced_im = np.zeros(image.shape, dtype=np.uint8)
    for k in range(3):
        im_ch = image[:, :, k]
        bins_vec, freq_vec, prob_vec = image_hist(im_ch)
        cdf = np.cumsum(prob_vec)
        low_t = np.where(cdf < percent)
        high_t = np.where(cdf > (1 - percent))
        if (low_t[0].size != 0):  # set darkest pixels to 0
            low_t = low_t[0][-1]
            im_ch[np.where(im_ch < low_t)] = 0
        if (high_t[0].size != 0):  # set brightest pixels to 0 and perform linear contrast stretch
            high_t = high_t[0][0]
            gain = 255 / high_t
            im_ch = im_ch * gain
            im_ch[np.where(im_ch > 255)] = 255

        if powerlaw:    #Apply power law contrast stretch to enhance dark pixels
            im_ch = powerlaw_contrast(im_ch, exp=0.5)
        enhanced_im[:, :, k] = np.array(im_ch, dtype=np.uint8)

        # plot image and histogram before and after saturated contrast stretch
        if plot:
            pyplot.subplot(221, title=f'{ch[k]} channel')
            pyplot.imshow(image[:, :, k], cmap='gray', norm=NoNorm())
            pyplot.subplot(222, title=f'After contrast stretch')
            pyplot.imshow(enhanced_im[:, :, k], cmap='gray', norm=NoNorm())
            pyplot.subplot(223)
            pyplot.plot(bins_vec, prob_vec)
            pyplot.subplot(224)
            bins_vec, freq_vec, prob_vec = image_hist(enhanced_im[:, :, k])
            pyplot.plot(bins_vec, prob_vec)
            pyplot.show()
    return enhanced_im


def resize(image: np.ndarray, scale: int, type='nearest', method='vectorized') -> np.ndarray:
    h, w = image.shape
    resized_im = np.zeros((int((h - 1) * scale), int((w - 1) * scale)))
    if method == 'ref':
        for y in range(resized_im.shape[0]):
            for x in range(resized_im.shape[1]):
                resized_im[y, x] = interpolate(image, x / scale - 1, y / scale - 1, type=type)
    if method == 'vectorized':
        indices = np.indices(resized_im.shape)
        indices = indices / scale
        resized_im = interpolate_vec(image, indices, type=type)
    return np.array(resized_im, dtype=np.uint8)


def rotate(image: np.ndarray, theta: int, type='nearest', method='vectorized') -> np.ndarray:
    h, w = image.shape

    # Compute width, height and offsets for rotated image
    c = math.cos(math.radians(theta))
    s = math.sin(math.radians(theta))
    x_corner = np.array([0, 0, w - 1, w - 1], dtype=float)
    y_corner = np.array([0, h - 1, 0, h - 1], dtype=float)
    # Transforming the rotation matrix from cartesian coordinates to image coordinates
    h_index = h - 1
    r_x_corner = c * x_corner + (y_corner - h) * s
    r_y_corner = h - (s * x_corner + (h - y_corner) * c)
    r_w = int(max(r_x_corner)) - int(min(r_x_corner))
    r_h = int(max(r_y_corner)) - int(min(r_y_corner))
    Kx = int(min(r_x_corner))
    Ky = int(min(r_y_corner))

    r_image = np.zeros((r_h, r_w))

    c = math.cos(math.radians(-theta))
    s = math.sin(math.radians(-theta))
    # For each pixel in rotated image get the intensity from corresponding location in original image
    if method == 'ref':
        for y in range(Ky, Ky + r_h):
            for x in range(Kx, Kx + r_w):
                x_actual = c * x + (y - h) * s
                y_actual = h - (s * x + (h - y) * c)
                if x_actual >= 0 and x_actual < w - 1 and y_actual >= 0 and y_actual < h - 1:  # valid range in original image
                    r_image[y - Ky, x - Kx] = interpolate(image, x_actual, y_actual, type=type)

    if method == 'vectorized':
        indices = np.indices(r_image.shape)
        y = indices[0] + Ky
        x = indices[1] + Kx
        x_actual = c * x + (y - h) * s
        y_actual = h - (s * x + (h - y) * c)
        # valid range of indices in original image
        valid = np.bitwise_and(np.bitwise_and(x_actual >= 0, x_actual < w - 1),
                               np.bitwise_and(y_actual >= 0, y_actual < h - 1))
        valid_indices = np.where(valid != 0)
        if (theta % 90 == 0):
            r_image[valid_indices] = image[
                np.array(y_actual[valid_indices], dtype=int), np.array(x_actual[valid_indices], dtype=int)]
        else:
            r_image[valid_indices] = interpolate_vec(image, [y_actual[valid_indices], x_actual[valid_indices]],
                                                     type=type)
    return np.array(r_image, dtype=np.uint8)
