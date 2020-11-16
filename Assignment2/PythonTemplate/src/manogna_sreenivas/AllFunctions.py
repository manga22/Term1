# This file should contain all the functions required by Wrapper.py

from pathlib import Path
import time
import skimage
import numpy as np
from matplotlib import pyplot
import math

from manogna_sreenivas.helper import image_hist


def linear_contrast(image: np.ndarray, gain: float) -> np.ndarray:
    enhanced_image = image * gain
    return enhanced_image.astype(np.uint8)


def powerlaw_contrast(image: np.ndarray, exp: float) -> np.ndarray:
    image = image / 255
    enhanced_image = image ** exp
    enhanced_image = np.array(enhanced_image * 255, dtype=np.uint8)
    return enhanced_image


def hist_equalization(image: np.ndarray, clip=None) -> np.ndarray:
    bins_vec, freq_vec, prob_vec = image_hist(image)
    valid_hist = np.where(freq_vec > 0)
    bins_vec = bins_vec[valid_hist]
    prob_vec = prob_vec[valid_hist]
    freq_vec = freq_vec[valid_hist]
    if (clip):
        mass = np.sum(prob_vec[np.where(prob_vec > clip)] - clip)
        prob_vec[np.where(prob_vec > clip)] = clip
        dist = np.where(prob_vec <= clip)
        prob_vec[dist] += mass / (dist[0].shape[0])
        # print(mass, np.cumsum(prob_vec)[-1])
        # bins_vec, freq_vec, prob_vec = image_hist(image)
        # pyplot.plot(bins_vec, prob_vec)
        # pyplot.show()
    cdf_image = np.cumsum(prob_vec)
    enhanced_image = np.zeros(image.shape)
    for k in range(bins_vec.shape[0]):
        enhanced_image[np.where(image == bins_vec[k])] = cdf_image[k]
    return enhanced_image


def clahe(image: np.ndarray, clip=0.015, overlap=True) -> np.ndarray:
    k = 8
    image = image
    h, w = image.shape
    clahe_im = np.zeros(image.shape)
    if (overlap):
        n_x = 4 * int(w / 25 + 1)
        n_y = 4 * int(h / 25 + 1)
        overlap_x = int(n_x / 4)
        overlap_y = int(n_y / 4)
        shift_x = int(3 / 4 * n_x)
        shift_y = int(3 / 4 * n_y)
        # print(n_x, n_y)
        for i in range(8):
            for j in range(8):
                tile = image[i * shift_y:i * shift_y + n_y, j * shift_x:j * shift_x + n_x]
                enhanced_tile = hist_equalization(tile, clip=clip)
                if (j > 0):
                    # pyplot.imshow(enhanced_tile[:,:overlap_x], cmap='gray')
                    # pyplot.show()
                    # pyplot.imshow(clahe_im[i * shift_y:i * shift_y + n_y, j * shift_x:j * shift_x + overlap_x], cmap='gray')
                    # pyplot.show()
                    enhanced_tile[:, :overlap_x] = (enhanced_tile[:, :overlap_x] + clahe_im[
                                                                                   i * shift_y:i * shift_y + n_y,
                                                                                   j * shift_x:j * shift_x + overlap_x]) / 2
                if (i > 0):
                    enhanced_tile[:overlap_y, :] = (enhanced_tile[:overlap_y, :] + clahe_im[
                                                                                   i * shift_y:i * shift_y + overlap_y,
                                                                                   j * shift_x:j * shift_x + n_x]) / 2
                clahe_im[i * shift_y:i * shift_y + n_y, j * shift_x:j * shift_x + n_x] = enhanced_tile
                # pyplot.imshow(clahe_im)
                # pyplot.show()
                print(i * shift_y + n_y, j * shift_x + n_x,
                      clahe_im[i * shift_y:i * shift_y + n_y, j * shift_x:j * shift_x + n_x].shape)
    else:
        n_x = int(w / 8)
        n_y = int(h / 8)
        for i in range(8):
            for j in range(8):
                tile = image[i * n_y:(i + 1) * n_y, j * n_x:(j + 1) * n_x]
                clahe_im[i * n_y:(i + 1) * n_y, j * n_x:(j + 1) * n_x] = hist_equalization(tile, clip=clip)
    clahe_im = np.array(clahe_im * 255, dtype=np.uint8)
    return clahe_im


def saturated_contrast(image: np.ndarray) -> np.ndarray:
    image_r = image[:, :, 0]
    image_g = image[:, :, 1]
    image_b = image[:, :, 2]
    enhanced_im = np.zeros(image.shape, dtype=np.uint8)
    gain = [3, 3, 1.2]
    for k in range(3):
        im_ch = image[:, :, k]
        bins_vec, freq_vec, prob_vec = image_hist(im_ch)
        # print(np.amax(image[:,:,k]), np.amin(image[:,:,k]))
        bright_pixels = np.where(im_ch > 250)
        l = bright_pixels[0].shape[0]
        index = np.random.randint(l - 1, size=int(0.25 * l))
        new = tuple([bright_pixels[0][index], bright_pixels[1][index]])
        print(index)
        im_ch = im_ch * gain[k]
        im_ch[np.where(im_ch >= 255)] = 255
        im_ch[new] = 255
        enhanced_im[:, :, k] = im_ch

        pyplot.subplot(121)
        pyplot.plot(bins_vec, prob_vec)
        print(freq_vec[255])
        pyplot.subplot(122)
        bins_vec, freq_vec, prob_vec = image_hist(enhanced_im[:, :, k])
        print(freq_vec[255])
        pyplot.plot(bins_vec, prob_vec)
        pyplot.show()

        pyplot.subplot(121)
        pyplot.imshow(image[:, :, k], cmap='gray')
        pyplot.subplot(122)
        pyplot.imshow(enhanced_im[:, :, k], cmap='gray')
        pyplot.show()
    pyplot.subplot(121)
    pyplot.imshow(image)
    pyplot.subplot(122)
    pyplot.imshow(enhanced_im)
    pyplot.show()

    return image


def interpolate(image, x, y, type='nearest'):
    if type == 'nearest':
        return image[int(y + 0.5), int(x + 0.5)]
    if type == 'bilinear':
        x1 = int(x)
        y1 = int(y)
        x2 = x1 + 1
        y2 = y1 + 1
        # print(x1,y1,x2,y2)
        a_x1y1 = float(image[y1][x1])
        a_x2y1 = float(image[y1][x2])
        a_x1y2 = float(image[y2][x1])
        a_x2y2 = float(image[y2][x2])
        w = np.array([a_x1y1, (a_x2y1 - a_x1y1) / (x2 - x1), (a_x1y2 - a_x1y1) / (y2 - y1),
                      (a_x2y2 - a_x1y2 - a_x2y1 + a_x1y1) / ((x2 - x1) * (y2 - y1))], dtype=float)
        v = np.array([1, x - x1, y - y1, (x - x1) * (y - y1)])
        return np.dot(w, v)


def resize(image: np.ndarray, scale: int, type='nearest') -> np.ndarray:
    h, w = image.shape
    resized_im = np.zeros((int(h * scale), int(w * scale)))
    print(resized_im.shape)
    for y in range(resized_im.shape[0]):
        for x in range(resized_im.shape[1]):
            resized_im[y, x] = interpolate(image, x / scale - 1, y / scale - 1,
                                           type=type)  # image[int(i / scale + 0.5), int(j / scale + 0.5)]

    return resized_im


def rotate_bottomleft(image: np.ndarray, theta: int, type='nearest') -> np.ndarray:

    c = math.cos(math.radians(theta))
    s = math.sin(math.radians(theta))

    h, w = image.shape

    a1 = lambda x, y: c * x + (y - h) * s
    a2 = lambda x, y: h - (s * x + (h - y) * c)

    # Compute width, height and offsets for rotated image
    width = int(max(a1(0, 0), a1(w, 0), a1(w, h), a1(0, h))) - int(min(a1(0, 0), a1(w, 0), a1(w, h), a1(0, h)))
    height = int(max(a2(0, 0), a2(w, 0), a2(w, h), a2(0, h))) - int(min(a2(0, 0), a2(w, 0), a2(w, h), a2(0, h)))
    Kx = int(min(a1(0, 0), a1(w, 0), a1(w, h), a1(0, h)))
    Ky = int(min(a2(0, 0), a2(w, 0), a2(w, h), a2(0, h)))

    print(width, height, Kx, Ky)
    print(f'x:{a1(0, 0), a1(w, 0), a1(w, h), a1(0, h)}')
    print(f'x:{a2(0, 0), a2(w, 0), a2(w, h), a2(0, h)}')

    r_image = np.zeros((height, width))

    c = math.cos(math.radians(-theta))
    s = math.sin(math.radians(-theta))

    for y in range(Ky, Ky + height):
        for x in range(Kx, Kx + width):
            x_actual = a1(x, y)
            y_actual = a2(x, y)
            if (x_actual >= 0 and x_actual <= w - 1 and y_actual >= 0 and y_actual <= h - 1):
                r_image[y - Ky, x - Kx] = interpolate(image, x_actual - 1, y_actual - 1, type=type)

    return r_image


def rotate_topleft(image: np.ndarray, theta: int, type='nearest') -> np.ndarray:
    a1 = lambda x, y: c * x - y * s
    a2 = lambda x, y: s * x + y * c

    if theta > 270 and theta <360:
        theta = 360-theta

    c = math.cos(math.radians(theta))
    s = math.sin(math.radians(theta))

    h, w = image.shape

    # Compute width, height and offsets for rotated image
    width = int(max(a1(0, 0), a1(w, 0), a1(w, h), a1(0, h))) - int(min(a1(0, 0), a1(w, 0), a1(w, h), a1(0, h)))
    height = int(max(a2(0, 0), a2(w, 0), a2(w, h), a2(0, h))) - int(min(a2(0, 0), a2(w, 0), a2(w, h), a2(0, h)))
    Kx = int(min(a1(0, 0), a1(w, 0), a1(w, h), a1(0, h)))
    Ky = int(min(a2(0, 0), a2(w, 0), a2(w, h), a2(0, h)))

    print(width, height, Kx, Ky)
    print(max(a1(0, 0), a1(w, 0), a1(w, h), a1(0, h)), min(a1(0, 0), a1(w, 0), a1(w, h), a1(0, h)))
    print(f'x:{a1(0, 0), a1(w, 0), a1(w, h), a1(0, h)}')
    print(f'y:{a2(0, 0), a2(w, 0), a2(w, h), a2(0, h)}')

    r_image = np.zeros((height, width))

    c = math.cos(math.radians(-theta))
    s = math.sin(math.radians(-theta))
    for y in range(Ky, Ky + height):
        for x in range(Kx, Kx + width):
            x_actual = a1(x, y)
            y_actual = a2(x, y)
            if (x_actual >= 0 and x_actual <= w - 1 and y_actual >= 0 and y_actual <= h - 1):
                # print(x_actual, y_actual)
                r_image[y - Ky, x - Kx] = interpolate(image, x_actual - 1, y_actual - 1, type=type)

    return r_image


def rotate(image: np.ndarray, theta: int, type='nearest', corner='top_left') -> np.ndarray:
    h, w = image.shape
    if (theta % 90 == 0):
        print('if')
        c = math.cos(math.radians(-theta))
        s = math.sin(math.radians(-theta))
        a1 = lambda x, y: c * x - y * s
        a2 = lambda x, y: s * x + y * c
        r_image = np.zeros((h, w))
        for y in range(h):
            for x in range(w):
                print(x, y, a2(x, y), a1(x, y))
                r_image[y, x] = image[int(a2(x, y)), int(a1(x, y))]
    else:
        if (corner == 'top_left'):
            r_image = rotate_topleft(image, theta=theta, type=type)
        elif (corner == 'bottom_left'):
            r_image = rotate_bottomleft(image, theta=theta, type=type)
    return r_image
