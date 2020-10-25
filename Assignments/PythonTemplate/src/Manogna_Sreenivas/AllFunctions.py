# This file should contain all the functions required by Wrapper.py

from pathlib import Path

import numpy as np
import skimage

from manogna_sreenivas.helper import otsu_min_wclass_var, otsu_max_bclass_var, get_padded_image, \
    get_connected_components


def compute_hist(image_path: Path, num_bins: int) -> list:
    # bins_vec and freq_vec should contain values computed by custom function
    # bins_vec_lib and freq_vec_lib should contain values computed by python library function
    image = skimage.io.imread(image_path)
    image = image.flatten()

    # custom function
    bins = np.linspace(np.amin(image), np.amax(image), num_bins + 1)
    freq_vec = np.zeros(num_bins)
    for k in range(num_bins):
        freq_vec[k] = np.sum(np.bitwise_and(np.greater_equal(image, bins[k]), np.less(image, bins[k + 1])))
    freq_vec[num_bins - 1] += np.sum(np.equal(image, bins[num_bins]))

    # using library function
    freq_vec_lib, bin_edges_lib = np.histogram(image, bins=num_bins)

    # get bin centres from bin edges
    bins_vec = (bins[:num_bins] + bins[1:]) / 2
    bins_vec_lib = (bin_edges_lib[:num_bins] + bin_edges_lib[1:]) / 2

    return [bins_vec, freq_vec, bins_vec_lib, freq_vec_lib]


def otsu_threshold(gray_image_path: Path) -> list:
    gray_image = skimage.io.imread(gray_image_path)
    # Minimizing within class variance
    thr_w, time_w = otsu_min_wclass_var(gray_image)
    # Maximizing between class variance
    thr_b, time_b, bin_image = otsu_max_bclass_var(gray_image)

    return [thr_w, thr_b, time_w, time_b, bin_image]


def change_background(quote_image_path: Path, bg_image_path: Path) -> np.ndarray:
    quote_image = skimage.io.imread(quote_image_path)
    modified_image = skimage.io.imread(bg_image_path)
    # get binary image by otsu thresholding
    _, _, bin_image = otsu_max_bclass_var(quote_image)
    # overlay text on background image
    modified_image[np.where(bin_image == 0)] = 0

    return modified_image


def count_connected_components(gray_image_path: Path) -> int:
    gray_image = skimage.io.imread(gray_image_path.as_posix())
    # get binary image by otsu thresholding
    _, _, bin_image = otsu_max_bclass_var(gray_image)
    # get connected components from binary image
    regions, _, cc_length = get_connected_components(bin_image)
    # Ignore small connected components like punctuations.
    # Threshold of length 100 is used based on the histogram here.
    cc_length[cc_length < 100] = 0
    num_characters = np.count_nonzero(cc_length)

    return num_characters


def binary_morphology(gray_image_path: Path) -> np.ndarray:
    gray_image = skimage.io.imread(gray_image_path.as_posix())
    # get binary image using otsu thresholding and normalize it to [0,1]
    _, _, bin_image = otsu_max_bclass_var(gray_image)
    bin_image = bin_image / 255

    # pad image with boundary pixel values
    k = 3
    padded_im = get_padded_image(bin_image, kernel=k)

    # Majority filter with a 3x3 window is used to remove noise here
    h, w = bin_image.shape
    sum_image = np.zeros(bin_image.shape)
    # At each location, the pixel values within a 3x3 window are summed
    for k_h in range(k):
        for k_w in range(k):
            sum_image += padded_im[k_h:k_h + h, k_w:k_w + w]
    # For a 3x3 window, if the sum at a location is >4, then the majority value is 1, otherwise it is 0.
    cleaned_image = np.zeros(bin_image.shape)
    cleaned_image[np.where(sum_image > 4)] = 255

    return cleaned_image


def count_mser_components(gray_image_path: Path) -> list:

    gray_image = skimage.io.imread(gray_image_path)
    n_pixels = gray_image.shape[0] * gray_image.shape[1]

    black_letters = []  # list to collect lengths of black letters at each threshold
    white_letters = []  # list to collect lengths of white letters at each threshold
    black_threshold_max = -1  # In [0,black_threshold_max] binary image has black foreground, white background
    white_threshold_min = -1  # In [white_threshold_min,255] binary image has white foreground, black background

    for thr in range(256):  # sweep over all thresholds
        bin_image = np.zeros(gray_image.shape, dtype=np.uint8)
        pixels_above_thr = np.where(gray_image > thr)
        white_pixels = len(pixels_above_thr[0])
        black_pixels = n_pixels - white_pixels
        bin_image[pixels_above_thr] = 255
        if black_pixels < white_pixels:  # Identify the background based on pixel count
            black_threshold_max = thr
            foreground = 'black'
        else:
            foreground = 'white'

        # get lengths of connected components and collect them in relevant lists based on the type of foreground
        _, _, cc_length = get_connected_components(bin_image, foreground=foreground)
        if thr <= black_threshold_max:  # connected components are black letters in image
            black_letters.append(cc_length)
        else:                           # connected components are white letters in image
            white_letters.append(cc_length)
    white_threshold_min = black_threshold_max + 1

    '''
    eps = 5, delta = 5
    black thresholds: [128. 144. 137. 144. 113.]
    white thresholds: [250. 238. 221. 250. 227.]
    eps = 3, delta = 5
    black thresholds: [136. 145. 137. 145. 113.]
    white thresholds: [251. 238. 239. 251. 235.]
    '''
    eps = 3
    delta = 5

    n_black = len(black_letters[0])
    black_thresh = -1 * np.ones((n_black))   # array to get stable thresholds for black letters
    black_intensities = len(black_letters)      # relevant thresholds for black letters [0,black_threshold_max]
    for t in range(eps, black_intensities - eps - 1):
        curr_range = np.array(black_letters[t - eps:t + eps + 1])
        curr_len = curr_range[eps - 1]
        max_range = np.max(curr_range, axis=0) - curr_len
        min_range = curr_len - np.min(curr_range, axis=0)
        for k in range(n_black):
            if (max_range[k] < delta and min_range[k] < delta):
                if (t > black_thresh[k]):
                    black_thresh[k] = t

    n_white = len(white_letters[0])
    white_thresh = -1 * np.ones((n_white))   # array to get stable thresholds for black letters
    white_intensities = len(white_letters)      # relevant thresholds for black letters [0,black_threshold_max]
    for t in range(eps, white_intensities - eps - 1):
        curr_range = np.array(white_letters[t - eps:t + eps + 1])
        curr_len = curr_range[eps - 1]
        max_range = np.max(curr_range, axis=0) - curr_len
        min_range = curr_len - np.min(curr_range, axis=0)
        for k in range(n_white):
            if max_range[k] < delta and min_range[k] < delta:
                if t > white_thresh[k]:
                    white_thresh[k] = t
    white_thresh += white_threshold_min

    # Get letters based on the respective stable threshold identified above and overlay on white background
    mser_binary_image = np.ones(gray_image.shape, dtype=np.uint8) * 255
    # Overlay originally black letters
    for k in range(n_black):
        # Get kth connected component from binary image obtained with the stable threshold
        pixels_above_thr = np.where(gray_image > black_thresh[k])
        bin_image = np.zeros(gray_image.shape, dtype=np.uint8)
        bin_image[pixels_above_thr] = 255
        regions, cc_index, cc_length = get_connected_components(bin_image, foreground='black')
        mser_binary_image[np.where(regions == cc_index[k])] = 0

    # Overlay originally white letters
    for k in range(n_white):
        # Get kth connected component from binary image obtained with the stable threshold
        pixels_above_thr = np.where(gray_image > white_thresh[k])
        bin_image = np.zeros(gray_image.shape, dtype=np.uint8)
        bin_image[pixels_above_thr] = 255
        regions, cc_index, cc_length = get_connected_components(bin_image, foreground='white')
        mser_binary_image[np.where(regions == cc_index[k])] = 0

    # get otsu threshold image
    otsu_thr, _, otsu_binary_image = otsu_max_bclass_var(gray_image)
    num_otsu_components = count_connected_components(gray_image_path)
    num_mser_components = n_black + n_white
    return [mser_binary_image, otsu_binary_image, num_mser_components, num_otsu_components]
