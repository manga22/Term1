# This file should contain all the functions required by Wrapper.py

from pathlib import Path

import numpy
import skimage
import time


def compute_hist(image_path: Path, num_bins: int) -> list:
    # bins_vec and freq_vec should contain values computed by custom function
    # bins_vec_lib and freq_vec_lib should contain values computed by python library function
    image = skimage.io.imread(image_path)
    image = image.flatten()

    # custom function
    bins = numpy.linspace(numpy.amin(image), numpy.amax(image), num_bins + 1)
    freq_vec = numpy.zeros(num_bins)
    for k in range(num_bins):
        freq_vec[k] = numpy.sum(numpy.bitwise_and(numpy.greater_equal(image, bins[k]), numpy.less(image, bins[k + 1])))
    freq_vec[num_bins - 1] += numpy.sum(numpy.equal(image, bins[num_bins]))

    # using library function
    freq_vec_lib, bin_edges_lib = numpy.histogram(image, bins=num_bins)

    # get bin centres from bin edges
    bins_vec = (bins[:num_bins] + bins[1:]) / 2
    bins_vec_lib = (bin_edges_lib[:num_bins] + bin_edges_lib[1:]) / 2

    # print(bins, bins_vec, freq_vec)
    return [bins_vec, freq_vec, bins_vec_lib, freq_vec_lib]


def otsu_hist(image: numpy.ndarray, custom: bool) -> list:
    image = image.flatten()

    # custom function to get histogramG
    bins_vec = numpy.arange(numpy.amin(image), numpy.amax(image) + 1)
    freq_vec = numpy.zeros(bins_vec.shape)
    num_bins = bins_vec.shape[0]
    for k in range(num_bins):
        freq_vec[k] = numpy.sum(numpy.equal(image, bins_vec[k]))
    prob_vec = freq_vec / (image.shape[0])
    return [bins_vec, freq_vec, prob_vec]


def otsu_threshold(gray_image_path: Path) -> list:
    thr_w = thr_b = time_w = time_b = 0

    gray_image = skimage.io.imread(gray_image_path)
    bins_vec, freq_vec, prob_vec = otsu_hist(gray_image, custom=True)

    # Compute class probabilities at each threshold
    class0_prob_t = numpy.cumsum(prob_vec)
    class1_prob_t = 1 - class0_prob_t

    # Compute class means at each threshold
    kpk = bins_vec * prob_vec
    cum_kpk_0t = numpy.cumsum(kpk)
    cum_kpk_1t = cum_kpk_0t[-1] - cum_kpk_0t
    class0_mean_t = cum_kpk_0t / class0_prob_t
    class1_mean_t = cum_kpk_1t / class1_prob_t

    # Minimizing within class variance

    # Compute Class variance at each threshold
    start_time = time.time()
    class0_var_t = numpy.zeros(bins_vec.shape, dtype=float)
    class1_var_t = numpy.zeros(bins_vec.shape, dtype=float)
    num_bins = bins_vec.shape[0]
    for t in range(num_bins):
        class0_var_t[t] = numpy.sum(numpy.square(bins_vec[:t + 1] - class0_mean_t[t]) * prob_vec[:t + 1]) / \
                          class0_prob_t[t]
        class1_var_t[t] = numpy.sum((numpy.square(bins_vec[t + 1:] - class1_mean_t[t]) * prob_vec[t + 1:])) / \
                          class1_prob_t[t]
    within_class_var = class0_prob_t * class0_var_t + class1_prob_t * class1_var_t  # within class variance at each threshold
    thr_w = bins_vec[numpy.argmin(within_class_var)]
    end_time = time.time()
    time_w = end_time - start_time

    # Maximizing between class variance
    start_time = time.time()
    total_mean_ref = numpy.sum(bins_vec * prob_vec)
    between_class_var = class0_prob_t * numpy.square(class0_mean_t - total_mean_ref) + class1_prob_t * numpy.square(
        class1_mean_t - total_mean_ref)
    thr_b = bins_vec[numpy.argmax(between_class_var)]
    end_time = time.time()
    time_b = end_time - start_time

    # Verify mean, variance computation
    total_mean = class0_mean_t * class0_prob_t + class1_mean_t * class1_prob_t
    total_mean_ref = numpy.sum(bins_vec * prob_vec)
    total_var = within_class_var + between_class_var
    total_var_ref = numpy.sum(numpy.square(bins_vec - total_mean_ref) * prob_vec)
    assert numpy.all(numpy.isclose(total_var, total_var_ref))
    assert numpy.all(numpy.isclose(total_mean, total_mean_ref))

    #get binary image
    bin_image = numpy.zeros(gray_image.shape)
    bin_image[numpy.where(gray_image > thr_b)] = 255

    return [thr_w, thr_b, time_w, time_b, bin_image]


def change_background(quote_image_path: Path, bg_image_path: Path) -> numpy.ndarray:
    quote_image = skimage.io.imread(quote_image_path)
    bg_image = skimage.io.imread(bg_image_path)

    #get binary image by otsu thresholding
    _,_,_,_,bin_image = otsu_threshold(quote_image_path)

    #overlay text on background image
    modified_image = bg_image
    modified_image[numpy.where(bin_image==0)] = 0
    return modified_image


def count_connected_components(gray_image_path: Path) -> int:
    num_characters = 0
    return num_characters


def binary_morphology(gray_image_path: Path) -> numpy.ndarray:
    cleaned_image = None
    return cleaned_image


def count_mser_components(gray_image_path: Path) -> list:
    mser_binary_image = None
    otsu_binary_image = None
    num_mser_components = 0
    num_otsu_components = 0
    return [mser_binary_image, otsu_binary_image, num_mser_components, num_otsu_components]
