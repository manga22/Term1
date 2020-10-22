# This file should contain all the functions required by Wrapper.py

from pathlib import Path

import numpy
import skimage
import time

from matplotlib import pyplot


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


def otsu_hist(image: numpy.ndarray) -> list:
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
    bins_vec, freq_vec, prob_vec = otsu_hist(gray_image)
    # Compute class probabilities at each threshold
    class0_prob_t = numpy.cumsum(prob_vec)
    class1_prob_t = 1 - class0_prob_t

    # Compute class means at each threshold
    kpk = bins_vec * prob_vec
    cum_kpk_0t = numpy.cumsum(kpk)
    cum_kpk_1t = cum_kpk_0t[-1] - cum_kpk_0t
    class0_mean_t = cum_kpk_0t / (class0_prob_t + 10 ** (-20))
    class1_mean_t = cum_kpk_1t / (class1_prob_t + 10 ** (-20))

    # Minimizing within class variance

    # Compute Class variance at each threshold
    start_time = time.time()
    class0_var_t = numpy.zeros(bins_vec.shape, dtype=float)
    class1_var_t = numpy.zeros(bins_vec.shape, dtype=float)
    num_bins = bins_vec.shape[0]
    for t in range(num_bins):
        class0_var_t[t] = numpy.sum(numpy.square(bins_vec[:t + 1] - class0_mean_t[t]) * prob_vec[:t + 1]) / \
                          (class0_prob_t[t] + 10 ** (-20))
        # print(bins_vec, class0_prob_t[t], class1_prob_t[t])
        class1_var_t[t] = numpy.sum((numpy.square(bins_vec[t + 1:] - class1_mean_t[t]) * prob_vec[t + 1:])) / \
                          (class1_prob_t[t] + 10 ** (-20))
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

    # get binary image
    bin_image = numpy.zeros(gray_image.shape, dtype=numpy.uint8)
    bin_image[numpy.where(gray_image > thr_b)] = 255

    return [thr_w, thr_b, time_w, time_b, bin_image]


def change_background(quote_image_path: Path, bg_image_path: Path) -> numpy.ndarray:
    quote_image = skimage.io.imread(quote_image_path)
    bg_image = skimage.io.imread(bg_image_path)

    # get binary image by otsu thresholding
    _, _, _, _, bin_image = otsu_threshold(quote_image_path)

    # overlay text on background image
    modified_image = bg_image
    modified_image[numpy.where(bin_image == 0)] = 0
    return modified_image


def count_connected_components(gray_image_path: Path) -> int:
    # get binary image by otsu thresholding
    _, _, _, _, bin_image = otsu_threshold(gray_image_path)
    gray_image = skimage.io.imread(gray_image_path.as_posix())
    pyplot.subplot(121)
    pyplot.imshow(gray_image, cmap='gray')
    pyplot.subplot(122)
    pyplot.imshow(bin_image, cmap='gray')
    pyplot.show()

    R = numpy.zeros(bin_image.shape)
    count = 0
    repeated = 0
    left = lambda i, j: bin_image[i - 1, j]
    top = lambda i, j: bin_image[i, j - 1]
    for col in range(1, bin_image.shape[0]):
        for row in range(1, bin_image.shape[1]):
            curr = bin_image[col][row]
            if curr == 0:
                if left(col, row) == 255 and top(col, row) == 255:
                    count += 1
                    R[col][row] = count
                elif left(col, row) == 0 and top(col, row) == 255:
                    R[col][row] = R[col - 1][row]
                elif left(col, row) == 255 and top(col, row) == 0:
                    R[col][row] = R[col][row - 1]
                else:
                    R[col][row] = R[col - 1][row]
                    if R[col - 1][row] != R[col][row - 1]:
                        R[numpy.where(R == R[col][row - 1])] = R[col - 1][row]
                        repeated += 1
    num_characters = count - repeated
    cc_index, cc_length, _ = otsu_hist(R)
    cc_index = cc_index[1:]
    cc_length = cc_length[1:]
    cc_length[cc_length < 100] = 0
    #pyplot.plot(cc_index, cc_length)
    #pyplot.show()
    num_characters = numpy.count_nonzero(cc_length)

    return num_characters


def binary_morphology(gray_image_path: Path) -> numpy.ndarray:
    _, _, _, _, bin_image = otsu_threshold(gray_image_path)
    bin_image = bin_image / 255
    h, w = bin_image.shape

    k = 3
    pad = int(k / 2)
    padded_im = numpy.ones((h + k - 1, w + k - 1))
    padded_im[pad:pad + h, pad:pad + w] = bin_image
    start = time.time()
    sum_image = numpy.zeros(bin_image.shape)
    for k_h in range(k):
        for k_w in range(k):
            sum_image += padded_im[k_h:k_h + h, k_w:k_w + w]
    cleaned_image = numpy.zeros(bin_image.shape)
    cleaned_image[numpy.where(sum_image > 4)] = 255

    return cleaned_image


def count_mser_components(gray_image_path: Path) -> list:
    # get binary image by otsu thresholding

    _, _, _, _, bin_image = otsu_threshold(gray_image_path)
    gray_image = skimage.io.imread(gray_image_path.as_posix())
    n_pixels = gray_image.shape[0] * gray_image.shape[1]
    pyplot.subplot(121)
    pyplot.imshow(gray_image, cmap='gray')
    pyplot.subplot(122)
    pyplot.imshow(bin_image, cmap='gray')
    #pyplot.show()
    hist, bin_edges = numpy.histogram(gray_image, bins=numpy.arange(256))
    pyplot.plot(bin_edges[:145], hist[:145])
    #pyplot.show()
    # print(hist, bin_edges)
    lengths_at_t = numpy.zeros((255,5))
    for thr in range(255): #[0, 1, 10, 140, 145, 149, 150, 175, 250]

        pixels_above_thr = numpy.where(gray_image > thr)
        white_pixels = len(pixels_above_thr[0])
        black_pixels = n_pixels - white_pixels
        if black_pixels < white_pixels:
            bin_image = numpy.zeros(gray_image.shape, dtype=numpy.uint8)
            bin_image[pixels_above_thr] = 1
        else:
            bin_image = numpy.ones(gray_image.shape, dtype=numpy.uint8)
            bin_image[pixels_above_thr] = 0
        '''
        pyplot.subplot(121)
        pyplot.imshow(gray_image, cmap='gray')
        pyplot.subplot(122)
        pyplot.imshow(bin_image, cmap='gray')
        pyplot.title(f"threshold: {thr}")
        pyplot.show()
        '''
        R = numpy.zeros(bin_image.shape)
        count = 0
        repeated = 0
        left = lambda i, j: bin_image[i - 1, j]
        top = lambda i, j: bin_image[i, j - 1]
        for col in range(1, bin_image.shape[0]):
            for row in range(1, bin_image.shape[1]):
                curr = bin_image[col][row]
                if curr == 0:
                    if left(col, row) == 1 and top(col, row) == 1:
                        count += 1
                        R[col][row] = count
                    elif left(col, row) == 0 and top(col, row) == 1:
                        R[col][row] = R[col - 1][row]
                    elif left(col, row) == 1 and top(col, row) == 0:
                        R[col][row] = R[col][row - 1]
                    else:
                        R[col][row] = R[col - 1][row]
                        if R[col - 1][row] != R[col][row - 1]:
                            R[numpy.where(R == R[col][row - 1])] = R[col - 1][row]
                            repeated += 1
        cc_index, cc_length, _ = otsu_hist(R)
        cc_index = cc_index[1:]
        cc_length = cc_length[1:]
        cc_index = cc_index[numpy.where(cc_length > 0)]
        cc_length = cc_length[numpy.where(cc_length > 0)]
        print(thr, count - repeated, cc_index, cc_length)
        lengths_at_t[thr] = cc_length


    pyplot.subplot(231)
    pyplot.plot(numpy.arange(150),lengths_at_t[:150,0])
    pyplot.subplot(232)
    pyplot.plot(numpy.arange(150),lengths_at_t[:150,1])
    pyplot.subplot(233)
    pyplot.plot(numpy.arange(150),lengths_at_t[:150,2])
    pyplot.subplot(234)
    pyplot.plot(numpy.arange(150),lengths_at_t[:150,3])
    pyplot.subplot(235)
    pyplot.plot(numpy.arange(150),lengths_at_t[:150,4])
    pyplot.show()

    pyplot.subplot(231)
    pyplot.plot(numpy.arange(150,255),lengths_at_t[150:,0])
    pyplot.subplot(232)
    pyplot.plot(numpy.arange(150,255),lengths_at_t[150:,1])
    pyplot.subplot(233)
    pyplot.plot(numpy.arange(150,255),lengths_at_t[150:,2])
    pyplot.subplot(234)
    pyplot.plot(numpy.arange(150,255),lengths_at_t[150:,3])
    pyplot.subplot(235)
    pyplot.plot(numpy.arange(150,255),lengths_at_t[150:,4])
    pyplot.show()


    mser_binary_image = bin_image
    _, _, _, _, otsu_binary_image = otsu_threshold(gray_image_path)
    num_mser_components = 0
    num_otsu_components = count_connected_components(gray_image_path)
    return [mser_binary_image, otsu_binary_image, num_mser_components, num_otsu_components]
