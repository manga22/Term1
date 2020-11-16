import numpy as np
import time


def image_hist(image: np.ndarray) -> list:
    # custom function to get histogram for each intensity
    image = image.flatten()
    bins_vec = np.arange(0, 256)
    freq_vec = np.zeros(bins_vec.shape)
    num_bins = bins_vec.shape[0]
    for k in range(num_bins):
        freq_vec[k] = np.sum(np.equal(image, bins_vec[k]))
    prob_vec = freq_vec / (image.shape[0])
    return [bins_vec, freq_vec, prob_vec]


def get_class_prob_mean(bins_vec: np.ndarray, prob_vec: np.ndarray) -> list:
    # Compute class probabilities at each threshold
    class0_prob_t = np.cumsum(prob_vec)
    class1_prob_t = 1 - class0_prob_t
    class0_prob_t += 10 ** (-20)  # adding a small epsilon to avoid error on division by 0
    class1_prob_t += 10 ** (-20)
    # Compute class means at each threshold
    kpk = bins_vec * prob_vec
    cum_kpk_0t = np.cumsum(kpk)
    cum_kpk_1t = cum_kpk_0t[-1] - cum_kpk_0t
    class0_mean_t = cum_kpk_0t / class0_prob_t
    class1_mean_t = cum_kpk_1t / class1_prob_t
    return [class0_prob_t, class1_prob_t, class0_mean_t, class1_mean_t]


def otsu_min_wclass_var(gray_image: np.ndarray) -> list:
    # Get image histogram
    start_time = time.time()
    bins_vec, freq_vec, prob_vec = image_hist(gray_image)
    # Compute class probabilities at each threshold
    class0_prob_t, class1_prob_t, class0_mean_t, class1_mean_t = get_class_prob_mean(bins_vec, prob_vec)
    # Compute class variances at each threshold
    class0_var_t = np.zeros(bins_vec.shape, dtype=float)
    class1_var_t = np.zeros(bins_vec.shape, dtype=float)
    num_bins = bins_vec.shape[0]
    for t in range(num_bins):
        class0_var_t[t] = np.sum(np.square(bins_vec[:t + 1] - class0_mean_t[t]) * prob_vec[:t + 1]) / class0_prob_t[t]
        class1_var_t[t] = np.sum(np.square(bins_vec[t + 1:] - class1_mean_t[t]) * prob_vec[t + 1:]) / class1_prob_t[t]
    # Get within class variance at each threshold
    within_class_var = class0_prob_t * class0_var_t + class1_prob_t * class1_var_t
    # Otsu threshold is the one that minimizes within class variance
    thr_w = bins_vec[np.argmin(within_class_var)]
    end_time = time.time()
    time_w = end_time - start_time
    return [thr_w, time_w]


def otsu_max_bclass_var(gray_image: np.ndarray) -> list:
    # Get image histogram
    start_time = time.time()
    bins_vec, freq_vec, prob_vec = image_hist(gray_image)
    # Compute class probabilities at each threshold
    class0_prob_t, class1_prob_t, class0_mean_t, class1_mean_t = get_class_prob_mean(bins_vec, prob_vec)
    # Get within class variance at each threshold
    total_mean = np.sum(bins_vec * prob_vec)
    between_class_var = class0_prob_t * np.square(class0_mean_t - total_mean) + \
                        class1_prob_t * np.square(class1_mean_t - total_mean)
    # Otsu threshold is the one that maximizes between class variance
    thr_b = bins_vec[np.argmax(between_class_var)]
    end_time = time.time()
    time_b = end_time - start_time

    # get binary image after otsu thresholding
    bin_image = np.zeros(gray_image.shape, dtype=np.uint8)
    bin_image[np.where(gray_image > thr_b)] = 255
    return [thr_b, time_b, bin_image]


def get_padded_image(bin_image: np.ndarray, kernel: int) -> np.ndarray:
    h, w = bin_image.shape
    pad = int(kernel / 2)
    padded_im = np.ones((h + kernel - 1, w + kernel - 1))
    padded_im[pad:pad + h, pad:pad + w] = bin_image
    # pad rows at the top and bottom
    padded_im[:pad] = padded_im[pad]
    padded_im[-pad:] = padded_im[-pad-1]
    # pad columns at the left and right end
    padded_im[:,:pad] = np.expand_dims(padded_im[:,pad], 1)
    padded_im[:,-pad:] = np.expand_dims(padded_im[:,-pad-1], 1)
    return padded_im


def get_connected_components(bin_image: np.ndarray, foreground='black') -> list:
    regions = np.zeros(bin_image.shape)
    count = 0
    left = lambda i, j: bin_image[i - 1, j]
    top = lambda i, j: bin_image[i, j - 1]
    # Get connected components based on black or white foreground
    if foreground == 'black':
        foreground_pixel = 0
        background_pixel = 255
    elif foreground == 'white':
        foreground_pixel = 255
        background_pixel = 0
    for col in range(1, bin_image.shape[0]):
        for row in range(1, bin_image.shape[1]):
            curr = bin_image[col][row]
            if curr == foreground_pixel:
                if left(col, row) == background_pixel and top(col, row) == background_pixel:
                    # new connected component
                    count += 1
                    regions[col][row] = count
                elif left(col, row) == foreground_pixel and top(col, row) == background_pixel:
                    # current pixel belongs to the left pixel's connected component
                    regions[col][row] = regions[col - 1][row]
                elif left(col, row) == background_pixel and top(col, row) == foreground_pixel:
                    # current pixel belongs to the top pixel's connected component
                    regions[col][row] = regions[col][row - 1]
                else:
                    # current pixel belongs to the left and top pixel's connected component
                    regions[col][row] = regions[col - 1][row]
                    # index of left pixel's and top pixel's connected component are made same if not equal
                    if regions[col - 1][row] != regions[col][row - 1]:
                        regions[np.where(regions == regions[col][row - 1])] = regions[col - 1][row]
    # Get histogram of regions array.
    # cc_index contains the IDs and cc_length contains the lengths of the connected components
    cc_index, cc_length, _ = image_hist(regions)
    # non-zero elements in R correspond to connected components.
    cc_index = cc_index[1:]     # ignore 0 as it is the background index
    cc_length = cc_length[1:]
    cc_index = cc_index[np.where(cc_length > 0)]
    cc_length = cc_length[np.where(cc_length > 0)]

    return [regions, cc_index, cc_length]
