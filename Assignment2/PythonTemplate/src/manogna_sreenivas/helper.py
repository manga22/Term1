import numpy as np
import time


def image_hist(image: np.ndarray, normalize = False) -> list:
    # custom function to get histogram for each intensity
    image = image.flatten()
    bins_vec = np.arange(0, 256)
    freq_vec = np.zeros(bins_vec.shape)
    num_bins = bins_vec.shape[0]
    for k in range(num_bins):
        freq_vec[k] = np.sum(np.equal(image, bins_vec[k]))
    prob_vec = freq_vec / (image.shape[0])
    return [bins_vec, freq_vec, prob_vec]
