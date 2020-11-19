import numpy as np


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


def clip_histogram(prob_vec, clip):
    if np.all(prob_vec <= clip):
        return prob_vec
    else:
        mass = np.sum(prob_vec[np.where(prob_vec > clip)] - clip)
        prob_vec[np.where(prob_vec > clip)] = clip
        valid_range = np.where(prob_vec < clip)
        prob_vec[valid_range] += mass / (valid_range[0].shape[0]+10**(-15))
        prob_vec = clip_histogram(prob_vec, clip)
    return prob_vec


def interpolate(image, x, y, type='nearest'):
    if type == 'nearest':
        return image[int(y + 0.5), int(x + 0.5)]
    if type == 'bilinear':
        x1 = int(x)
        y1 = int(y)
        x2 = x1 + 1
        y2 = y1 + 1
        a_x1y1 = float(image[y1][x1])
        a_x2y1 = float(image[y1][x2])
        a_x1y2 = float(image[y2][x1])
        a_x2y2 = float(image[y2][x2])
        w = np.array([a_x1y1, (a_x2y1 - a_x1y1) / (x2 - x1), (a_x1y2 - a_x1y1) / (y2 - y1),
                      (a_x2y2 - a_x1y2 - a_x2y1 + a_x1y1) / ((x2 - x1) * (y2 - y1))], dtype=float)
        v = np.array([1, x - x1, y - y1, (x - x1) * (y - y1)])
        return np.dot(w, v)


def interpolate_vec(image: np.ndarray, indices: list, type='nearest') -> np.ndarray:
    if (type == 'nearest'):
        interp_vec = image[np.array(indices[0] + 0.5, dtype=int), np.array(indices[1] + 0.5, dtype=int)]
    elif (type == 'bilinear'):
        y, x = indices
        y1, x1 = [np.array(indices[0], dtype=int), np.array(indices[1], dtype=int)]
        y2 = y1 + 1
        x2 = x1 + 1
        a_x1y1 = np.array(image[y1, x1], dtype=float)
        a_x2y1 = np.array(image[y1, x2], dtype=float)
        a_x1y2 = np.array(image[y2, x1], dtype=float)
        a_x2y2 = np.array(image[y2, x2], dtype=float)
        interp_vec = a_x1y1 + ((a_x2y1 - a_x1y1) / (x2 - x1)) * (x - x1) + ((a_x1y2 - a_x1y1) / (y2 - y1)) * (y - y1) + \
                     ((a_x2y2 - a_x1y2 - a_x2y1 + a_x1y1) / ((x2 - x1) * (y2 - y1))) * (x - x1) * (y - y1)
    return interp_vec

