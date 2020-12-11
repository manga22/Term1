import numpy as np


def get_padded_image(bin_image: np.ndarray, kernel: int) -> np.ndarray:
    h, w = bin_image.shape
    pad = int(kernel / 2)
    padded_im = np.ones((h + kernel - 1, w + kernel - 1))
    padded_im[pad:pad + h, pad:pad + w] = bin_image
    # pad rows at the top and bottom
    padded_im[:pad] = padded_im[pad]
    padded_im[-pad:] = padded_im[-pad - 1]
    # pad columns at the left and right end
    padded_im[:, :pad] = np.expand_dims(padded_im[:, pad], 1)
    padded_im[:, -pad:] = np.expand_dims(padded_im[:, -pad - 1], 1)
    return padded_im


def get_distance_matrix(im_size):
    h, w = im_size
    indices = np.indices(im_size)
    D = np.sqrt((indices[0] - h / 2) ** 2 + (indices[1] - w / 2) ** 2)
    return D


def freq_filter(image, H):  #Filtering in frequency domain
    f_image = np.fft.fft2(image)                    #get fft of image
    f_centered = np.fft.fftshift(f_image)           #shift dc value to center
    f_inv_center = np.fft.ifftshift(f_centered * H) #filter and shift back dc value to 0
    im_filtered = np.fft.ifft2(f_inv_center)        #take inverse fft
    im_filtered = np.real(im_filtered)
    return im_filtered


def min_mse(g, h):
    K = np.sum(h * g) / np.sum(h * h)
    return [K, h * K]
