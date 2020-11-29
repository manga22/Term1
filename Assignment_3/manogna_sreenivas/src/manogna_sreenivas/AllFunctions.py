# This file should contain all the functions required by Wrapper.py

import numpy as np
from skimage.filters import rank
from skimage.morphology import square


def highboost_filter(image: np.ndarray, K: float) ->np.ndarray:
    im_blur = rank.mean(image, square(3))
    print(type(im_blur[0][0]))
    im_filtered = (1+K)*image - K*im_blur

    return np.array(im_filtered, dtype=np.uint8)


def get_sinusoidal_image(size: tuple, freq: tuple):
    indices = np.indices(size)
    M = size[0]
    N = size[1]
    u_0 = freq[0]
    v_0 = freq[1]
    theta = 2*np.pi*((u_0 * indices[0])/M + (v_0 * indices[1])/N)
    im_sine = np.sin(theta)
    return im_sine


def get_distance_matrix(image):
    h, w = image.shape
    indices = np.indices(image.shape)
    D = np.sqrt((indices[0]-h/2)**2 + (indices[1]-w/2)**2)
    return  D


def filter(image, H):
    dft = np.fft.fft2(image)                        #Get DFT of image
    dft_shift = np.fft.fftshift(dft)                #Center the DFT
    filt_dft_shifted = dft_shift * H                #Filter in frequency domain
    filt_dft = np.fft.ifftshift(filt_dft_shifted)   #Undo the centering in DFT
    filtered_im = np.fft.ifft2(filt_dft)            #Get filtered image computing inverse dft
    filtered_im = np.abs(filtered_im)
    return  filtered_im


def freq_domain_filter(image:np.ndarray, type = 'Gaussian', D0=100) ->np.ndarray:
    D = get_distance_matrix(image)

    #Get filter coefficients H(u,v)
    H = np.zeros(image.shape)
    if type == 'Ideal_lowpass':
        H[np.where(D<=D0)]=1
    elif type == 'Gaussian':
        H = np.exp(-1*(D**2)/(2*D0**2))

    #Get image after filtering in frequency domain
    filtered_im = filter(image, H)

    return filtered_im


def homomorphic_filter(image:np.ndarray, param = [2,1,100]) ->np.ndarray:
    D = get_distance_matrix(image)

    #Get filter coefficients H(u,v)
    gH, gL, D0 = param
    H = (gH - gL)*(1-np.exp(-1*(D**2)/(2*D0**2))) + gL

    #Transform image to log domain
    image = np.array(image, dtype=np.float)
    log_image = np.log(image+1)

    #Get image after filtering in frequency domain
    filtered_im = filter(image, H)

    #Get actual pixel values from log domain using exp
    filtered_im = np.exp(filtered_im) - 1

    return filtered_im
