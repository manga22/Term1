# Wrapper file to test all functions
from pathlib import Path

import numpy as np
import skimage.io
from matplotlib import pyplot as plt
from skimage import img_as_float
from skimage.filters import rank
from skimage.morphology import square

from manogna_sreenivas.AllFunctions import average_filter, highboost_filter, get_sinusoidal_image, \
    freq_domain_filter, homomorphic_filter
from manogna_sreenivas.helper import min_mse

noisy_path = Path('../Data/images/noisy.tif')
characters_path = Path('../Data/images/characters.tif')
pet_path = Path('../Data/images/PET_image.tif')


def problem1():
    im_noisy = skimage.io.imread(noisy_path)

    # Square average filtering using mask size 5, 10, 15
    filtered_5 = average_filter(im_noisy, 5)
    filtered_10 = average_filter(im_noisy, 10)
    filtered_15 = average_filter(im_noisy, 15)

    plt.subplot(121, title='Original image')
    plt.imshow(im_noisy, cmap='gray', vmin=0, vmax=255, interpolation=None)
    plt.subplot(122, title='5x5 Mask')
    plt.imshow(filtered_5, cmap='gray', vmin=0, vmax=255, interpolation=None)
    plt.show()

    plt.subplot(121, title='10x10 Mask')
    plt.imshow(filtered_10, cmap='gray', vmin=0, vmax=255, interpolation=None)
    plt.subplot(122, title='15x15 Mask')
    plt.imshow(filtered_15, cmap='gray', vmin=0, vmax=255, interpolation=None)
    plt.show()

    im_characters = skimage.io.imread(characters_path)
    im_characters = img_as_float(im_characters)

    # Sweep over (0,2) to get best K for highboost filtering
    K = np.arange(0, 2, 0.1)
    mse = np.zeros(K.shape)
    image = filtered_5
    for i in range(K.shape[0]):
        im_highboost = highboost_filter(image, k=K[i])
        mse[i] = np.sum((im_highboost - im_characters) ** 2)

    mse = mse / (image.shape[0] * image.shape[1])
    K_best = K[np.argmin(mse)]
    im_highboost = highboost_filter(image, k=K_best)

    print(f'Best K obtained for high boost filtering is {K_best}')

    # Plot ref and output images, MSE vs K
    plt.subplot(131, title='Input denoised image')
    plt.imshow(img_as_float(image), cmap='gray', vmin=0, vmax=1, interpolation=None)
    plt.subplot(132, title=f'Highboost filtered image K = {K_best}')
    plt.imshow(im_highboost, cmap='gray', vmin=0, vmax=1, interpolation=None)
    plt.subplot(133, title='Reference image')
    plt.imshow(im_characters, cmap='gray', vmin=0, vmax=1, interpolation=None)

    plt.show()
    plt.plot(K, mse)
    plt.title('% MSE Vs K')
    plt.show()

    return


def problem2():
    im_sine = get_sinusoidal_image((1001, 1001), (100, 200))

    # Perform fft and get magnitude frequency spectrum
    dft = np.fft.fft2(im_sine)
    dft_shift = np.fft.fftshift(dft)

    # Apply log and normalize to visualize fft
    mag_spectrum = np.log(1 + np.abs(dft_shift))
    print(f'Peaks in magnitude fourier spectrum are at {np.where(mag_spectrum == np.max(mag_spectrum))}')
    M = np.max(mag_spectrum)
    m = np.min(mag_spectrum)
    mag_spectrum = 255 / (M - m) * (mag_spectrum - m)
    mag_spectrum = np.array(mag_spectrum, dtype=np.uint8)

    # Plot sinusoidal image and its fft
    plt.subplot(121, title=f'Sinusoidal image (u,v) = {100, 200}')
    plt.imshow(im_sine, cmap='gray', interpolation=None)
    plt.subplot(122, title='Magnitude spectrum')
    plt.imshow(mag_spectrum, cmap='gray', interpolation=None)
    plt.show()

    # Filtering in frequency domain
    im_characters = skimage.io.imread(characters_path)
    ilpf_characters = freq_domain_filter(im_characters, type='Ideal_lowpass')
    gaussian_characters = freq_domain_filter(im_characters, type='Gaussian')
    plt.subplot(121, title='Ideal low pass filter')
    plt.imshow(ilpf_characters, cmap='gray', vmin=0, vmax=255, interpolation=None)
    plt.subplot(122, title='Gaussian low pass filter')
    plt.imshow(gaussian_characters, cmap='gray', vmin=0, vmax=255, interpolation=None)
    plt.show()
    return


def problem3():
    im_pet = skimage.io.imread(pet_path)
    param = [2, 0.5, 25]  # param = [gH,gL,D0]
    im_pet_filtered1 = homomorphic_filter(im_pet, param=param)
    plt.subplot(121, title='Input image')
    plt.imshow(im_pet, cmap='gray', vmin=0, vmax=255, interpolation=None)
    plt.subplot(122, title=f'Homomorphic filtered image\n [gH,gL,D0] ={param}')
    plt.imshow(im_pet_filtered1, cmap='gray', vmin=0, vmax=255, interpolation=None)
    plt.show()
    return


def problem4():
    # Given Spatial discrete laplacian filters
    g1 = np.array([[0, 0, 1, 0, 0], [0, 1, 2, 1, 0], [1, 2, -16, 2, 1], [0, 1, 2, 1, 0], [0, 0, 1, 0, 0]])
    g2 = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, -24, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])

    # Given Laplacian in frequency domain H(u,v)
    indices = np.indices((5, 5)) - 2
    H = indices[0] ** 2 + indices[1] ** 2

    # Get spatial representation of H(u,v) by getting dft and centering it
    h = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(H)))
    h = h.real  # h(m,n) for -2 <= m<= 2, -2 <= n <= 2

    #Get K that minimizes total mean squared error between g(m,n) and k*h(m,n)
    K1, H1 = min_mse(g1, h)
    K2, H2 = min_mse(g2, h)

    print(f'Best k for (4a): {K1}')
    print(f'h(m,n) = {H1}')
    print(f'Best k for (4a): {K2}')
    print(f'h(m,n) = {H2}')
    return


def main():
    problem1()
    problem2()
    problem3()
    problem4()
    return


if __name__ == '__main__':
    main()
