# Wrapper file to test all functions
import time
from pathlib import Path

import cv2
import numpy as np
import skimage.io
from skimage.filters import rank
from skimage.morphology import square
from matplotlib import pyplot as plt
from matplotlib.colors import NoNorm

from manogna_sreenivas.AllFunctions import highboost_filter, get_sinusoidal_image, freq_domain_filter, homomorphic_filter

noisy_path = Path('../Data/images/noisy.tif')
characters_path = Path('../Data/images/characters.tif')
pet_path = Path('../Data/images/PET_image.tif')



def problem1():
    im_noisy = skimage.io.imread(noisy_path)
    filtered_5 = rank.mean(im_noisy, square(5))
    filtered_10 = rank.mean(im_noisy, square(10))
    filtered_15 = rank.mean(im_noisy, square(15))

    cv2.imshow('Noisy image', im_noisy)
    cv2.imshow('5x5 Mean filtered', filtered_5)
    cv2.imshow('10x10 Mean filtered', filtered_10)
    cv2.imshow('15x15 Mean filter', filtered_15)
    cv2.waitKey(0)

    highboost_5 = highboost_filter(filtered_15,K=2)
    im_characters = skimage.io.imread(characters_path)
    mse = np.sum((highboost_5-im_characters)**2)
    print(mse)
    cv2.imshow('Noisy image', im_noisy)
    cv2.imshow('5x5 High boost filtered', highboost_5)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return


def problem2():
    im_sine = get_sinusoidal_image((1000,1000), (100,200))
    print(np.max(im_sine),np.min(im_sine))
    dft = np.fft.fft2(im_sine)
    dft_shift = np.fft.fftshift(dft)
    mag_spectrum = np.log(np.abs(dft_shift))
    mag_spectrum2 = np.array(np.log(np.abs(dft_shift)),dtype=np.uint8)
    print(np.max(mag_spectrum), np.min(mag_spectrum))
    plt.imshow(im_sine[:100,:100]*127)
    plt.show()
    plt.imshow(mag_spectrum)
    plt.show()
    return

def problem2b():
    im_characters = skimage.io.imread(characters_path)
    ilpf_characters = freq_domain_filter(im_characters, type='Ideal_lowpass')
    gaussian_characters = freq_domain_filter(im_characters, type='Gaussian')
    cv2.imshow('Characters image', im_characters)
    cv2.imshow('Applying Ideal low pass filter', ilpf_characters)
    cv2.imshow('Applying Gaussian filter', gaussian_characters)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def problem3():
    im_pet = skimage.io.imread(pet_path)
    im_pet_filtered = homomorphic_filter(im_pet, param = [2,1,100])  #param = [gH, gL, D0]
    cv2.imshow('PET image', im_pet)
    cv2.imshow('After homomorphic filtering', im_pet_filtered)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return


def problem4():
    g = np.array([[0,0,1,0,0],[0,1,2,1,0],[1,2,-16,2,1],[0,1,2,1,0],[0,0,1,0,0]])

    indices = np.indices((5,5)) - 2
    H = indices[0]**2 + indices[1]**2
    print(indices, H)

    h = np.fft.ifft2(H)
    h = np.fft.fftshift(h)
    h = np.real(h)
    print(h)


    return


def main():
    #problem1()
    #problem2b()
    #problem3()
    problem4()

    return


if __name__ == '__main__':
    main()
