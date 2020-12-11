# Wrapper file to test all functions
import time
from pathlib import Path

import cv2
import numpy as np
import skimage.io
from matplotlib import pyplot
from matplotlib.colors import NoNorm

from manogna_sreenivas.AllFunctions import linear_contrast, powerlaw_contrast, hist_equalization, clahe, \
    saturated_contrast, resize, rotate

lowlight1_path = Path('../Data/images/LowLight_1.png')
lowlight2_path = Path('../Data/images/LowLight_2.png')
lowlight3_path = Path('../Data/images/LowLight_3.png')
hazy_path = Path('../Data/images/Hazy.png')
mathbooks_path = Path('../Data/images/MathBooks.png')
stoneface_path = Path('../Data/images/StoneFace.png')


def problem1():
    lowlight1 = skimage.io.imread(lowlight1_path)
    lowlight2 = skimage.io.imread(lowlight2_path)
    lowlight3 = skimage.io.imread(lowlight3_path)
    hazy = skimage.io.imread(hazy_path)
    mathbooks = skimage.io.imread(mathbooks_path)
    stoneface = skimage.io.imread(stoneface_path)

    # Linear contrast stretching
    lin_lowlight1 = linear_contrast(lowlight1, 255.0 / np.amax(lowlight1))
    lin_lowlight2 = linear_contrast(lowlight2, 255.0 / np.amax(lowlight2))

    # Power law contrast stretching
    pow_lowlight1 = powerlaw_contrast(lowlight1, 0.5)
    pow_lowlight2 = powerlaw_contrast(lowlight2, 0.6)
    pow_hazy = powerlaw_contrast(hazy, 1.5)

    # Histogram equalization
    hist_lowlight2 = hist_equalization(lowlight2, plot=True)       #Set plot=True to view histograms of images
    hist_lowlight3 = hist_equalization(lowlight3, plot=True)
    hist_hazy = hist_equalization(hazy, plot=True)
    hist_stoneface = hist_equalization(stoneface, plot=True)

    # CLAHE
    clahe_stoneface = clahe(stoneface, clip = 0.015, overlap=False)
    clahe_stoneface_overlap = clahe(stoneface, clip = 0.015)

    # Lowlight1
    pyplot.subplot(131, title='Original')
    pyplot.imshow(lowlight1, cmap='gray', norm=NoNorm())
    pyplot.subplot(132, title='Linear contrast')
    pyplot.imshow(lin_lowlight1, cmap='gray', norm=NoNorm())
    pyplot.subplot(133, title='Power law contrast')
    pyplot.imshow(pow_lowlight1, cmap='gray', norm=NoNorm())
    pyplot.show()

    # Lowlight2
    pyplot.subplot(221, title='Original')
    pyplot.imshow(lowlight2, cmap='gray', norm=NoNorm())
    pyplot.subplot(222, title='Linear contrast')
    pyplot.imshow(lin_lowlight2, cmap='gray', norm=NoNorm())
    pyplot.subplot(223, title='Power law contrast')
    pyplot.imshow(pow_lowlight2, cmap='gray', norm=NoNorm())
    pyplot.subplot(224, title='Histogram Equalization')
    pyplot.imshow(hist_lowlight2, cmap='gray', norm=NoNorm())
    pyplot.show()

    # Hazy
    pyplot.subplot(131, title='Original')
    pyplot.imshow(hazy, cmap='gray', norm=NoNorm())
    pyplot.subplot(132, title='Power law contrast')
    pyplot.imshow(pow_hazy, cmap='gray', norm=NoNorm())
    pyplot.subplot(133, title='Histogram Equalization')
    pyplot.imshow(hist_hazy, cmap='gray', norm=NoNorm())
    pyplot.show()

    # Stoneface
    pyplot.subplot(221, title='Original')
    pyplot.imshow(stoneface, cmap='gray', norm=NoNorm())
    pyplot.subplot(222, title='Histogram Equalization')
    pyplot.imshow(hist_stoneface, cmap='gray', norm=NoNorm())
    pyplot.subplot(223, title='CLAHE')
    pyplot.imshow(clahe_stoneface, cmap='gray', norm=NoNorm())
    pyplot.subplot(224, title='CLAHE with overlap')
    pyplot.imshow(clahe_stoneface_overlap, cmap='gray', norm=NoNorm())
    pyplot.show()
    return


def problem2():
    mathbooks = skimage.io.imread(mathbooks_path)
    enhanced_im1 = saturated_contrast(mathbooks, percent = 0.1)
    enhanced_im2 = saturated_contrast(mathbooks, percent=0.1, plot=False, powerlaw=True)
    pyplot.subplot(131, title='Mathbooks image')
    pyplot.imshow(mathbooks, norm=NoNorm())
    pyplot.subplot(132, title='After saturated contrast stretch')
    pyplot.imshow(enhanced_im1, norm=NoNorm())
    pyplot.subplot(133, title='Applying power law')
    pyplot.imshow(enhanced_im2, norm=NoNorm())
    pyplot.show()
    return


def problem3():
    image = skimage.io.imread(stoneface_path)
    scale = 3

    #Call vectorized implementation
    start = time.time()
    resized_nearest = resize(image, scale=scale, type='nearest')
    end = time.time()
    print(f'Time taken to resize by nearest neighbour and vectorized code:{end-start}')

    start = time.time()
    resized_bilinear = resize(image, scale=scale, type='bilinear')
    end = time.time()
    print(f'Time taken to resize by bilinear interpolation and vectorized code:{end-start}')

    ''' Call function with for loops
    start = time.time()
    resized_nearest_ref = resize(image, scale=scale, type='nearest', method='ref')
    end = time.time()
    print(f'Time taken to resize by nearest neighbour and ref code:{end-start}')
    
    start = time.time()
    resized_bilinear_ref = resize(image, scale=scale, type='bilinear', method='ref')
    end = time.time()
    print(f'Time taken to resize by bilinear interpolation and ref code:{end-start}')
    '''

    skimage.io.imsave(Path('../Data/outputs/resize.png').as_posix(), image)
    skimage.io.imsave(Path('../Data/outputs/resize_near_3.png').as_posix(), resized_nearest)
    skimage.io.imsave(Path('../Data/outputs/resize_bilinear_3.png').as_posix(), resized_bilinear)

    cv2.imshow('Image before resize', image)
    cv2.imshow('Resize using nearest neighbour',resized_nearest)
    cv2.imshow('Resize using bilinear interpolation', resized_bilinear)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return


def problem4():
    image = skimage.io.imread(stoneface_path)
    theta = 60

    #Call Vectorized implementations
    start = time.time()
    rotated_nearest = rotate(image,theta=theta,type='nearest')
    end = time.time()
    print(f'Time taken to rotate by nearest neighbour and vectorized code:{end-start}')

    start = time.time()
    rotated_bilinear = rotate(image,theta=theta,type='bilinear')
    end = time.time()
    print(f'Time taken to rotate by bilinear interpolation and vectorized code:{end-start}')

    ''' Call function with for loops 
    start = time.time()
    rotated_nearest_ref = rotate(image,theta=theta,type='nearest', method='ref')
    end = time.time()
    print(f'Time taken to rotate by nearest neighbour and ref code:{end-start}')
    
    start = time.time()
    rotated_bilinear_ref = rotate(image,theta=theta,type='bilinear', method='ref')
    end = time.time()
    print(f'Time taken to rotate by bilinear interpolation and ref code:{end-start}')
    '''

    skimage.io.imsave(Path('../Data/outputs/rotate.png').as_posix(), image)
    skimage.io.imsave(Path('../Data/outputs/rotate_near_60.png').as_posix(), rotated_nearest)
    skimage.io.imsave(Path('../Data/outputs/rotate_bilinear_60.png').as_posix(), rotated_bilinear)

    cv2.imshow('Image before resize', image)
    cv2.imshow('Rotation with nearest neighbour',rotated_nearest)
    cv2.imshow('Rotation with bilinear interpolation', rotated_bilinear)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return


def main():
    problem1()
    problem2()
    problem3()
    problem4()

    return


if __name__ == '__main__':
    main()
