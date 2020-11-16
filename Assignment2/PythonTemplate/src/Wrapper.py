# Wrapper file to test all functions

from pathlib import Path

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
    pow_lowlight2 = powerlaw_contrast(lowlight2, 0.5)
    pow_hazy = powerlaw_contrast(hazy, 2.5)

    # Histogram equalization
    hist_lowlight2 = hist_equalization(lowlight2)
    hist_lowlight3 = hist_equalization(lowlight3)
    hist_hazy = hist_equalization(hazy)
    hist_stoneface = hist_equalization(stoneface)

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
    #print(np.amin(clahe_stoneface), np.amax(clahe_stoneface), clahe_stoneface.shape, stoneface.shape)
    pyplot.imshow(clahe_stoneface, cmap='gray', norm=NoNorm())
    pyplot.subplot(224, title='CLAHE with overlap')
    pyplot.imshow(clahe_stoneface_overlap, cmap='gray', norm=NoNorm())
    pyplot.show()

    from skimage import exposure
    img_eq = np.array(exposure.equalize_hist(lowlight2) * 255, dtype=np.uint8)
    test_clahe = exposure.equalize_adapthist(stoneface)
    #print(np.amax(test_clahe), np.amax(clahe_stoneface))
    #print(clahe_stoneface)
    pyplot.imshow(test_clahe, cmap='gray')
    pyplot.show()
    # print(np.amax(img_eq),np.amax(hist_lowlight2))
    # print(np.sum(img_eq-hist_lowlight2))

    '''
    pyplot.subplot(221)
    pyplot.imshow(hist_lowlight2, cmap='gray', norm =NoNorm())
    pyplot.subplot(222)
    pyplot.imshow(img_eq, cmap='gray', norm =NoNorm())
    pyplot.subplot(223)
    pyplot.imshow(hist_hazy, cmap='gray', norm =NoNorm())
    pyplot.subplot(224)
    pyplot.imshow(hist_stoneface, cmap='gray', norm =NoNorm())
    pyplot.show()
    img_hist, bins = exposure.histogram(stoneface)
    pyplot.plot(bins, img_hist/ img_hist.max())
    pyplot.show()
    '''
    return


def problem2():
    mathbooks = skimage.io.imread(mathbooks_path)
    image = saturated_contrast(mathbooks)

    return


def problem3():
    image = skimage.io.imread(stoneface_path)
    scale=3
    resized_nearest = resize(image,scale=scale,type='nearest')
    resized_bilinear = resize(image, scale=scale,type='bilinear')
    pyplot.subplot(131)
    pyplot.imshow(image)
    pyplot.subplot(132)
    pyplot.imshow(resized_nearest)
    pyplot.subplot(133)
    pyplot.imshow(resized_bilinear)
    pyplot.show()
    return


def problem4():
    image = skimage.io.imread(hazy_path)
    theta = 120

    #image = image[:100,:100]

    rotated_nearest = rotate(image,theta=theta,type='nearest')
    rotated_bilinear = rotate(image, theta=theta,type='bilinear')
    pyplot.subplot(131)
    pyplot.imshow(image)
    pyplot.subplot(132)
    pyplot.imshow(rotated_nearest)
    pyplot.subplot(133)
    pyplot.imshow(rotated_bilinear)
    pyplot.show()

    rotated_nearest = rotate(image,theta=theta,type='nearest', corner='bottom_left')
    rotated_bilinear = rotate(image, theta=theta,type='bilinear', corner='bottom_left')
    pyplot.subplot(131)
    pyplot.imshow(image)
    pyplot.subplot(132)
    pyplot.imshow(rotated_nearest)
    pyplot.subplot(133)
    pyplot.imshow(rotated_bilinear)
    pyplot.show()
    return


def problem5():
    noisy_image_path = Path('../Data/images/noisy.png')

    noisy_image = skimage.io.imread(noisy_image_path.as_posix())
    pyplot.subplot(121)
    pyplot.imshow(noisy_image, cmap='gray')
    pyplot.subplot(122)
    pyplot.imshow(noisy_image, cmap='gray')
    pyplot.show()
    return


def main():
    #problem1()
    #problem2()
    #problem3()
    problem4()

    return


if __name__ == '__main__':
    main()
