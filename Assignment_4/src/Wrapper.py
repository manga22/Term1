# Wrapper file to test all functions
from pathlib import Path

import cv2
from matplotlib import pyplot as plt
from scipy.io import loadmat
from skimage.io import imread
from skimage.transform import resize

from manogna_sreenivas.AllFunctions import decimate, image_denoise, \
    wiener_filter, inverse_filter, cls_filter, bilateral_filter, nonlocalmeans_filter, edge_detect

barbara_path = Path('../Data/images/barbara.tif')
blur_kernel_path = Path('../Data/images/BlurKernel.mat')
blur_high_path = Path('../Data/images/Blurred-HighNoise.png')
blur_low_path = Path('../Data/images/Blurred-LowNoise.png')
blur_med_path = Path('../Data/images/Blurred-MedNoise.png')
noisybook1_path = Path('../Data/images/noisy-book1.png')
noisybook2_path = Path('../Data/images/noisy-book2.png')
origbook_path = Path('../Data/images/Original-book.png')
characters_path = Path('../Data/images/characters.tif')
table_path = Path('../Data/images/table.png')

def problem1():
    low_noise = imread(blur_low_path)
    med_noise = imread(blur_med_path)
    high_noise = imread(blur_high_path)
    blur_kernel = loadmat(blur_kernel_path)['h']

    low_weiner_deblur = wiener_filter(low_noise, blur_kernel, noise_var=1)
    med_weiner_deblur = wiener_filter(med_noise, blur_kernel, noise_var=5)
    high_weiner_deblur = wiener_filter(high_noise, blur_kernel, noise_var=10)

    plt.subplot(121, title='Low noise')
    plt.imshow(low_noise, cmap='gray', vmin=0, vmax=255, interpolation=None)
    plt.subplot(122, title='Weiner filtered')
    plt.imshow(low_weiner_deblur, cmap='gray', vmin=0, vmax=255, interpolation=None)
    plt.show()

    plt.subplot(121, title='Med noise')
    plt.imshow(med_noise, cmap='gray', vmin=0, vmax=255, interpolation=None)
    plt.subplot(122, title='Weiner filtered')
    plt.imshow(med_weiner_deblur, cmap='gray', vmin=0, vmax=255, interpolation=None)
    plt.show()

    plt.subplot(121, title='High noise')
    plt.imshow(high_noise, cmap='gray', vmin=0, vmax=255, interpolation=None)
    plt.subplot(122, title='Weiner filtered')
    plt.imshow(high_weiner_deblur, cmap='gray', vmin=0, vmax=255, interpolation=None)
    plt.show()

    low_inverse_deblur = inverse_filter(low_noise, blur_kernel)
    med_inverse_deblur = inverse_filter(med_noise, blur_kernel)
    high_inverse_deblur = inverse_filter(high_noise, blur_kernel)

    plt.subplot(121, title='Low noise')
    plt.imshow(low_noise, cmap='gray', vmin=0, vmax=255, interpolation=None)
    plt.subplot(122, title='Inverse filtered')
    plt.imshow(low_inverse_deblur, cmap='gray', vmin=0, vmax=255, interpolation=None)
    plt.show()

    plt.subplot(121, title='Med noise')
    plt.imshow(med_noise, cmap='gray', vmin=0, vmax=255, interpolation=None)
    plt.subplot(122, title='Inverse filtered')
    plt.imshow(med_inverse_deblur, cmap='gray', vmin=0, vmax=255, interpolation=None)
    plt.show()

    plt.subplot(121, title='High noise')
    plt.imshow(high_noise, cmap='gray', vmin=0, vmax=255, interpolation=None)
    plt.subplot(122, title='Inverse filtered')
    plt.imshow(high_inverse_deblur, cmap='gray', vmin=0, vmax=255, interpolation=None)
    plt.show()

    low_cls_deblur = cls_filter(low_noise, blur_kernel)
    med_cls_deblur = cls_filter(med_noise, blur_kernel)
    high_cls_deblur = cls_filter(high_noise, blur_kernel)

    plt.subplot(121, title='Low noise')
    plt.imshow(low_noise, cmap='gray', vmin=0, vmax=255, interpolation=None)
    plt.subplot(122, title='CLS filtered')
    plt.imshow(low_cls_deblur, cmap='gray', vmin=0, vmax=255, interpolation=None)
    plt.show()

    plt.subplot(121, title='Med noise')
    plt.imshow(med_noise, cmap='gray', vmin=0, vmax=255, interpolation=None)
    plt.subplot(122, title='CLS filtered')
    plt.imshow(med_cls_deblur, cmap='gray', vmin=0, vmax=255, interpolation=None)
    plt.show()

    plt.subplot(121, title='High noise')
    plt.imshow(high_noise, cmap='gray', vmin=0, vmax=255, interpolation=None)
    plt.subplot(122, title='CLS filtered')
    plt.imshow(high_cls_deblur, cmap='gray', vmin=0, vmax=255, interpolation=None)
    plt.show()

    return


def problem2():
    noisybook1 = imread(noisybook1_path)
    noisybook2 = imread(noisybook2_path)
    gaussian_filt = image_denoise(noisybook1, type='Gaussian', param=5)  # param = sigma for Gaussian kernel
    median_filt = image_denoise(noisybook1, type='Median', param=5)  # param = size of neighbourhood
    bilateral_filt = bilateral_filter(noisybook2,
                                      param=[3, 0.3, (7,7)])  # param = [sigma_gaussian, signma_intensity_diff, kernel_size]
    nlmeans_filt = nonlocalmeans_filter(noisybook2,
                                      param=[3, 0.3, (15,15)])
    gaussian2_filt = image_denoise(noisybook2, type='Gaussian', param=3)

    plt.subplot(121, title='Noisy book1')
    plt.imshow(noisybook1, cmap='gray', vmin=0, vmax=255, interpolation=None)
    plt.subplot(122, title='Gaussian filtered')
    plt.imshow(gaussian_filt, cmap='gray', vmin=0, vmax=255, interpolation=None)
    plt.show()

    plt.subplot(121, title='Noisy book1')
    plt.imshow(noisybook1, cmap='gray', vmin=0, vmax=255, interpolation=None)
    plt.subplot(122, title='Median filtered')
    plt.imshow(median_filt, cmap='gray', vmin=0, vmax=255, interpolation=None)
    plt.show()

    plt.subplot(121, title='Noisy book2')
    plt.imshow(noisybook2, cmap='gray', vmin=0, vmax=255, interpolation=None)
    plt.subplot(122, title='Bilateral filtered')
    plt.imshow(bilateral_filt, cmap='gray', vmin=0, vmax=255, interpolation=None)
    plt.show()

    plt.subplot(121, title='Noisy book2')
    plt.imshow(noisybook2, cmap='gray', vmin=0, vmax=255, interpolation=None)
    plt.subplot(122, title='Non local means filtered')
    plt.imshow(nlmeans_filt, cmap='gray', vmin=0, vmax=255, interpolation=None)
    plt.show()

    plt.subplot(121, title='Noisy book2')
    plt.imshow(noisybook2, cmap='gray', vmin=0, vmax=255, interpolation=None)
    plt.subplot(122, title='Gaussian filtered')
    plt.imshow(gaussian2_filt, cmap='gray', vmin=0, vmax=255, interpolation=None)
    plt.show()

    return


def problem3():
    barbara = imread(barbara_path)
    deci_image = decimate(barbara, d=2)
    deci_lpf_image = decimate(barbara, d=2, filter='Gaussian', D0=100)
    sk_resize_image = resize(barbara, (barbara.shape[0] // 2, barbara.shape[1] // 2))

    cv2.imshow('Image before resize', barbara)
    cv2.imshow('Decimation by 2', deci_image)
    cv2.imshow('Gaussian LPF and Decimation by 2', deci_lpf_image)
    cv2.imshow('Resize using lib', sk_resize_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return


def problem4():
    book = imread(origbook_path)
    sobel_out = edge_detect(book, thresh=70, type='Sobel', plot_grad=True)
    prewitt_out = edge_detect(book, thresh=70, type='Prewitt')
    plt.subplot(131, title='Input image')
    plt.imshow(book, cmap='gray', vmin=0, vmax=255, interpolation=None)
    plt.subplot(132, title='Sobel filter')
    plt.imshow(sobel_out, cmap='gray', vmin=0, vmax=255, interpolation=None)
    plt.subplot(133, title='Prewitt filter')
    plt.imshow(prewitt_out, cmap='gray', vmin=0, vmax=255, interpolation=None)
    plt.show()

    characters = imread(characters_path)
    sobel_out = edge_detect(characters, thresh=25, type='Sobel', plot_grad=True)
    prewitt_out = edge_detect(characters, thresh=100, type='Prewitt')
    plt.subplot(131, title='Input image')
    plt.imshow(characters, cmap='gray', vmin=0, vmax=255, interpolation=None)
    plt.subplot(132, title='Sobel filter(Thr=25)')
    plt.imshow(sobel_out, cmap='gray', vmin=0, vmax=255, interpolation=None)
    plt.subplot(133, title='Prewitt filter(Thr=100)')
    plt.imshow(prewitt_out, cmap='gray', vmin=0, vmax=255, interpolation=None)
    plt.show()

    table = imread(table_path)
    sobel_out = edge_detect(table, thresh=40, type='Sobel', plot_grad=True)
    prewitt_out = edge_detect(table, thresh=75, type='Prewitt')
    plt.subplot(131, title='Input image')
    plt.imshow(table, cmap='gray', vmin=0, vmax=255, interpolation=None)
    plt.subplot(132, title='Sobel filter(Thr=40)')
    plt.imshow(sobel_out, cmap='gray', vmin=0, vmax=255, interpolation=None)
    plt.subplot(133, title='Prewitt filter(Thr=75)')
    plt.imshow(prewitt_out, cmap='gray', vmin=0, vmax=255, interpolation=None)
    plt.show()
    return


def main():
    problem1()
    problem2()
    problem3()
    problem4()
    return


if __name__ == '__main__':
    main()
