import argparse
import glob
import numpy as np
import cv2
from skimage.io import imread, imsave
from tensorflow.keras import models
import matplotlib.pyplot as plt
from train import psnr

def eval_test(model_path, test_dir, noise_level, plot=False):
  pretrained_model = models.load_model(args.model_path)
  psnr_test=[]
  test_files=glob.glob(test_dir+'/*')
  for im_path in test_files:
    image = imread(im_path)
    h, w, _ = image.shape
    noisy_im = image + np.random.normal(0,noise_level,image.shape)
    image_downscaled = cv2.resize(image, (int(w/2),int(h/2)), interpolation=cv2.INTER_CUBIC )
    noisy_image_downscaled = image_downscaled + np.random.normal(0,noise_level,image_downscaled.shape)

    image = image/255.0

    #Get model prediction
    input_im = np.expand_dims(noisy_image_downscaled/255.0, axis=0)
    hr_pred = pretrained_model.predict(input_im)[0]

    #Evaluate PSNR
    output_psnr = psnr(hr_pred, image)
    psnr_test.append(output_psnr)
    im_name = im_path.split('/')[-1]
    print(f'PSNR for image {im_name} for noise level: {noise_level}: {output_psnr}')

    if plot:
      plt.imshow(noisy_image_downscaled)
      plt.show()
      plt.imshow(image)
      plt.show()
      plt.imshow(hr_pred)
      plt.show()
  return np.mean(psnr_test)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='models/upscale2x_model.h5')
    parser.add_argument('--test_dir', default='data/test')
    parser.add_argument('--noise_level', default=0)
    args = parser.parse_args()
    psnr_test = eval_test(args.model_path, args.test_dir,int(args.noise_level))
    print(f'Mean PSNR on test set for noise level {args.noise_level}: {psnr_test}')