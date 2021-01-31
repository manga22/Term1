import argparse
import numpy as np
from skimage.io import imread, imsave
from tensorflow.keras import models
import matplotlib.pyplot as plt

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='models/upscale2x_model.h5')
    parser.add_argument('--test_image_path', default='baboon.png')
    parser.add_argument('--output_path', default='result.png')
    args = parser.parse_args()

    pretrained_model = models.load_model(args.model_path)
    image = imread(args.test_image_path)
    input = np.expand_dims(image/255.0, axis=0)
    hr_pred = pretrained_model.predict(input)[0]
    plt.imshow(image)
    plt.show()
    plt.imshow(hr_pred)
    plt.show()

    output_im = np.array(np.clip(hr_pred,0,1)*255,dtype=np.uint8)
    imsave(args.output_path,output_im)