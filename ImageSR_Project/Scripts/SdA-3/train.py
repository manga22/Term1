import os
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras import layers, losses

import prepare_data

def psnr(gt, pred):
  mse = np.sum((gt - pred) ** 2)
  mse /= float(pred.size)
  psnr = 20* math.log10(1 / math.sqrt(mse))
  return psnr

class MetricsCallback(keras.callbacks.Callback):
    def __init__(self, test_data):
        super(MetricsCallback, self).__init__()
        self.val_hr = test_data["hr"]
        self.val_lr = test_data["lr"]
        self.prediction = np.zeros(self.val_hr.shape)      

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 20 == 0:
            psnr_test =[]
            for i in range(self.val_lr.shape[0]):
              im = self.val_lr[i]
              gt = self.val_hr[i]
              pred = self.model.predict(np.expand_dims(im, axis=0))
              psnr_im = psnr(gt, pred[0])
              psnr_test.append(psnr_im)
              if i==0:
                plt.subplot(131)
                plt.imshow(self.val_lr[i])
                plt.title('lr input')
                plt.subplot(132)
                plt.imshow(self.val_hr[i])
                plt.title('hr gt')
                plt.subplot(133)
                plt.imshow(pred[0])
                plt.title('hr pred')
                plt.show()
            print('PSNR: ', np.mean(psnr_test))

def create_upscale2x_model():
  
      model = keras.Sequential([
      layers.Input(shape=(None, None, 3)), 
      layers.Conv2D(32, (3,3), activation='relu', padding='same', strides=2),
      layers.Conv2D(64, (3,3), activation='relu', padding='same', strides=2),
      layers.Conv2D(128, (3,3), activation='relu', padding='same', strides=2),
      layers.Conv2DTranspose(64, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2DTranspose(32, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2DTranspose(3, kernel_size=3, strides=2, activation='sigmoid', padding='same')])
    
      model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=losses.MeanSquaredError()) 
      return model


def train(config='config/config.ini',save_model_path='models/upscale2x_model.h5', epochs = 200):

  #Prepare data
  train_data, test_data= prepare_data.prepare_data(config_file_path=config)

  model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
                              filepath=save_model_path,
                              save_weights_only=False,
                              monitor='loss',
                              mode='min',
                              save_best_only=True)

  #Get model definition
  upscale2x = create_upscale2x_model()

  #Train the model
  upscale2x.fit(train_data["lr"], train_data["hr"],
                epochs=epochs,
                shuffle=True,
                batch_size=32,
                callbacks=[MetricsCallback(test_data),model_checkpoint_callback])

  return


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/config.ini')
    parser.add_argument('--save_model_path', default='models/upscale2x_model.h5')
    parser.add_argument('--epochs', default=200, type=int)
    args = parser.parse_args()
    train(config = args.config, save_model_path = args.save_model_path, epochs = args.epochs)
