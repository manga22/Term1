import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
import cv2
import glob
from configparser import ConfigParser

def prepare_data(config_file_path='config.ini'):
  config = ConfigParser()
  config.read(config_file_path)
  train_config = config['train']
  test_config = config['test']
  train_data = prepare_training_data(train_config)
  test_data = prepare_test_data(test_config)
  return [train_data, test_data]

def prepare_training_data(train_config):

  print('Preparing training data...')

  patch_size = int(train_config['patch_size'])
  train_img_dir = train_config['train_img_dir']
  sigma = int(train_config['noise_sigma'])
  scale = int(train_config['upscale_factor'])
  stride = int(train_config['stride'])

  img_paths = glob.glob(train_img_dir+'/*')
  sigma_norm = sigma/255.0
  lr_patches = []
  hr_patches = []

  for k in range(len(img_paths)):
    hr_im = imread(img_paths[k])
    hr_im = hr_im/255.0
    h, w, _ = hr_im.shape
    n_h = int((h - patch_size + 1)/stride)
    n_w = int((w - patch_size + 1)/stride)
    for i in range(n_h):
      for j in range(n_w):
        hr_patch = hr_im[i*stride:i*stride +patch_size, j*stride:j*stride+patch_size]
        hr_patches.append(hr_patch)
        lr_patch = cv2.resize(hr_patch, dsize= (int(patch_size/scale),int(patch_size/scale)), interpolation=cv2.INTER_CUBIC)
        lr_patch = lr_patch + np.random.normal(0, sigma_norm, lr_patch.shape)
        lr_patch = np.clip(lr_patch,0,1) 
        lr_patches.append(lr_patch)
        
  patch_lr_hr={}
  patch_lr_hr["lr"] = np.array(lr_patches)
  patch_lr_hr["hr"] = np.array(hr_patches)
  return patch_lr_hr


def prepare_test_data(test_config):

  print('Preparing testing data...')

  test_img_dir = test_config['test_img_dir']
  sigma = int(test_config['noise_sigma'])
  scale = int(test_config['upscale_factor'])
  test_img_paths= glob.glob(test_img_dir+'/*')
  test_hr=[]
  test_lr=[]
  for img_path in test_img_paths:
    hr_im = imread(img_path)
    test_hr.append(hr_im/255.0)
    h, w, _ = hr_im.shape
    lr_im = cv2.resize(hr_im, dsize= (int(w/scale),int(h/scale)), interpolation=cv2.INTER_CUBIC)
    lr_im = lr_im + np.random.normal(0, sigma, lr_im.shape)
    test_lr.append(lr_im/255.0)
  test_data={}
  test_data["hr"] = np.array(test_hr)
  test_data["lr"] = np.array(test_lr)
  return test_data