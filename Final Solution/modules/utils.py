'''
###########################################################################################################################
-> FileName : utils.py
-> Description : This file contains all the important functions used 
                 throughout this project
-> Author : Hardik Vagadia
-> E-Mail : vagadia49@gmail.com
-> Date : 16th June 2020
###########################################################################################################################
'''

#---------------------------------------------------------------------------------------------------	

#Importing essential libraries
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import math
import cv2
import os
import shutil
from google.colab import files
import pathlib


import utils as u

#Tensorflow stable version
import tensorflow
tensorflow.compat.v1.enable_eager_execution()
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.initializers import he_normal, zeros, glorot_normal, RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.regularizers import l1, l2, l1_l2

#---------------------------------------------------------------------------------------------------	

class Sharpen(tensorflow.keras.layers.Layer):
  """
  Sharpen layer sharpens the edges of the image.
  """
  def __init__(self, num_outputs) :
    super(Sharpen, self).__init__()
    self.num_outputs = num_outputs

  def build(self, input_shape) :
    self.kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    self.kernel = tensorflow.expand_dims(self.kernel, 0)
    self.kernel = tensorflow.expand_dims(self.kernel, 0)
    self.kernel = tensorflow.cast(self.kernel, tensorflow.float32)

  def call(self, input_) :
    return tensorflow.nn.conv2d(input_, self.kernel, strides=[1, 1, 1, 1], padding='SAME')

#---------------------------------------------------------------------------------------------------	

def get_model(IMG_WIDTH, IMG_HEIGHT, N_CHANNELS, N_CLASSES) :
  #Input layer
  input_layer = Input(shape=(IMG_HEIGHT, IMG_WIDTH, N_CHANNELS, ), name="input_layer", dtype='float32')
  #Sharpen Layer to sharpen the edges of the image.
  sharp = Sharpen(num_outputs=(IMG_HEIGHT, IMG_WIDTH, N_CHANNELS, ))(input_layer)
  #Convolution, maxpool and dropout layers
  conv_1 = Conv2D(filters=32, kernel_size=(5,5), activation=relu,
                  kernel_initializer=he_normal(seed=54), bias_initializer=zeros(),
                  name="first_convolutional_layer") (sharp)
  conv_2 = Conv2D(filters=64, kernel_size=(3,3), activation=relu,
                  kernel_initializer=he_normal(seed=55), bias_initializer=zeros(),
                  name="second_convolutional_layer") (conv_1)                  
  maxpool_1 = MaxPool2D(pool_size=(2,2), name = "first_maxpool_layer")(conv_2)
  dr1 = Dropout(0.25)(maxpool_1)
  conv_3 = Conv2D(filters=64, kernel_size=(3,3), activation=relu,
                  kernel_initializer=he_normal(seed=56), bias_initializer=zeros(),
                  name="third_convolutional_layer") (dr1)
  maxpool_2 = MaxPool2D(pool_size=(2,2), name = "second_maxpool_layer")(conv_3)
  dr2 = Dropout(0.25)(maxpool_2) 
  flat = Flatten(name="flatten_layer")(dr2)

  #Fully connected layers
  d1 = Dense(units=256, activation=relu, kernel_initializer=he_normal(seed=45),
             bias_initializer=zeros(), name="first_dense_layer_classification", kernel_regularizer = l2(0.001))(flat)
  dr3 = Dropout(0.5)(d1)
  
  classification = Dense(units = N_CLASSES, activation=None, name="classification",  kernel_regularizer = l2(0.0001))(dr3)
  
  regression = Dense(units = 4, activation = 'linear', name = "regression", 
                     kernel_initializer=RandomNormal(seed=43), kernel_regularizer = l2(0.1))(dr3)
  #Model
  model = Model(inputs = input_layer, outputs = [classification, regression])
  return model        

#---------------------------------------------------------------------------------------------------	

def get_test_df(test_df_path, IMG_WIDTH, IMG_HEIGHT) :
  """
  Function to load the test dataframe and modify
  the bounding box coordinates as per the IMG_WIDTH
  and IMG_HEIGHT
  """
  test_df = pd.read_csv(test_df_path)
  for idx, row in test_df.iterrows() :
    w = row['Width']
    h = row['Height']
    if w > IMG_WIDTH :
      diff = w-IMG_WIDTH
      test_df.iloc[idx, 4] = test_df.iloc[idx]['Roi.X2'] - diff
    else :
      diff = IMG_WIDTH-w
      test_df.iloc[idx, 4] = test_df.iloc[idx]['Roi.X2'] + diff
    if h > IMG_HEIGHT :
      diff = h - IMG_HEIGHT
      test_df.iloc[idx, 5] = test_df.iloc[idx]['Roi.Y2'] - diff
    else :
      diff = IMG_HEIGHT - h
      test_df.iloc[idx, 5] = test_df.iloc[idx]['Roi.Y2'] + diff
  return test_df

#---------------------------------------------------------------------------------------------------	

def evaluate_test_images(path, model, IMG_WIDTH, IMG_HEIGHT, N_CHANNELS) :
  """
  Function to make predictions for the test set images
  """
  labels = []
  bbox = []
  all_imgs = os.listdir(path)
  all_imgs.sort()
  for img in tqdm(all_imgs) :
    if '.png' in img :
      image_string = tensorflow.io.read_file(path + '/' + img)
      #Loading and decoding image
      image = tensorflow.image.decode_png(image_string, channels=N_CHANNELS)
      #Converting image data type to float
      image = tensorflow.image.convert_image_dtype(image, tensorflow.float32)
      #Adjusting image brightness and contrast
      if tensorflow.math.reduce_mean(image) < 0.3 :
        image = tensorflow.image.adjust_contrast(image, 5)
        image = tensorflow.image.adjust_brightness(image, 0.2)
      #Resizing image
      image = tensorflow.image.resize(image, [IMG_HEIGHT, IMG_WIDTH], method="nearest", preserve_aspect_ratio=False)
      image = image/255.0
      image = np.expand_dims(image, axis=0)
      #Predicting output
      pred = model.predict(image)
      labels.append(np.argmax(pred[0][0]))
      bbox.append(pred[1][0])
  return labels, bbox

#---------------------------------------------------------------------------------------------------  