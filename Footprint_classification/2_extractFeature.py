#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 22:29:39 2017

@author: veraqin
"""
#import csv
#import os
import numpy as np

from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

from keras.applications.vgg19 import VGG19
model = VGG16(weights='imagenet', include_top=False)


data = np.empty((2000,7*7*512), dtype="uint8")
for j in range(8000,10000):
    path = './data_test/2_test_deal/'+ str(j+1)+'.png'
    img = image.load_img(path, target_size=(224, 224))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    features = model.predict(arr)
    features = features.reshape(1,7*7*512)
    data[j-8000,:] = features

np.savetxt('./data_test/2_feat/5.csv', data, fmt = '%d')


big_file = os.listdir('./2_train/train_10')
num = len(big_file)
for j in range(num):
    if not big_file[j].startswith('.'):
        img_file = os.listdir('./2_train/train_10/'+big_file[j])
        num2 = len(img_file)
        print j,num2
        data = np.empty((num2,7*7*512),dtype= 'uint8')
        label = np.empty((1,num2),dtype='int')
        for i in range(num2):
            if not img_file[i].startswith('.'):
                img = image.load_img('./2_train/train_10/'+big_file[j]+'/'+img_file[i], target_size=(224, 224))
                arr = image.img_to_array(img)
                arr = np.expand_dims(arr, axis=0)
                arr = preprocess_input(arr)
                features = model.predict(arr)
                features = features.reshape(1,7*7*512)
                data[i,:] = features
                label[0,i] = int(float(img_file[i].split('_')[0]))
        store = np.concatenate((data,label.T),axis = 1)
        np.savetxt('./2_train/train_10_feat/' + str(j) + '.csv', store, fmt = '%d')



img_file = os.listdir('./2_train/train_10/')
num2 = len(img_file)
data = np.empty((num2,7*7*512),dtype= 'uint8')
label = np.empty((1,num2),dtype='int')
for i in range(num2):
    if not img_file[i].startswith('.'):
        img = image.load_img('./2_train/train_10/'+img_file[i], target_size=(224, 224))
        arr = image.img_to_array(img)
        arr = np.expand_dims(arr, axis=0)
        arr = preprocess_input(arr)
        features = model.predict(arr)
        features = features.reshape(1,7*7*512)
        data[i,:] = features
        label[0,i] = int(float(img_file[i].split('_')[0]))
        print label[0,i]
store = np.concatenate((data,label.T),axis = 1)
np.savetxt('./2_train/train_10_feat.csv', store, fmt = '%d')

