#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 20:23:09 2017

@author: veraqin
"""

import numpy as np
import os
from PIL import Image
from sklearn.svm import LinearSVC


def load_train_data():
    img_file = os.listdir('./2_train/big_no_noise')
    num = len(img_file)
    data = np.empty((num,30000),dtype= 'uint8')
    label = np.empty((num,),dtype='int')
    for i in range(2):
        if not img_file[i].startswith('.'):
            img = Image.open('./train/' + img_file[i])
            img = img.resize((100,300))
            label[i-1] = int(float(img_file[i].split('_')[0]))
            arr = np.asarray(img,dtype='uint8')
            arr = arr.reshape((1,30000))
            data[i-1,:] = arr
            img.close()
    return data, label

train_data, train_label = load_train_data()


'''
def load_test_data():
    data = np.empty((10000,30000), dtype="uint8")
    for j in range(10000):
        path = './valid2/'+ str(j+1)+'.png'
        img = Image.open(path)
        arr = np.asarray(img, dtype="uint8")
        arr = arr.reshape((1,30000))
        data[j,:] = arr
        img.close()
    return data

test_data = load_test_data()

clf = LinearSVC()
clf.fit(train_data, train_label)
test_label = clf.predict(test_data)

np.savetxt('2.csv', test_label, fmt = '%d')
'''

