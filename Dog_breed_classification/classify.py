#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 14:50:38 2017

@author: veraqin
"""
import csv
import os
from shutil import copyfile
import numpy as np

'''
# load in train label
label_file = open('labels.csv','rt')
label_data = csv.reader(label_file, delimiter=",")
x = list(label_data)
label = np.array(x).astype("string")
label = label[1:,:]
# build folder
for i in range(len(label)):
    if not os.path.exists('train_cnn/' + label[i][1]):
        os.mkdir('train_cnn/' + label[i][1])
    if not os.path.exists('valid_cnn/' + label[i][1]):
        os.mkdir('valid_cnn/' + label[i][1])
'''


# load in train data
train_filenames = os.listdir('train')
# put the same breed of images into one folder
for filename in train_filenames:
    for i in range(len(label)):
        if label[i][0] == os.path.splitext(filename)[0]:
            copyfile('train/' + filename, 'valid_cnn/' + label[i][1] + '/' + filename)
            break

'''
# verify correctness
allfile = os.listdir('train_split')
length = 0
for filename in allfile:
    if not filename.startswith('.'):
        thisfile = os.listdir('train_split/' + filename)
        for f in thisfile:
            if not f.startswith('.'):
                length += 1
print(length)
'''