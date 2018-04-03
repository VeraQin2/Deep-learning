#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 2 14:20:34 2017

@author: Wenhao Qin
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
import csv
from PIL import Image


data = []
train_data = []

fmt = '.png'

for i in range(0,300):
    path = './3_test/'
    tmp_i = i #'{num:05d}'.format(num = i)
    name = str(tmp_i)
    fullname = path + name + fmt
    with Image.open(fullname) as img:
        im = np.reshape(img,(1,300*100))[0]
        train_data.append(im)
        data.append(im)


train_data = scale(train_data)
train_data = np.array(train_data)
data = scale(data)

estimator = KMeans(n_clusters = 20)

estimator.fit(data)
print 'finished fitting'

output = estimator.labels_

with open('./3.csv', 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',')
    for op in output:
        spamwriter.writerow([op]);
