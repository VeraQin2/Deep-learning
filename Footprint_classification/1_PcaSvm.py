#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 19:53:42 2017

@author: veraqin
"""
import numpy as np
import os
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.svm import SVC
#from sklearn import cross_validation


def load_train_data():
    data = np.empty((10000,30000),dtype= 'uint8')
    label = np.empty((10000,),dtype="int")
    
    img_file = os.listdir('./1/train1')
    i = 0
    j = 2
    for ff in img_file:
        if not ff.startswith('.'):
            j -= 1
            imgs = os.listdir('./1/train1/' + ff)
            for f in imgs:
                if not f.startswith('.'):
                    img = Image.open('./1/train1/' + ff +'/'+ f)
                    arr = np.asarray(img,dtype='uint8')
                    arr = arr.reshape(1,30000)
                    data[i,:] = arr
                    label[i] = j
                    i += 1
                    img.close()
    return data,label

data, label = load_train_data()

def load_test_data():
    data = np.empty((10000,30000), dtype="uint8")
    #imgs = os.listdir('./valid')
    #num = len(imgs)
    for j in range(10000):
        path = './data_test/1_test_deal/'+ str(j+1)+'.png'
        img = Image.open(path)
        arr = np.asarray(img, dtype="uint8")
        arr = arr.reshape(1,30000)
        data[j,:] = arr
        img.close()
    return data

test_data = load_test_data()

#X_train, X_test, y_train, y_test = cross_validation.train_test_split(data, label, test_size=0.1, random_state=0)


pca = PCA(n_components=300, svd_solver = 'randomized', whiten = True)
pca.fit(data)
X_train = pca.transform(data)
#X_t_test = pca.transform(X_test)

test = pca.transform(test_data)

clf = SVC()
clf.fit(X_train, label)
#print 'score', clf.score(X_t_test, y_test)

test_label = clf.predict(test)
for i in range(10000):
    test_label[i] += 1

np.savetxt('./data_test/1.csv', test_label, fmt = '%d')
