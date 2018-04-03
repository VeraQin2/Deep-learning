#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 19:09:04 2017

@author: veraqin
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import h5py
import numpy as np
import random
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical


X_train = []
X_test = []
for filename in ["gap_InceptionV3.h5"]:
    with h5py.File(filename, 'r') as h:
        X_train.append(np.array(h['train']))
        y_train = np.array(h['label'])

for filename in ["gap2_InceptionV3.h5"]:
    with h5py.File(filename, 'r') as h:
        X_test.append(np.array(h['test']))

X_train = np.concatenate(X_train, axis=1)
X_test = np.concatenate(X_test, axis=1)

X_train_breed=[]
first_t=0
rank_breed=[]
for t in range(0,len(y_train)):
    if (t>0) and (y_train[t]!=y_train[t-1]):
        last_t=t-1
        X_train_breed=X_train[first_t:last_t+1]
        first_t=last_t

        var_breed=np.var(X_train_breed,0)
        mean_breed=np.mean(X_train_breed,0)

        rank_breed.append((np.argsort(var_breed))[0:1500])

rank_breed=np.array(rank_breed)

selected_features=np.unique(rank_breed)

X_train_selected=np.zeros((len(X_train),len(selected_features)))
X_test_selected=np.zeros((len(X_test),len(selected_features)))
t=0
for i in selected_features:
    X_train_selected[:,t]=X_train[:,i]
    X_test_selected[:,t]=X_test[:,i]
    t=t+1

X_train_shuffle, y_train_shuffle = shuffle(X_train_selected, y_train)

length=len(y_train)

y_train_all_shuffle=to_categorical(y_train_shuffle, num_classes=120)

from keras.models import *
from keras.layers import *

input_tensor = Input(X_train.shape[1:])
model = Sequential()
model.add(Dropout(0.5,input_shape=(len(selected_features),)))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(120, activation='softmax'))

adam=optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

#history=model.fit(X_train_shuffle, y_train_all_shuffle, batch_size=150, epochs=50, validation_split=0.2)

#y_pred_in = model.predict(X_test_selected, verbose=2)
#y_pred_X = model.predict(X_test_selected, verbose=2)
#y_pred_res = model.predict(X_test_selected, verbose=2)

y_pred = (y_pred_X + y_pred_in) * 0.5

np.savetxt('np.csv',y_pred, delimiter=',')

'''
X_fea = np.concatenate((y_pred_in, y_pred_X), axis=1) #np.concatenate((np.concatenate((y_pred_in, y_pred_X), axis=1), y_pred_res), axis=1)
print(X_fea.shape)

model = Sequential()
model.add(Dense(120,input_shape=(240,), use_bias=False))
adam=optimizers.Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

X_fea_shuffle, y_train_shuffle = shuffle(X_fea, y_train)
y_train_one_vs_all = to_categorical(y_train_shuffle, num_classes=120)

history=model.fit(X_fea_shuffle, y_train_one_vs_all, batch_size=32, epochs=100, validation_split=0.2)

y_pred_combine = model.predict(X_fea, verbose=2)


import math
def loss_func(expect, output):
    return -math.log(output[expect])
def mean_classify_error(expect, output):
    if np.argmax(output) == expect:
        return 0
    else:
        return 1.0

w_in =0.65
w_x = 0.34
w_res = 0.01
#new_res = y_pred_in
new_res = w_in * y_pred_in + w_x * y_pred_X + w_res * y_pred_res
entropy_loss = 0
accuracy_loss = 0
#print len(y_train)
for i in range(len(y_train)):
    #entropy_loss += loss_func(y_train[i], new_res[i])
    accuracy_loss += mean_classify_error(y_train[i], new_res[i])
#entropy_loss = entropy_loss/len(y_train)
accuracy_loss = 1.0-accuracy_loss/len(y_train)
'''