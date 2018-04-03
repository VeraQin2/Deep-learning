#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 16:04:48 2017

@author: veraqin
"""


from keras.models import Sequential
from keras.models import Model
import numpy as np

from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Dropout
import cv2
from keras.utils.np_utils import to_categorical
import os

'''
model = Sequential()
# first convolutional layer, with 4 kernel, each 5*5
model.add(Convolution2D(4, 5, 5,input_shape=(224, 224,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#second convolutional layer, with 8 kernel, each 3*3
model.add(Convolution2D(8, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#third convolutional layer, with 16 kernel, each 3*3
model.add(Convolution2D(16, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#fully connected layer
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))

#softmax classify
model.add(Dense(120))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
'''
'''
from keras.preprocessing.image import *
gen = ImageDataGenerator()
train_set = gen.flow_from_directory("train_cnn", (224, 224), shuffle=True,
                                              batch_size=32)
test_set = gen.flow_from_directory("valid_cnn", (224, 224), shuffle=True,
                                         batch_size=32)

history = model.fit_generator(train_set,steps_per_epoch=259, epochs=1, 
                              validation_data=test_set,validation_steps=60)

'''
'''
n=10222
X = np.zeros((n, 224, 224, 3), dtype=np.uint8)
y = np.zeros((n, 1), dtype=np.uint8)

path = './train_class'
label = 0
idx = 0
for filename in os.listdir(path):
    print(label)
    if os.path.isdir(path + '/' + filename):
        for image in os.listdir(path + '/' + filename):
            if not image.startswith('.'):
                X[idx] = cv2.resize(cv2.imread(path + '/' + filename+ '/' + image), (224, 224))
                y[idx] = label
                idx += 1
        label += 1

y = to_categorical(y, num_classes=120)
print(y)

model.fit(X, y,batch_size=50, epochs=1, validation_split=0.2)

'''

print zip([x.name for x in model.layers], range(len(model.layers)))

#print(len(model.layers))
weights = model.layers[6].get_weights()[0] # 128*1
model2 = Model(model.input, [model.layers[8].output, model.output])
print len(weights)


img = cv2.imread('./Dog-lawn-1030x688.jpg')
img = cv2.resize(img, (224, 224))
x = img.copy()
x.astype(np.float32)
out, prediction = model2.predict(np.expand_dims(x, axis=0))
print out.shape, prediction.shape
prediction = prediction # 120*1
out = out[0] #7*7*2048

'''
cam = np.matmul(np.matmul(out, weights),prediction.reshape(120,1)) #7*7*120
print(cam.shape)
cam = (cam - cam.min())/(cam.max()-cam.min())

cam = cv2.resize(cam, (224, 224))
print(cam.shape)
heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
heatmap[np.where(cam <= 0.2)] = 0

out = cv2.addWeighted(img, 0.8, heatmap, 0.4, 0)


import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Cross entropy loss')
plt.ylabel('loss')
plt.xlabel('epoch')

plt.legend(['train', 'valid'], loc='upper left')
plt.savefig('error.png')

plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')

plt.legend(['train', 'valid'], loc='upper left')
plt.savefig('accuracy.png')
'''