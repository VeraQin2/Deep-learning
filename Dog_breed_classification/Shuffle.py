import h5py
import numpy as np
import random
from sklearn.utils import shuffle

np.random.seed(2017)
X_train = []
X_test = []
for filename in ["gap_ResNet50.h5", "gap_Xception.h5", "gap_InceptionV3.h5"]:
    with h5py.File(filename, 'r') as h:
        X_train.append(np.array(h['train']))
        X_test.append(np.array(h['test']))
        y_train = np.array(h['label'])

#print (len(y_train))
#print (len(X_train))
#print (len(X_test))

print(X_train)
print(y_train)
#print(X_test)

X_train = np.concatenate(X_train, axis=1)
X_test = np.concatenate(X_test, axis=1)
X_train, y_train = shuffle(X_train, y_train)

from keras.models import *
from keras.layers import *
np.random.seed(2017)
input_tensor = Input(X_train.shape[1:])
x = Dropout(0.5)(input_tensor)
x = Dense(3, activation='softmax')(x)
model = Model(input_tensor, x)
model.compile(optimizer='adadelta',
              loss='binary_crossentropy',
              metrics=['accuracy'])

y_train=[[1,0,0],[1,0,0],[0,1,0],[0,1,0],[0,0,1],[0,0,1]];
model.fit(X_train, y_train, batch_size=1, epochs=8, validation_split=0.2)

y_pred = model.predict(X_test, verbose=2)

#y_pred = y_pred.clip(min=0.005, max=0.995)

print (y_pred)
import pandas as pd
from keras.preprocessing.image import *

#df = pd.read_csv("sample_submission.csv")
#gen = ImageDataGenerator()
#test_generator = gen.flow_from_directory("test2", (224, 224), shuffle=False,
#                                         batch_size=16, class_mode=None)

#for i, fname in enumerate(test_generator.filenames):
#    index = int(fname[fname.rfind('/')+1:fname.rfind('.')])
#    df.set_value(index-1, 'label', y_pred[i])
#df.to_csv('pred.csv', index=None)
#df.head(10)