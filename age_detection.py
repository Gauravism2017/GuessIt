
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import matplotlib.pyplot as plt
import cv2
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras_applications import resnet50
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.models import load_model

#import tensorflow as tf
#import keras


def load_images(folder):
    images = []
    y_label = []
    for filename in os.listdir(folder):
        a = [0, 0, 0]
        a[0], a[1], a[2], _ = list(filename.split("_"))
        #print(a)
        #print(filename)
        y_label.append(a)
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)

    return images, y_label


def load_images(folder):
    images = []
    age = []
    gender= [] 
    ethinicity = []
    for filename in os.listdir(folder):
        #a = [0, 0, 0]
        a[0], a[1], a[2], _ = list(filename.split("_"))
        age.append(a[0])
        gender.append(a[1])
        ethinicity.append(a[2])
        #y_label.append(a)
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)

#    return images, age, gender, ethinicity


#array, label = load_images("E:\\source\\repos\\project\\sem_v\\age_detection\\age_detection\\UTKFace")
#x_train = np.array(array)
#y_train = np.array(label)


array, age, gender, etinicity = load_images("E:\\source\\repos\\project\\sem_v\\age_detection\\age_detection\\crop_part1")
array = np.array(array)
label = np.array(age)
#print(label)

x_train, x_test, y_train, y_test = train_test_split(array, label, test_size=0.2, random_state=42)
print(y_train)
#le = LabelEncoder()
#le.fit(y_train)
#le.fit_transform(y_test)


#on = OneHotEncoder()
#y_train = on.fit(y_train)
#y_test = on.fit_transform(y_test)
x_train = x_train.reshape(x_train.shape[0], 200, 200, 3).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 200, 200, 3).astype('float32')
# normalize inputs from 0-255 to 0-1
x_train = x_train / 255
x_test = x_test / 255
# one hot encode outputs

on = OneHotEncoder()
y_train = on.fit(y_train)
y_test = on.fit_transform(y_test)

num_classes = y_test.shape[1]
print(num_classes)

# define the larger model
def larger_model():
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(200, 200, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
   # model.add(Dropout(0.2))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = larger_model()

tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

filepath="weights-improvement-all-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10,callbacks=[tbCallBack], batch_size=32)
model.save('my_model_ethinici1ty.h5')

#model = load_model('E:\\source\\repos\\project\\sem_v\\age_detection\\age_detection\my_model_ethinicity.h5')
#def load_images(folder):
#    images = []
#    y_label = []
#    for filename in os.listdir(folder):
        
#        img = cv2.imread(os.path.join(folder,filename))
#        if img is not None:
#            img = np.array(img)
#            img = img / 255
#            img = img.reshape(1, 200, 200, 3)
#            print(on.inverse_transform(model.predict(img)))
#            #print(img.shape)


#load_images("E:\\source\\repos\\project\\sem_v\\age_detection\\age_detection\\UTKFace")








