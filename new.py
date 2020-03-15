
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

print("Module_imported")


def load_images(folder):
    images = []
    age = []
    gender= [] 
    ethinicity = []
    y_label = []

    for filename in os.listdir(folder):
        a = [0, 0 , 0]
        age_, gender_, ethinicity_, _ = list(filename.split("_"))
        age.append(age_)
        gender.append(gender)
        ethinicity.append(ethinicity_)
        a[0] = age_
        a[1] = gender_
        a[2] = ethinicity_
        y_label.append(a)
        img = cv2.imread(os.path.join(folder,filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
        
            
    return images, age, gender, ethinicity, y_label

array, age, gender, etinicity, y_label = load_images("E:\\source\\repos\\project\\sem_v\\age_detection\\age_detection\\UTKFace")
array = np.array(array)
label = np.array(gender)

print("Data loaded")
x_train, x_test, y_train, y_test = train_test_split(array, label, test_size=0.2, random_state=42)

x_train = x_train.reshape(x_train.shape[0], 200, 200, 1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 200, 200, 1).astype('float32')
# normalize inputs from 0-255 to 0-1
x_train = x_train / 255
x_test = x_test /255
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
print((y_train).shape, type(y_test).shape)


print("data manipulated")


num_classes = y_test.shape[1]
print(num_classes)

def larger_model():
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(200, 200, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("model compiled")
    return model

model = larger_model()


tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

filepath="weights-improvement-latest-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint, tbCallBack]
print("Training_Starting:")

model.fit(x_train, y_train, validation_data=(x_test, y_test),callbacks= callbacks_list, epochs=50, batch_size=128)
model.save('total1.h5')