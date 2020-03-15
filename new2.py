import time
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
        gender.append(gender_)
        ethinicity.append(ethinicity_)
        a[0] = age_
        a[1] = gender_
        a[2] = ethinicity_
        y_label.append(a)
        img = cv2.imread(os.path.join(folder,filename), cv2.IMREAD_GRAYSCALE)
        print(gender)
        time.sleep(5)
        if img is not None:
            images.append(img)
        
            
    return images, age, gender, ethinicity, y_label

array, age, gender, etinicity, y_label = load_images("E:\\source\\repos\\project\\sem_v\\age_detection\\age_detection\\UTKFace")
array = np.array(array)
print("####")
