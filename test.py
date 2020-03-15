import tensorflow as tf
from tensorflow.keras import backend as K
import os
import cv2
import numpy as np
from PIL import Image
import random
from multiprocessing import Process
import warnings
warnings.filterwarnings("ignore")

def rmse(y_true, y_pred):
    #print(K.sqrt(K.mean(K.square(y_pred - y_true))).dtype)
    return K.sqrt(K.mean(K.square(y_pred - y_true))) 

global img
# model_age = tf.keras.models.load_model('age50.h5', custom_objects={'rmse': rmse})
# model_gender = tf.keras.models.load_model('gender50.h5', custom_objects={'rmse': rmse})
# model_eth = tf.keras.models.load_model('ethnicity50.h5', custom_objects={'rmse': rmse})


# print(np.around(model_age.predict(img)))
# print(np.argmax(model_gender.predict(img), axis = 1))
# print(np.argmax(model_eth.predict(img), axis = 1))


def age(img):
    model_age = tf.keras.models.load_model('age50.h5', custom_objects={'rmse': rmse})
    print(np.around(model_age.predict(img)))

def gender(img):
    model_gender = tf.keras.models.load_model('gender50.h5', custom_objects={'rmse': rmse})
    print(np.argmax(model_gender.predict(img), axis = 1))



def ethnicity(img):
    model_eth = tf.keras.models.load_model('ethnicity50.h5', custom_objects={'rmse': rmse})
    print(np.argmax(model_eth.predict(img), axis = 1))


if __name__ == '__main__':
    img = cv2.imread('./img/3_1_0_20170109191801408.jpg.chip.jpg')
    print(img.shape)
    img = np.reshape(img, (1, 200, 200, 3))
    print(img.shape)
    p1 = Process(target=age, args=(img,))
    p1.start()
    p2 = Process(target=gender, args=(img,))
    p2.start()
    p3 = Process(target=ethnicity, args=(img,))
    p3.start()
    p1.join()
    p2.join()
    p3.join()


