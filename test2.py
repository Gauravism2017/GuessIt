import tensorflow as tf
from tensorflow.keras import backend as K
import os
import cv2
import numpy as np
from PIL import Image
import random
from multiprocessing import Process, Pool, Pipe, Queue
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import time


def rmse(y_true, y_pred):
    #print(K.sqrt(K.mean(K.square(y_pred - y_true))).dtype)
    return K.sqrt(K.mean(K.square(y_pred - y_true))) 

ageList = []
genderList = []
ethnicityList = []
# model_age = tf.keras.models.load_model('age50.h5', custom_objects={'rmse': rmse})
# model_gender = tf.keras.models.load_model('gender50.h5', custom_objects={'rmse': rmse})
# model_eth = tf.keras.models.load_model('ethnicity50.h5', custom_objects={'rmse': rmse})


# print(np.around(model_age.predict(img)))
# print(np.argmax(model_gender.predict(img), axis = 1))
# print(np.argmax(model_eth.predict(img), axis = 1))



def age(q):
    model_age = tf.keras.models.load_model('age50.h5', custom_objects={'rmse': rmse})
    li = []
    while(True):
        if(q.empty()):
            continue
        img = q.get()
        if(img == 'END'):
            break
        else:
            li.append(np.around(model_age.predict(img))[0])
    print('age')
    print(list(li))
    

def gender(q, r):
    global genderList
    model_gender = tf.keras.models.load_model('gender50.h5', custom_objects={'rmse': rmse})
    li = []
    while(True):
        if(q.empty()):
            continue
        img = q.get()
        if(img == 'END'):
            break
        else:
            li.append(np.argmax(model_gender.predict(img), axis = 1)[0])
    r.put(li)
    r.put('END')
    



def ethnicity(q, r):
    global ethnicityList
    model_eth = tf.keras.models.load_model('ethnicity50.h5', custom_objects={'rmse': rmse})
    li = []
    while(True):
        if(q.empty()):
            continue
        img = q.get()
        if(img == 'END'):
            break
        else:
            # print("Done")
            li.append(np.argmax(model_eth.predict(img), axis = 1)[0])
    print(li)
    r.put(li)
    r.put('END')

def read_data(conn1, conn2, conn3, r3, r4):
    global genderList, ethnicityList
    for filename in os.listdir('./crop_part1'):
        li = filename.split('_')
        if(len(li) == 4):
            age, gender, eth , _ = li
            ageList.append(age)
            genderList.append(int(gender))
            ethnicityList.append(int(eth))
            img = cv2.imread(os.path.join('./crop_part1', filename))
            img = np.reshape(img, (1, 200, 200, 3))
            conn1.put(img)
            conn2.put(img)
            conn3.put(img)
    conn1.put('END')
    conn2.put('END')
    conn3.put('END')
    print("Age List {}", format(ageList))
    print("Gender List {}", format(genderList))
    print("Eth List {}", format(ethnicityList))
    r3.put(genderList)
    r4.put(ethnicityList)
    r3.put('END')
    r4.put('END')


def main():
    q1 = Queue()
    q2 = Queue()
    q3 = Queue()
    r1 = Queue()
    r2 = Queue()
    r3 = Queue()
    r4 = Queue()
    p4 = Process(target=read_data, args=(q1, q2, q3,r3, r4))
    p4.start()
    p1 = Process(target=age, args=(q1,))
    p1.start()

    p2 = Process(target=gender, args=(q2,r1))
    p2.start()

    p3 = Process(target=ethnicity, args=(q3,r2))
    p3.start()
    p4.join()
    p1.join()
    p2.join()
    p3.join()
    y_true = r3.get()
    y_pred = r1.get()
    df_cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(df_cm, annot = True)
    plt.show()
    y_true = r4.get()
    y_pred = r2.get()
    df_cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(df_cm, annot = True)
    plt.show()
    print("boring")



if __name__ == '__main__':
    # p = Pool(processes=20)
    # modelList = p.map(model, ['age50.h5', 'ethnicity50.h5', 'gender50.h5'])
    # p.close()
    # print(modelList)
    t = time.time()
    main()
    print(time.time() - t)