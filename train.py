


import os
import sys
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import cv2
from tensorflow import keras
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.utils import np_utils
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dropout, Flatten, Dense
from imutils import paths
from src.Generator import DataGenerator
import tensorflow.keras.backend as K
import tensorflow as tf


#baseModel = VGG16(include_top=False,
#	input_tensor=Input(shape=(200, 200, 3)))

baseModel = keras.applications.resnet_v2.ResNet50V2(include_top=False, weights='imagenet',
                                        input_tensor=Input(shape=(200, 200, 3)))

# baseModel = tf.keras.applications.EfficientNetB7(include_top=False, weights='imagenet',
#                                         input_tensor=Input(shape=(200, 200, 3)))

ageModel = baseModel.output
ageModel = AveragePooling2D(pool_size=(2, 2), name='agepool1')(ageModel)
ageModel = Flatten(name="flatten_age")(ageModel)
ageModel = Dense(128, activation="relu")(ageModel)
ageModel = Dropout(0.3)(ageModel)
ageModel = Dense(64, activation="relu")(ageModel)
ageModel = Dropout(0.3)(ageModel)
#ageModel = Dense(5, activation="softmax", kernel_regularizer=keras.regularizers.L1L2(0.1))(ageModel)
ageModel = Dense(2, activation="sigmoid", name='age')(ageModel)


ethModel = baseModel.output
ethModel = AveragePooling2D(pool_size=(2, 2))(ethModel)
ethModel = Flatten(name="flatten_eth")(ethModel)
ethModel = Dense(128, activation="relu")(ethModel)
ethModel = Dropout(0.3)(ethModel)
#ethModel = Dense(5, activation="softmax", kernel_regularizer=keras.regularizers.L1L2(0.1))(ethModel)
ethModel = Dense(5, activation="softmax", name='ethnicity')(ethModel)

genModel = baseModel.output
genModel = AveragePooling2D(pool_size=(2, 2))(genModel)
genModel = Flatten(name="flatten_gen")(genModel)
genModel = Dense(128, activation="relu")(genModel)
genModel = Dropout(0.3)(genModel)
#genModel = Dense(5, activation="softmax", kernel_regularizer=keras.regularizers.L1L2(0.1))(genModel)
genModel = Dense(2, activation="softmax", name='gender')(genModel)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=[ageModel, genModel, ethModel])

for layer in baseModel.layers:
	layer.trainable = True

def rmse(y_true, y_pred):
    #print(K.sqrt(K.mean(K.square(y_pred - y_true))).dtype)
    return K.sqrt(K.mean(K.square(y_pred - y_true))) 

losses = {
	"age": rmse,
	"ethnicity": "categorical_crossentropy",
    "gender" : "binary_crossentropy"
}

metric = {
	"ethnicity": 'acc',
    "gender" : 'acc'
}

opt = Adam(lr=1e-4)
model.compile(loss=losses, optimizer=opt, metrics=metric)

print(model.summary())

gen = DataGenerator(r'D:\source\sem_9\GuessIt\UTKFace\UTKFace', batch_size=32)
val_gen = DataGenerator(r'D:\source\sem_9\GuessIt\crop_part1', batch_size=32)
checkpoint = tf.keras.callbacks.ModelCheckpoint('./save/model.h5', verbose=0, save_best_only=True)
tboard_log_dir = os.path.join(".logs")
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=tboard_log_dir, write_graph=True, write_images=True, histogram_freq=0)
tensorboard.set_model(model)
# #model.load_weights('./save/ethnicity/model.h5')
history = model.fit(gen, epochs=30, callbacks=[checkpoint, tensorboard], validation_data=val_gen)

#history = model.fit_generator(gen)

# while(True):
#     gen.__getitem__(5)
