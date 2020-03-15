import tensorflow as tf
import os
import cv2
import keras
import argparse
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.pooling import AveragePooling2D
from keras.applications import VGG16
from keras.layers.core import Dropout, Flatten, Dense
from imutils import paths
# from src.Generator_eth import DataGenerator
import keras.backend as K


#baseModel = VGG16(include_top=False,
#	input_tensor=Input(shape=(200, 200, 3)))

baseModel = keras.applications.resnet_v2.ResNet152V2(include_top=False, weights='imagenet',
                                        input_tensor=Input(shape=(200, 200, 3)))

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(200, activation="relu")(headModel)
headModel = Dropout(0.2)(headModel)
#headModel = Dense(5, activation="softmax", kernel_regularizer=keras.regularizers.L1L2(0.1))(headModel)
headModel = Dense(5, activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
	layer.trainable = True

def rmse(y_true, y_pred):
    #print(K.sqrt(K.mean(K.square(y_pred - y_true))).dtype)
    return K.sqrt(K.mean(K.square(y_pred - y_true))) 

opt = Adam(lr=1e-4)
model.compile(loss='categorical_crossentropy', optimizer=opt,
	metrics=['acc'])

print(model.summary())


gen = DataGenerator('./UTKFace/', batch_size=32)
checkpoint = keras.callbacks.ModelCheckpoint('./save/ethnicity/ethnicity.h5', monitor='acc', verbose=0, save_best_only=True, 
                                                           mode='auto', period=1)
tensorboard = keras.callbacks.TensorBoard(log_dir='./logs/ethnicity_1/', histogram_freq=0, batch_size=64, write_graph=True, 
                                               write_grads=False, write_images=True, embeddings_freq=0, 
                                               embeddings_layer_names=None, embeddings_metadata=None, 
                                               embeddings_data=None, update_freq='epoch')
#model.load_weights('./save/age/model.h5')
history = model.fit_generator(gen, epochs=25, callbacks=[checkpoint, tensorboard])

#history = model.fit_generator(gen)