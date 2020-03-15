
import os
import cv2
import numpy as np
from PIL import Image
import keras
import random


    
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self,image_dir, batch_size=32, shuffle=True):
        'Initialization'
        #self.dim = dim
        self.batch_size = batch_size
        self.img_dir = image_dir
        self.img = os.listdir(self.img_dir)
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.img) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.img[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.img))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        images = []
        temp = []
        for filename in list_IDs_temp:
            #a = [0, 0, 0]
            li = list(filename.split("_"))
            if(len(li) == 4):
                _, _, k, _ = li
            
            #y_label.append(a)
                img = cv2.imread(os.path.join(self.img_dir,filename))
                if img is not None:
                    images.append(img)
                    temp.append(int(k))


        temp = np.array(temp)
        ethnicity = np.zeros((temp.size, 5))
        ethnicity[np.arange(temp.size),temp] = 1
        images = np.array(images)
        #ethnicity = np.array(ethnicity)
        #print(images.shape)
        #print(ethnicity.shape)
        return images, ethnicity
        





