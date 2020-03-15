
import os
import cv2
import numpy as np
from PIL import Image
import keras
import random

original_image_dir = r"E:\source\sem_7\RemoveHaze\RemoveHaze\original_image\image"
training_image_dir = r"E:\source\sem_7\RemoveHaze\RemoveHaze\\training_images\data"
    
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self,original_image_dir = original_image_dir, training_image_dir = training_image_dir,patch_size=16, batch_size=32, shuffle=True, num_t = 10):
        'Initialization'
        #self.dim = dim
        self.num_t = num_t
        self.batch_size = batch_size
        self.patch_size = patch_size
        #self.labels = labels
        #self.list_IDs = list_IDs
        #self.n_channels = n_channels
        #self.n_classes = n_classes
        self.img_dir = original_image_dir
        self.train_dir = training_image_dir
        self.orig_images = os.listdir(self.img_dir)
        self.train_images = os.listdir(self.train_dir)
        self.shuffle = shuffle
        self.patch_size = patch_size
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.orig_images) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.orig_images[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.orig_images))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        images = []
        age = []
        gender= [] 
        ethinicity = []
        for filename in list_IDs_temp:
            #a = [0, 0, 0]
            a[0], a[1], a[2], _ = list(filename.split("_"))
            age.append(a[0])
            gender.append(a[1])
            ethinicity.append(a[2])
            #y_label.append(a)
            img = cv2.imread(os.path.join(folder,filename))
            if img is not None:
                images.append(img)

        return images, age, gender, ethinicity
        

########################## GENERATOR FUNCTION ##################################
################################################################################
def create_dataset(original_image_dir, training_image_dir, batch_size = 64):
    
    print(os.getcwd())
    orig_images = os.listdir(original_image_dir)
    train_images = os.listdir(training_image_dir)
    
    while True:
        orig_images_batch = np.random.choice(a = orig_images, size = batch_size, replace = False);
        x_train = []
        y_train = []

        for image_name in img_path:
            fullname = os.path.join(img_dir, image_name)
            img = cv2.imread(fullname)

            w,h,_ = img.shape

            num_w = int(w / patch_size)
            num_h = int(h / patch_size)
            for i in range(num_w):
                for j in range(num_h):

                    free_patch = img[0+i*patch_size:patch_size+i*patch_size, 0+j*patch_size:patch_size+j*patch_size, :]

                    for k in range(num_t):

                        t = random.random()
                        hazy_patch = free_patch * t + 255 * (1 - t)
                    
                        x_train.append(hazy_patch)
                        y_train.append(t)

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        # to make the dimension of y_train equal with x_train's
        y_train = np.expand_dims(y_train,axis=-1)
        y_train = np.expand_dims(y_train,axis=-1)
        y_train = np.expand_dims(y_train,axis=-1)

        print('The shape of x_train: ',x_train.shape)
        print('The shape of y_train: ',y_train.shape)

        yield (x_train, y_train)
        
        
#        X_train = []
#        y_train = []
#        for image in orig_images_batch:
#            image_path = os.path.join(original_image_dir, image)
#            #print(image_path)
#            img = cv2.imread(image_path)
#            #print(type(img))
#            #print(img.shape)
#            first_name, _ = image.split(".")
#            #print(first_name)
#            for files in train_images:
#                #print(files)
#                if(files.startswith(first_name)):
#                    #print(files)
#                    file_path = os.path.join(training_image_dir, files)
#                    train_img = cv2.imread(file_path)
#                    X_train.append(train_img)
#                    y_train.append(img)
#        X_train = np.array(X_train)
#        y_train = np.array(y_train)
#        yield (X_train, y_train)




##print(create_dataset(original_image_dir, training_image_dir))


