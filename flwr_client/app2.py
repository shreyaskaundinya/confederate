# import os
# import random

# import cv2
# import flwr as fl
# import keras
# import numpy as np
# import pandas as pd
# from keras.callbacks import ReduceLROnPlateau
# from keras.layers import (BatchNormalization, Conv2D, Dense, Dropout, Flatten,
#                           MaxPool2D)
# from keras.models import Sequential
# from keras.preprocessing.image import ImageDataGenerator
# from sklearn.metrics import classification_report, confusion_matrix
# from sklearn.model_selection import train_test_split


# def read_file(path):
#     for dirname, _, filenames in os.walk(path):
#         for filename in filenames:
#             print(os.path.join(dirname, filename))

# img_size = 224


# def get_training_data(data_dir):
#     data = []
#     labels = ['NORMAL','PNEUMONIA']
#     for label in labels:
#         path = os.path.join(data_dir, label)
#         class_num = labels.index(label)
#         #sizecut = len(os.listdir(path))/4
#         #countcut = 0
#         for img in os.listdir(path):
#             if random.uniform(0, 1)>=0.25:
#                 continue
#             #countcut +=1
#             #if countcut >= sizecut:
#             #  break
#             try:
#                 img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
#                 resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
#                 data.append([resized_arr, class_num])
#             except Exception as e:
#                 print(e)
#     return np.array(data)

# def clf(path):
#     l = []
#     for i in path:
#         if(i[1] == 1):
#             l.append("Pneumonia(1)")
#         else:
#             l.append("Normal(0)")
#     return l

# train_root  = "./data/chest_xray/chest_xray/train"
# test_root = "./data/chest_xray/chest_xray/test"
# val_root = "./data/chest_xray/chest_xray/val"

# train_n = train_root+'/NORMAL/'
# train_p = train_root+'/PNEUMONIA/'
# test_n  = test_root+'/NORMAL/'
# test_p  = test_root+'/PNEUMONIA/'
# val_n = val_root+'/NORMAL/'
# val_p = val_root+'/PNEUMONIA/'

# train = get_training_data('./data/chest_xray/chest_xray/train')
# test = get_training_data('./data/chest_xray/chest_xray/test')
# val = get_training_data('./data/chest_xray/chest_xray/val')

# def select(train, p):
#     client=[]
#     client.append(np.array(random.sample(train.T[0], p*len(train))))
#     client.append(np.array(random.sample(train.T[1], (1-p)*len(train))))
#     return np.array(client)

# train = select(train, 0.25)
# x_train = []
# y_train = []

# x_val = []
# y_val = []

# x_test = []
# y_test = []

# for feature, label in train:
#     x_train.append(feature)
#     y_train.append(label)

# for feature, label in test:
#     x_test.append(feature)
#     y_test.append(label)

# for feature, label in val:
#     x_val.append(feature)
#     y_val.append(label)

# X = x_train + x_test
# y = y_train + y_test
# x_train, x_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=42)

# x_train = np.array(x_train) / 255
# x_val = np.array(x_val) / 255
# x_test = np.array(x_test) / 255
# x_train = x_train.reshape(-1, img_size, img_size, 1)
# y_train = np.array(y_train)

# x_val = x_val.reshape(-1, img_size, img_size, 1)
# y_val = np.array(y_val)

# x_test = x_test.reshape(-1, img_size, img_size, 1)
# y_test = np.array(y_test)

# datagen = ImageDataGenerator(
#         featurewise_center=False,  # set input mean to 0 over the dataset
#         samplewise_center=False,  # set each sample mean to 0
#         featurewise_std_normalization=False,  # divide inputs by std of the dataset
#         samplewise_std_normalization=False,  # divide each input by its std
#         zca_whitening=False,  # apply ZCA whitening
#         rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
#         zoom_range = 0.2, # Randomly zoom image
#         width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
#         height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
#         horizontal_flip = True,  # randomly flip images
#         vertical_flip=False)  # randomly flip images


# datagen.fit(x_train)

# model = Sequential()
# model.add(Conv2D(32 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (224,224,1)))
# model.add(BatchNormalization())
# model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
# model.add(Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
# model.add(Dropout(0.1))
# model.add(BatchNormalization())
# model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
# model.add(Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
# model.add(BatchNormalization())
# model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
# model.add(Conv2D(128 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
# model.add(Dropout(0.2))
# model.add(BatchNormalization())
# model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
# model.add(Conv2D(256 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
# model.add(Dropout(0.2))
# model.add(BatchNormalization())
# model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
# model.add(Flatten())
# model.add(Dense(units = 128 , activation = 'relu'))
# model.add(Dropout(0.2))
# model.add(Dense(units = 1 , activation = 'sigmoid'))
# model.compile(optimizer = "rmsprop" , loss = 'binary_crossentropy' , metrics = ['accuracy'])
# model.summary()
# learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.3, min_lr=0.000001)

# class CFDClient(fl.client.NumPyClient):
#     def get_parameters(self, config):
#         return model.get_weights()

#     def fit(self, parameters, config):
#         model.set_weights(parameters)
#         #model.fit(x_train, y_train, epochs=1, batch_size=32, steps_per_epoch=3)
#         model.fit(datagen.flow(x_train,y_train, batch_size = 16) ,epochs = 2 , validation_data = datagen.flow(x_val, y_val) ,callbacks = [learning_rate_reduction])
#         return model.get_weights(), len(x_train), {}

#     def evaluate(self, parameters, config):
#         model.set_weights(parameters)
#         loss, accuracy = model.evaluate(x_test, y_test)
#         return loss, len(x_test), {"accuracy": float(accuracy)}

# fl.client.start_numpy_client(server_address="localhost:8080", client=CFDClient())

import os
import random

import cv2
import flwr as fl
import keras
import numpy as np
import pandas as pd
from keras.callbacks import ReduceLROnPlateau
from keras.layers import (BatchNormalization, Conv2D, Dense, Dropout, Flatten,
                          MaxPool2D)
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def read_file(path):
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            print(os.path.join(dirname, filename))
img_size = 224
def get_training_data(data_dir):
    data = []
    labels = ['NORMAL','PNEUMONIA']

    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        # sizecut = len(os.listdir(path))/4
        # countcut = 0
        for img in os.listdir(path):
            # countcut +=1
            # if countcut >= sizecut:
            #  break
            if random.uniform(0, 1)>=0.20:
                continue
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data)

def clf(path):
    l = []
    for i in path:
        if(i[1] == 1):
            l.append("Pneumonia(1)")
        else:
            l.append("Normal(0)")
    return l


train_root  = "./data/chest_xray/chest_xray/train"
test_root = "./data/chest_xray/chest_xray/test"
val_root = "./data/chest_xray/chest_xray/val"

train_n = train_root+'/NORMAL/'
train_p = train_root+'/PNEUMONIA/'
test_n  = test_root+'/NORMAL/'
test_p  = test_root+'/PNEUMONIA/'
val_n = val_root+'/NORMAL/'
val_p = val_root+'/PNEUMONIA/'

train = get_training_data('./data/chest_xray/chest_xray/train')
test = get_training_data('./data/chest_xray/chest_xray/test')
val = get_training_data('./data/chest_xray/chest_xray/val')

x_train = []
y_train = []

x_val = []
y_val = []

x_test = []
y_test = []

for feature, label in train:
    x_train.append(feature)
    y_train.append(label)

for feature, label in test:
    x_test.append(feature)
    y_test.append(label)

for feature, label in val:
    x_val.append(feature)
    y_val.append(label)

X = x_train + x_test
y = y_train + y_test
x_train, x_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=42)

x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255
x_test = np.array(x_test) / 255
x_train = x_train.reshape(-1, img_size, img_size, 1)
y_train = np.array(y_train)

x_val = x_val.reshape(-1, img_size, img_size, 1)
y_val = np.array(y_val)

x_test = x_test.reshape(-1, img_size, img_size, 1)
y_test = np.array(y_test)

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = True,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(x_train)

model = Sequential()
model.add(Conv2D(32 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (224,224,1)))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(128 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(256 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Flatten())
model.add(Dense(units = 128 , activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(units = 1 , activation = 'sigmoid'))
model.compile(optimizer = "rmsprop" , loss = 'binary_crossentropy' , metrics = ['accuracy'])
model.summary()
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.3, min_lr=0.000001)

class CFDClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        #model.fit(x_train, y_train, epochs=1, batch_size=32, steps_per_epoch=3)
        model.fit(datagen.flow(x_train,y_train, batch_size = 16) ,epochs = 2 , validation_data = datagen.flow(x_val, y_val) ,callbacks = [learning_rate_reduction])
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": float(accuracy)}

fl.client.start_numpy_client(server_address="localhost:8082", client=CFDClient())
