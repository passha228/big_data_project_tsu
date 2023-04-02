import tensorflow as tf
import keras
import cv2
import numpy as np


from tensorflow.keras import applications, Input
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.metrics import SpecificityAtSensitivity, Recall, CategoricalAccuracy

import os
import subprocess

#DENSNET

def perseptron() ->keras.Model:
    input_layer = Input(shape = (32, 32, 1))
    x = Dense(128, activation = 'relu')(input_layer)
    x1 = Dense(64, activation = 'relu')(x)
    x2 = Dense(32, activation = 'relu')(x1)
    x3 = Dense(16, activation = 'relu')(x2)
    flatten = Flatten()(x3)
    output = Dense(10, activation = 'softmax')(flatten)
    return keras.Model(inputs = input_layer, outputs = output)


# UNIT_TESTS_DONE
def create_vgg() -> keras.Model:
    model = applications.VGG16(classes = 10, input_shape = (32, 32, 3), include_top = False)
    x = Flatten()(model.output)
    x = Dense(10, activation = 'softmax')(x)

    return keras.Model(inputs = model.input, outputs = x)


# UNIT_TESTS_DONE
def create_InceptionV3() -> keras.Model:
    model = applications.VGG16(classes = 10, input_shape = (32, 32, 3), include_top = False)
    x = Flatten()(model.output)
    x = Dense(10, activation = 'softmax')(x)

    return keras.Model(inputs = model.input, outputs = x)


class Training:
    def __init__(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()

        self.x_train = self.x_train / 255
        self.x_test = self.x_test / 255
        self.y_train = keras.utils.to_categorical(self.y_train, 10)
        self.y_test = keras.utils.to_categorical(self.y_test, 10)

    def train_save(self, model):
        # Обучение
        #TODO: 
        model.compile(optimizer = keras.optimizers.Adam(0.05), 
                      loss = 'categorical_crossentropy', 
                      metrics = [CategoricalAccuracy(), SpecificityAtSensitivity(0.5), Recall()])
        print(model.summary())

        with tf.device('/GPU:0'):
             history = model.fit(self.x_train, self.y_train, epochs = 10, batch_size = 256)
             print(history)

        # сохранение
        nn_trained_path = os.path.join(os.getcwd(), 'nn_models', model.name + '.h5')
        model.save(nn_trained_path)

    def train_save_grayscale(self, model):
        self.x_train_grayscale = tf.image.rgb_to_grayscale(self.x_train)
        print(self.x_train_grayscale.shape)
        
        model.compile(optimizer = keras.optimizers.Adam(0.05), 
                      loss = 'categorical_crossentropy', 
                      metrics = [CategoricalAccuracy(), SpecificityAtSensitivity(0.65), Recall()])
        print(model.summary())

        with tf.device('/GPU:0'):
             history = model.fit(self.x_train_grayscale, self.y_train, epochs = 10, batch_size = 256)
             print(history)

        # сохранение
        nn_trained_path = os.path.join(os.getcwd(), 'nn_models', model.name + '.h5')
        model.save(nn_trained_path)
        
    
    def train_save_all(self):
        self.train_save(create_vgg())
        self.train_save(create_InceptionV3())
        self.train_save_grayscale(perseptron())

if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Разрешаем рост памяти GPU по мере необходимости
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # Ошибка при конфигурации GPU
            print(e)
    print(gpus)
    trainer = Training()
    trainer.train_save_all()