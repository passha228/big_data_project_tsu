import pandas as pd
import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10

from conv_neural_network import ConvNeuralNetwork

def func():
    (x_train,y_train), (x_test,y_test) = cifar10.load_data()
    x_train = x_train / 255
    x_test = x_test / 255

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    #y_train = y_train.ravel()
    print(y_train.shape)
    p = ConvNeuralNetwork(optimizer = keras.optimizers.Adam(0.1), loss = "categorical_crossentropy", epochs= 4)
    p.buildModel((32, 32, 3), 10)
    p.compileModel(x_train, y_train)
    print(p.validate(x_test,y_test))
    
func()
