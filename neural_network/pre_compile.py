import pandas as pd
import numpy as np
import cw2
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
import perseptron

def func():
    (x_train,y_train), (x_test,y_test) = cifar10.load_data()

    x_train = x_train/255
    y_train = y_train/255

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    p = perseptron.Perseptron(optimizer = keras.optimizers.Adam(0.1), loss = "mean_squared_error", epochs= 4)
    p.buildModel(32*32*3, 10)
    p.compileModel(x_train, y_train)
    p.validate(x_test,y_test)
    
func()
