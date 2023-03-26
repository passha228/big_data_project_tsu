import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

import tkinter.filedialog as tfd

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from conv_neural_network import ConvNeuralNetwork
from keras import optimizers

def func():
    (x_train,y_train), (x_test,y_test) = cifar10.load_data()
    x_train = x_train / 255
    x_test = x_test / 255

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    p = ConvNeuralNetwork(optimizer = keras.optimizers.Adam(0.001), loss = "categorical_crossentropy", epochs= 3)
    p.buildModel((32, 32, 3), 10)
    p.compileModel(x_train, y_train)
    

    p.saveModelTo(os.path.join(os.getcwd(), 'conv.h5'))
    

def func2():
    conv = ConvNeuralNetwork(None, None, 0)
    conv.loadModelFromFile(os.path.join(os.getcwd(), 'conv.h5'))
    fileName = tfd.askopenfilename()
    file = cv2.imread(fileName)
    img = cv2.resize(file,(32,32))
    img = np.expand_dims(img, axis=0)
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    pred = conv.predict(img)
    print(classes[np.argmax(pred)])
    print(pred)


def func3():
    (x_train,y_train), (x_test,y_test) = cifar10.load_data()
    # Отображение первых 25 изображений
    plt.figure(figsize=(10, 5))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks()
        plt.yticks()
        plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.show()


func3()
