import numpy as np
import matplotlib.pyplot as plt
import cv2

import tkinter.filedialog as tfd

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from conv_neural_network import ConvNeuralNetwork
from perseptron import Perseptron
from conv_neural_network import ConvNeuralNetwork
from keras import optimizers

def train_save_conv():
    (x_train,y_train), (x_test,y_test) = cifar10.load_data()
    x_train = x_train / 255
    x_test = x_test / 255

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    p = ConvNeuralNetwork(optimizer = keras.optimizers.Adam(0.5), loss = "categorical_crossentropy", epochs= 10)
    p.buildModel((32, 32, 3), 10)
    #history = 
    p.compileModel(x_train, y_train)
    #plt.plot(history['loss'])
    #plt.plot(history['val_loss'])
    #plt.title('conv loss')
    #plt.ylabel('Loss')
    #plt.xlabel('Epoch')
    #plt.legend(['Train', 'Validation'], loc='upper left')
    #plt.show()
    
    #plt.plot(history['accuracy'])
    #plt.plot(history['val_accuracy'])
    #plt.title('conv accuracy')
    #plt.ylabel('Accuracy')
    #plt.xlabel('Epoch')
    #plt.legend(['Train', 'Validation'], loc='upper left')
    #plt.show()
    p.saveModelTo(os.path.join(os.getcwd(), 'conv_v2.h5'))
    

def check_conv():
    conv = ConvNeuralNetwork(None, None, 0)
    conv.loadModelFromFile(os.path.join(os.getcwd(), 'conv_v2.h5'))
    fileName = tfd.askopenfilename()
    file = cv2.imread(fileName)
    img = cv2.resize(file,(32,32))
    img = np.expand_dims(img, axis=0)
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    pred = conv.predict(img)
    print(classes[np.argmax(pred)])
    print(pred)


def see_first_5():
    (x_train,y_train), (x_test,y_test) = cifar10.load_data()
    # Отображение первых 25 изображений
    plt.figure(figsize=(10, 5))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks()
        plt.yticks()
        plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.show()


func2()
