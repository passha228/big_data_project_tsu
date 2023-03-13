from base_neural_network import NeuralNetwork

import tensorflow as tf
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from keras.models import Sequential

class ConvNeuralNetwork(NeuralNetwork):
    #TODO не помню что такое input_shape
    def buildModel(self, inputSize, outputSize):
        super().buildModel(inputSize)
        self.model.add(Conv2D(64, (3, 3), padding='same', 
                        input_shape=(224, 224, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))        
        self.model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(outputSize, activation='softmax'))