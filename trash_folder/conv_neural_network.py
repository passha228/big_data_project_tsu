from base_neural_network import NeuralNetwork

import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D, add, GlobalAveragePooling2D, Dropout, Dense

class ConvNeuralNetwork(NeuralNetwork):
    def buildModel(self, inputShape, outputSize):
        """
        Метод, который совершает сборку модели
        
        inputShape: входной размер изображения, представить в виде одномерного массива numpy
        outputSize: кол-во классов изображений
        """
        
        inputs = keras.Input(shape=inputShape, name="img")
        x = Conv2D(64, (7, 7), activation="relu")(inputs)
        block_1_output = MaxPooling2D(2)(x)

        x = Conv2D(64, (3, 3), activation="relu")(block_1_output)
        x = Conv2D(64, (3, 3), activation="relu")(x)
        x = Conv2D(64, (3, 3), activation="relu")(x)
        x = Conv2D(64, (3, 3), activation="relu")(x)
        x = Conv2D(64, (3, 3), activation="relu")(x)
        x = Conv2D(64, (3, 3), activation="relu")(x)
        block_2_output = MaxPooling2D(2)(x)

        x = Conv2D(128, (3, 3), activation="relu")(block_2_output)
        x = Conv2D(128, (3, 3), activation="relu")(x)
        x = Conv2D(128, (3, 3), activation="relu")(x)
        x = Conv2D(128, (3, 3), activation="relu")(x)
        x = Conv2D(128, (3, 3), activation="relu")(x)
        x = Conv2D(128, (3, 3), activation="relu")(x)
        x = Conv2D(128, (3, 3), activation="relu")(x)
        x = Conv2D(128, (3, 3), activation="relu")(x)
        block_3_output = MaxPooling2D(2)(x)
        
        x = Conv2D(256, (3, 3), activation="relu")(block_3_output)
        x = Conv2D(256, (3, 3), activation="relu")(x)
        x = Conv2D(256, (3, 3), activation="relu")(x)
        x = Conv2D(256, (3, 3), activation="relu")(x)
        x = Conv2D(256, (3, 3), activation="relu")(x)
        x = Conv2D(256, (3, 3), activation="relu")(x)
        x = Conv2D(256, (3, 3), activation="relu")(x)
        x = Conv2D(256, (3, 3), activation="relu")(x)
        x = Conv2D(256, (3, 3), activation="relu")(x)
        x = Conv2D(256, (3, 3), activation="relu")(x)
        x = Conv2D(256, (3, 3), activation="relu")(x)
        x = Conv2D(256, (3, 3), activation="relu")(x)
        x = Conv2D(256, (3, 3), activation="relu")(x)
        block_4_output = MaxPooling2D(2)(x)

        x = Conv2D(512, (3, 3), activation="relu")(block_4_output)
        x = Conv2D(512, (3, 3), activation="relu")(x)
        x = Conv2D(512, (3, 3), activation="relu")(x)
        x = Conv2D(512, (3, 3), activation="relu")(x)
        x = Conv2D(512, (3, 3), activation="relu")(x)
        x = Conv2D(512, (3, 3), activation="relu")(x)
        
        x = GlobalAveragePooling2D()(x)
        x = Dense(1000, activation="relu")(x)
        outputs = Dense(outputSize)(x)

        self.model = keras.Model(inputs, outputs)