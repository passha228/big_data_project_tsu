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
        x = Conv2D(32, 3, activation="relu")(inputs)
        x = Conv2D(64, 3, activation="relu")(x)
        block_1_output = MaxPooling2D(3)(x)

        x = Conv2D(64, 3, activation="relu", padding="same")(block_1_output)
        x = Conv2D(64, 3, activation="relu", padding="same")(x)
        block_2_output = add([x, block_1_output])

        x = Conv2D(64, 3, activation="relu", padding="same")(block_2_output)
        x = Conv2D(64, 3, activation="relu", padding="same")(x)
        block_3_output = add([x, block_2_output])

        x = Conv2D(64, 3, activation="relu")(block_3_output)
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation="relu")(x)
        outputs = Dense(outputSize)(x)

        self.model = keras.Model(inputs, outputs)