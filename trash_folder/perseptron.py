from base_neural_network import NeuralNetwork

from keras.layers import Dense, Flatten
from keras import Model, Input

class Perseptron(NeuralNetwork):
    def buildModel(self, inputSize, outputSize):
        """
        Метод, который совершает сборку модели
        
        inputSize: входной размер изображения, представить в виде одномерного массива numpy
        outputSize: кол-во классов изображений
        """
        # super().buildModel(inputSize, outputSize)
        inputs = Input(shape=inputSize, name="img")
        # self.model = Flatten(input_shape = inputSize)
        x = Dense(256, activation='relu')(inputs)
        Dense(128, activation='relu')(x)
        Dense(64, activation='relu')(x)
        Dense(32, activation='relu')(x)

        outputs = Dense(outputSize, activation='softmax')(x)
        self.model = Model(inputs, outputs)