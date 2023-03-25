from base_neural_network import NeuralNetwork

from keras.layers import Dense, Flatten
from keras import Model

class Perseptron(NeuralNetwork):
    def buildModel(self, inputSize, outputSize):
        """
        Метод, который совершает сборку модели
        
        inputSize: входной размер изображения, представить в виде одномерного массива numpy
        outputSize: кол-во классов изображений
        """
        super().buildModel(inputSize, outputSize)
        # self.model.add(Flatten(input_shape = inputSize))
        x = Dense(256, activation='relu')(self.inputs)
        Dense(128, activation='relu')(x)
        Dense(64, activation='relu')(x)
        Dense(32, activation='relu')(x)

        Dense(outputSize, activation='softmax')(x)
        self.model = Model(self.inputs, x)