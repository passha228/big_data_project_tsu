from base_neural_network import NeuralNetwork

from keras.layers import Dense

class Perseptron(NeuralNetwork):
    def buildModel(self, inputSize, outputSize):
        """
        Метод, который совершает сборку модели
        
        inputSize: входной размер изображения, представить в виде одномерного массива numpy
        outputSize: кол-во классов изображений
        """
        super().buildModel(inputSize, outputSize)
        self.model.add(Dense(inputSize, activation='relu'))
        self.model.add(Dense(int(inputSize / 2), activation='relu'))
        self.model.add(Dense(int(inputSize / 4), activation='relu'))
        self.model.add(Dense(int(inputSize / 8), activation='relu'))

        self.model.add(Dense(outputSize, activation='softmax'))