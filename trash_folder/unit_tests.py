from neural_network.utils import *

import tensorflow as tf
import keras
from tensorflow.keras.datasets import cifar10

import unittest


class Testing(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()

        self.x_train = self.x_train / 255
        self.x_test = self.x_test / 255
        self.y_train = keras.utils.to_categorical(self.y_train, 10)
        self.y_test = keras.utils.to_categorical(self.y_test, 10)
        

    def test_evaluate(self):
        model = create_ResNetRS350()
        model.compile(optimizer = keras.optimizers.Adam(0.1), loss = 'categorical_crossentropy')
        model.fit(self.x_train, self.y_train, epochs = 10)
        model.evaluate(self.x_test, self.y_test)

if __name__ == '__main__':
    unittest.main()