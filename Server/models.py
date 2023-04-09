import tensorflow as tf
import keras
import cv2
import numpy as np
import util

from tensorflow.keras import applications

class NN:
    model = None
    input_shape = None
    name = None

    def predict(self):
        return np.argmax(self.model.predict(self.img))
    
    def take_image(self, img_path: str):
        """
        img_path: Абсолютный путь
        """
        self.img = cv2.imread(img_path)
        self.img = cv2.resize(self.img, self.input_shape)
        self.img = np.expand_dims(self.img, axis=0)

    def getName(self):
        return self.name;
    
class DenseNet121(NN):
    def __init__(self) -> None:
        self.model = applications.DenseNet121()
        self.input_shape = (224, 224)
        self.name = 'DenseNet121'


class VGG19(NN):
    def __init__(self) -> None:
        self.model = applications.VGG19()
        self.input_shape = (224, 224)
        self.name = 'VGG19'


def getResults(path):
    predicts = []
    modelVGG = VGG19()
    modelVGG.take_image(path)
    predicts.append((modelVGG.getName(), util.take_name_of_predict(modelVGG.predict())))
    modelDenseNet121 = DenseNet121()
    modelDenseNet121.take_image(path)
    predicts.append((modelDenseNet121.getName(), util.take_name_of_predict(modelDenseNet121.predict())))
    return predicts
